# -*- coding: utf-8 -*-
import os
import re
import cv2
import json
import base64
import asyncio
import numpy as np
import tensorflow as tf
from urllib.parse import quote
from mtcnn import MTCNN
from keras_facenet import FaceNet
from annoy import AnnoyIndex
import websockets

# =============================================================================
# 1. 全域參數與輔助函式
# =============================================================================
BASE_PATH = os.getenv('FACE_AUTH_PATH', '/content/drive/MyDrive/FaceAuthSystem')
ALLOWED_ID_PATTERN = r'^[a-zA-Z0-9_-]{4,20}$'
SAVE_IMAGE_FORMAT = 'png'
MIN_FACE_CONFIDENCE = 0.95
FACE_SIZE = (160, 160)
INDEX_TREES = 15

# 確保資料夾存在
for subdir in ['database', 'models', 'test_images']:
    os.makedirs(os.path.join(BASE_PATH, subdir), exist_ok=True)

def validate_user_id(user_id):
    """用戶ID合法性驗證：4-20位，允許大小寫字母、數字、_、-"""
    if not re.match(ALLOWED_ID_PATTERN, user_id):
        raise ValueError(f"無效用戶ID: {user_id}，格式要求 4-20 位字母/數字/下划線/減號")
    return user_id

def safe_filename(user_id):
    """將 user_id 編碼成安全的檔名（URL encode）"""
    return quote(user_id, safe='')

# =============================================================================
# 2. 人臉註冊模組（FaceRegister）
# =============================================================================
class FaceRegister:
    def __init__(self):
        self.detector = MTCNN()
        # 這裡使用 keras-facenet 的預設模型
        self.embedder = FaceNet()

    def _process_face(self, img_path):
        """從檔案路徑讀圖，檢測人臉，回傳裁剪與 resize 後的人臉"""
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片路徑不存在：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        if not results:
            raise ValueError("未檢測到人臉")

        main = max(results, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            raise ValueError(f"人臉品質不足 (confidence={main['confidence']:.2f})")

        x, y, w, h = (max(0, v) for v in main['box'])
        face_roi = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return face_roi, face_img

    def register(self, img_path, user_id, debug=False):
        """
        將指定影像註冊為 user_id：
        1. 檢測人臉、裁剪、resize
        2. 取得 embedding
        3. 存成 npy 與一張裁剪後的人臉圖
        """
        try:
            user_id = validate_user_id(user_id)
            safe_id = safe_filename(user_id)

            _, face_img = self._process_face(img_path)
            embedding = self.embedder.embeddings([face_img])[0]  # shape = (512,)

            # 儲存 embedding npy
            np.save(os.path.join(BASE_PATH, 'database', f"{safe_id}.npy"), embedding)
            # 同時把裁切後的人臉存成圖檔（.png）
            out_img_path = os.path.join(BASE_PATH, 'database', f"{safe_id}.{SAVE_IMAGE_FORMAT}")
            cv2.imwrite(out_img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR), 
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f"✅ 註冊成功：{user_id}，embedding 大小 {embedding.shape}")
            return True
        except Exception as e:
            print(f"❌ 註冊失敗：{e}")
            return False

# =============================================================================
# 3. 人臉辨識模組（FaceRecognizer）
# =============================================================================
class FaceRecognizer:
    def __init__(self, threshold=0.5):
        # 嘗試啟用 GPU 加速（如果有 GPU）
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU 加速已啟用")
            except RuntimeError as e:
                print(f"⚠️ GPU 設定失敗：{e}")

        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.threshold = threshold

        # 載入資料庫中的所有 embedding
        self._init_database()
        # 建立 Annoy Index
        self._build_index()

    def _init_database(self):
        """從 BASE_PATH/database 讀入所有 .npy 特徵檔，並存到 self.embeddings、self.user_ids"""
        self.user_ids = []
        self.embeddings = []
        db_dir = os.path.join(BASE_PATH, 'database')
        if not os.path.isdir(db_dir):
            raise FileNotFoundError(f"資料庫目錄不存在：{db_dir}")

        for fname in os.listdir(db_dir):
            if fname.endswith('.npy'):
                uid = os.path.splitext(fname)[0]
                emb = np.load(os.path.join(db_dir, fname))
                if emb.shape[0] != 512:
                    print(f"⚠️ 忽略 {fname}，維度不符：{emb.shape}")
                    continue
                self.user_ids.append(uid)
                self.embeddings.append(emb)

        if len(self.embeddings) == 0:
            raise ValueError("資料庫為空或特徵檔有誤，請先註冊用戶")
        print(f"✅ 已載入 {len(self.embeddings)} 個 使用者特徵")

    def _build_index(self):
        """用 Annoy 建立 angular index"""
        self.index = AnnoyIndex(512, 'angular')
        for idx, emb in enumerate(self.embeddings):
            self.index.add_item(idx, emb)
        self.index.build(INDEX_TREES)
        print(f"✅ Index 建立完成，共 {self.index.get_n_items()} 筆")

    def _recognize_embedding(self, emb):
        """
        在 Annoy Index 搜尋最近鄰，回傳 (user_id, similarity)：
        similarity 由 angular distance → cosine estimate：1 - dist^2/2
        """
        idxs, dists = self.index.get_nns_by_vector(emb, 3, include_distances=True)
        if not idxs:
            return None, 0.0
        best_idx, best_dist = idxs[0], dists[0]
        sim = 1.0 - (best_dist ** 2) / 2.0
        if sim >= self.threshold:
            return self.user_ids[best_idx], sim
        else:
            return None, sim

    def recognize_from_image(self, rgb_img: np.ndarray, debug=False):
        """
        直接傳入一張 RGB np.ndarray 圖片 → 偵測人臉 → 取得 embedding → 比對
        回傳 (user_id or None, similarity)
        """
        results = self.detector.detect_faces(rgb_img)
        if not results:
            return None, 0.0

        main = max(results, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            return None, float(main['confidence'])

        x, y, w, h = (max(0, v) for v in main['box'])
        face_roi = rgb_img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)

        if debug:
            # 如果需要可視化
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1); plt.imshow(face_roi); plt.title('原始 ROI')
            plt.subplot(1,2,2); plt.imshow(face_img); plt.title('裁切並縮放 160x160')
            plt.show()

        emb = self.embedder.embeddings([face_img])[0]
        return self._recognize_embedding(emb)

    def recognize_from_file(self, img_path, debug=False):
        """
        讀檔再呼叫 recognize_from_image
        """
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片不存在：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.recognize_from_image(rgb, debug)

# =============================================================================
# 4. 建立全域 recognizer 實例（在 WebSocket handler 外部，只建立一次）
# =============================================================================
recognizer = FaceRecognizer(threshold=0.5)

# =============================================================================
# 5. WebSocket Server：接收「檔案路徑」或「Base64 圖片」，回傳辨識結果
# =============================================================================
async def handler(websocket, path):
    """
    前端傳入 JSON 格式：
      1. {"type":"file",   "path":"/absolute/path/to/image.jpg"}
      2. {"type":"base64", "data":"data:image/png;base64,iVBORw0KGgoAAAANS..."}
    回傳 JSON：
      {"status":"ok",   "user":"harrylin", "similarity":0.87}
      {"status":"fail", "reason":"未檢測到人臉", "similarity":0.0000}
      {"status":"error","message":"Exception 訊息文字"}
    """
    print("≥ 客戶端已連線")
    # 可以先回傳一個 ready 訊息
    await websocket.send(json.dumps({"status": "ready", "message": "Server Ready"}))

    async for raw in websocket:
        try:
            msg = json.loads(raw)
            img_type = msg.get("type", "")

            if img_type == "file":
                img_path = msg.get("path", "")
                user, sim = recognizer.recognize_from_file(img_path, debug=False)

            elif img_type == "base64":
                b64_str = msg.get("data", "")
                # 可能帶有 "data:image/...;base64," 前綴，需要去掉
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Base64 解碼失敗")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                user, sim = recognizer.recognize_from_image(rgb, debug=False)

            else:
                # 不支援的 type
                await websocket.send(json.dumps({
                    "status": "fail",
                    "reason": "不支援的 type"
                }))
                continue

            # 根據比對結果回傳
            if user:
                resp = {
                    "status": "ok",
                    "user": user,
                    "similarity": round(sim, 4)
                }
            else:
                # 若沒有偵測到人臉，就 sim=0；若有偵測到人臉但 sim < threshold，sim 仍是 [0, threshold)
                reason_text = "未檢測到人臉" if sim == 0.0 else "相似度不足"
                resp = {
                    "status": "fail",
                    "reason": reason_text,
                    "similarity": round(sim, 4)
                }
            await websocket.send(json.dumps(resp))
            print("回傳 →", resp)

        except Exception as e:
            # 捕捉所有例外，並回傳錯誤訊息
            traceback_str = str(e)
            await websocket.send(json.dumps({
                "status": "error",
                "message": traceback_str
            }))
            continue

async def main():
    # 在 localhost:8765 啟動 WebSocket Server，你可以自行調整 port
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("WebSocket Server 已啟動，監聽 ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    # 以 asyncio.run 執行主程序
    asyncio.run(main())
