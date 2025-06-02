# ───────────────────────────────────────────────────────────────────────────────────────────────────────
import asyncio
import websockets
import base64
import os
import cv2
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet
from annoy import AnnoyIndex
# （請確保已安裝好以下套件：tensorflow==2.12.0、keras-facenet、mtcnn、deepface、annoy 等）

# -------------------- 以下為人臉辨識相關程式（與你原本的 FaceRegister/FaceRecognizer 類似） --------------------

BASE_PATH = '/content/drive/MyDrive/FaceAuthSystem'
MIN_FACE_CONFIDENCE = 0.95
FACE_SIZE = (160, 160)
INDEX_TREES = 10

class FaceUtils:
    @staticmethod
    def show_image_comparison(raw_img, processed_img):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.subplot(121); plt.imshow(raw_img); plt.title('原始人臉 ROI')
        plt.subplot(122); plt.imshow(processed_img); plt.title('縮放後 (160x160)')
        plt.show()

class FaceRecognizer:
    def __init__(self, threshold=0.5):
        """ 初始化人臉識別組件 """
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU 加速已啟用")
            except RuntimeError as e:
                print(f"⚠️ GPU 設定失敗：{e}")

        self.detector = MTCNN(
            min_face_size=50,
            steps_threshold=[0.6, 0.7, 0.7],
            scale_factor=0.8
        )
        self.embedder = FaceNet()
        self.threshold = threshold

        # 載入已註冊用戶的 embedding
        self._init_database()
        self._build_index()

    def _init_database(self):
        self.user_ids = []
        self.embeddings = []
        db_path = os.path.join(BASE_PATH, 'database')
        if not os.path.isdir(db_path):
            raise FileNotFoundError(f"資料庫目錄不存在：{db_path}")

        for fname in os.listdir(db_path):
            if fname.endswith('.npy'):
                uid = os.path.splitext(fname)[0]
                emb = np.load(os.path.join(db_path, fname))
                if emb.shape[0] != 512:
                    continue
                self.user_ids.append(uid)
                self.embeddings.append(emb)
        if len(self.embeddings) == 0:
            raise ValueError("資料庫為空或檔案錯誤，請先註冊用戶")
        print(f"✅ 已載入 {len(self.embeddings)} 名用戶的 embedding")

    def _build_index(self):
        self.index = AnnoyIndex(512, 'angular')
        for i, e in enumerate(self.embeddings):
            self.index.add_item(i, e)
        self.index.build(INDEX_TREES)
        print(f"✅ 特徵索引建構完成，共 {self.index.get_n_items()} 筆")

    def recognize_from_image(self, img: np.ndarray, debug=False):
        """ 接收一張 RGB np.ndarray 圖片，進行人臉偵測與辨識 """
        results = self.detector.detect_faces(img)
        if not results:
            return None, 0.0  # 找不到人臉

        # 選擇置信度最高的那張
        main = max(results, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            return None, float(main['confidence'])  # 置信度過低

        x, y, w, h = (max(0, v) for v in main['box'])
        face_roi = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)

        if debug:
            FaceUtils.show_image_comparison(face_roi, face_img)

        emb = self.embedder.embeddings([face_img])[0]
        idxs, dists = self.index.get_nns_by_vector(emb, 3, include_distances=True)
        if not idxs:
            return None, 0.0

        # 將 Angular distance 轉成相似度 (cosine estimate)：similarity ≈ 1 - (dist^2)/2
        best_idx, best_dist = idxs[0], dists[0]
        similarity = 1.0 - (best_dist ** 2) / 2.0

        if similarity >= self.threshold:
            return self.user_ids[best_idx], similarity
        else:
            return None, similarity

    def recognize_from_file(self, img_path, debug=False):
        """ 從檔案路徑讀圖，然後呼叫 recognize_from_image """
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片不存在：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.recognize_from_image(rgb, debug)

# -------------------- WebSocket Server: 接收前端傳來的 Base64 或檔案路徑，回傳辨識結果 --------------------

# 先在全域建立一個 recognizer 實例，避免每次 handler 被呼叫都重新載入 model
recognizer = FaceRecognizer(threshold=0.5)

async def handler(websocket):
    """
    支援兩種傳入訊息格式：
    1. JSON 字串：{"type": "file", "path": "/content/drive/xxx.jpg"}
    2. JSON 字串：{"type": "base64", "data": "<Base64 編碼的影像>"}
    回傳格式：
    {"status": "ok", "user": "Zulfiqar_Ahmed", "similarity": 0.87}
    或
    {"status": "fail", "reason": "未檢測到人臉"} / {"status": "fail", "reason": "相似度不足"}
    """
    print("Client connected")
    await websocket.send("Server Ready")

    async for raw in websocket:
        try:
            # 解析 JSON
            import json
            msg = json.loads(raw)

            # 根據 type 決定如何讀影像
            if msg.get("type") == "file":
                img_path = msg.get("path", "")
                user, sim = recognizer.recognize_from_file(img_path, debug=False)
            elif msg.get("type") == "base64":
                b64_data = msg.get("data", "")
                # 去掉「data:image/...;base64,」前綴（若前端帶來）
                if ',' in b64_data:
                    b64_data = b64_data.split(',', 1)[1]
                img_bytes = base64.b64decode(b64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("無法解碼 Base64 影像資料")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                user, sim = recognizer.recognize_from_image(rgb, debug=False)
            else:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "reason": "不支援的 type"
                }))
                continue

            # 根據辨識結果回傳
            if user:
                response = {
                    "status": "ok",
                    "user": user,
                    "similarity": round(sim, 4)
                }
            else:
                reason = "無授權人員" if sim >= 0 else "未檢測到人臉"
                # 若有檢測到人臉但相似度低於門檻，sim 會是 [0, threshold)；若根本沒檢到人臉，sim=0 也可能無法區分，建議前端再做前處理
                response = {
                    "status": "fail",
                    "reason": reason,
                    "similarity": round(sim, 4)
                }
            await websocket.send(json.dumps(response))

        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                "status": "error",
                "message": str(e)
            }))
            continue

async def main():
    start_server = await websockets.serve(handler, "localhost", 8080)
    print("Server running on ws://localhost:8080")
    await start_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
# ───────────────────────────────────────────────────────────────────────────────────────────────────────
