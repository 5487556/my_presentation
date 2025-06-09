# server.py（精簡示範）
import os
import re
import json
import base64
import asyncio
import numpy as np
import tensorflow as tf
import cv2
import websockets
from urllib.parse import quote
from mtcnn import MTCNN
from keras_facenet import FaceNet
from annoy import AnnoyIndex

# ------------------- 1. 調整 BASE_PATH 的設置 -------------------
# 優先讀環境變數 RENDER_DB_PATH（由 Render Persistent Disk 指定），否則 fallback 到相對路徑 "./data"
BASE_PATH = os.getenv("RENDER_DB_PATH", os.path.join(os.getcwd(), "data", "FaceAuthSystem"))

# 既然要寫入，就一次建立好需要的子目錄
for sub in ("database", "models", "test_images"):
    os.makedirs(os.path.join(BASE_PATH, sub), exist_ok=True)

ALLOWED_ID_PATTERN = r'^[a-zA-Z0-9_-]{4,20}$'
SAVE_IMAGE_FORMAT = 'png'
MIN_FACE_CONFIDENCE = 0.95
FACE_SIZE = (160, 160)
INDEX_TREES = 15

def validate_user_id(uid):
    if not re.match(ALLOWED_ID_PATTERN, uid):
        raise ValueError(f"無效用戶ID: {uid}")
    return uid

def safe_filename(uid):
    return quote(uid, safe='')

# ------------------- 2. FaceRegister / FaceRecognizer 皆使用相同的 BASE_PATH -------------------
class FaceRegister:
    def __init__(self):
        self.detector = MTCNN()
        self.embedder = FaceNet()

    def _process_face(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片找不到：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        if not faces:
            raise ValueError("沒偵測到人臉")
        main = max(faces, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            raise ValueError(f"人臉品質不夠：{main['confidence']:.2f}")

        x, y, w, h = (max(0, v) for v in main['box'])
        face_roi = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return face_roi, face_img

    def register(self, img_path, user_id, debug=False):
        try:
            uid = validate_user_id(user_id)
            safe_id = safe_filename(uid)

            _, face_img = self._process_face(img_path)
            emb = self.embedder.embeddings([face_img])[0]

            # 存到 BASE_PATH/database
            np.save(os.path.join(BASE_PATH, "database", f"{safe_id}.npy"), emb)
            cv2.imwrite(
                os.path.join(BASE_PATH, "database", f"{safe_id}.{SAVE_IMAGE_FORMAT}"),
                cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            print(f"✅ 註冊成功：{uid}")
            return True
        except Exception as e:
            print(f"❌ 註冊失敗：{e}")
            return False

class FaceRecognizer:
    def __init__(self, threshold=0.5):
        # 啟用 GPU（如果有的話）
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

        # 載入已存的 .npy 特徵
        self.user_ids = []
        self.embeddings = []
        db_dir = os.path.join(BASE_PATH, "database")
        for fname in os.listdir(db_dir):
            if fname.endswith(".npy"):
                uid = os.path.splitext(fname)[0]
                emb = np.load(os.path.join(db_dir, fname))
                if emb.shape[0] != 512:
                    continue
                self.user_ids.append(uid)
                self.embeddings.append(emb)

        if len(self.embeddings) == 0:
            print("⚠️ 資料庫尚未註冊任何使用者，辨識會一直失敗。")

        # 建立 Annoy Index（就算 embeddings 為空也不會當機）
        self.index = AnnoyIndex(512, 'angular')
        for i, e in enumerate(self.embeddings):
            self.index.add_item(i, e)
        self.index.build(INDEX_TREES)
        print(f"✅ Index 已建，總共 {self.index.get_n_items()} 筆特徵")

    def _recognize_embedding(self, emb):
        idxs, dists = self.index.get_nns_by_vector(emb, 3, include_distances=True)
        if not idxs:
            return None, 0.0
        best_idx, best_dist = idxs[0], dists[0]
        sim = 1.0 - (best_dist ** 2) / 2.0
        if sim >= self.threshold:
            return self.user_ids[best_idx], sim
        else:
            return None, sim

    def recognize_from_image(self, rgb_img, debug=False):
        faces = self.detector.detect_faces(rgb_img)
        if not faces:
            return None, 0.0
        main = max(faces, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            return None, float(main['confidence'])
        x, y, w, h = (max(0, v) for v in main['box'])
        face_roi = rgb_img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        emb = self.embedder.embeddings([face_img])[0]
        return self._recognize_embedding(emb)

    def recognize_from_file(self, img_path, debug=False):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到圖片：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.recognize_from_image(rgb, debug)

# ------------------- 3. 全域 recognizer 實例 -------------------
recognizer = FaceRecognizer(threshold=0.5)

# ------------------- 4. WebSocket Handler -------------------
async def handler(ws, path):
    # 先回傳 ready 訊息
    await ws.send(json.dumps({"status": "ready", "message": "Server Ready"}))

    async for raw in ws:
        try:
            msg = json.loads(raw)
            t = msg.get("type", "")
            if t == "file":
                img_path = msg.get("path", "")
                user, sim = recognizer.recognize_from_file(img_path, debug=False)
            elif t == "base64":
                b64str = msg.get("data", "")
                if "," in b64str:
                    b64str = b64str.split(",", 1)[1]
                data = base64.b64decode(b64str)
                arr = np.frombuffer(data, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Base64 解碼失敗")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                user, sim = recognizer.recognize_from_image(rgb, debug=False)
            else:
                await ws.send(json.dumps({"status": "fail", "reason": "不支援的 type"}))
                continue

            if user:
                resp = {"status": "ok", "user": user, "similarity": round(sim, 4)}
            else:
                reason = "未檢測到人臉" if sim == 0.0 else "相似度不足"
                resp = {"status": "fail", "reason": reason, "similarity": round(sim, 4)}
            await ws.send(json.dumps(resp))
        except Exception as e:
            await ws.send(json.dumps({"status": "error", "message": str(e)}))

async def main():
    # 讓 PORT 從環境變數取得，如果沒設就 fallback 8765
    port = int(os.getenv("PORT", "8765"))
    server = await websockets.serve(handler, "0.0.0.0", port)
    print(f"✅ WebSocket Server 已啟動，監聽 ws://0.0.0.0:{port}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
