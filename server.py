# server.py

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

# =========================== 1. 全域參數 ===========================
# 用环境变量或当前工作目录决定 BASE_PATH
BASE_PATH = os.getenv("RENDER_DB_PATH", os.path.join(os.getcwd(), "data", "FaceAuthSystem"))
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

# ====================== 2. FaceRegister / FaceRecognizer ======================
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
            # 存 embedding
            np.save(os.path.join(BASE_PATH, "database", f"{safe_id}.npy"), emb)
            # 存裁切後人臉圖
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
        # 嘗試啟用 GPU
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

        # 載入已註冊使用者的 embedding
        self.user_ids = []
        self.embeddings = []
        db_dir = os.path.join(BASE_PATH, "database")
        if not os.path.isdir(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        for fname in os.listdir(db_dir):
            if fname.endswith(".npy"):
                uid = os.path.splitext(fname)[0]
                emb = np.load(os.path.join(db_dir, fname))
                if emb.shape[0] != 512:
                    continue
                self.user_ids.append(uid)
                self.embeddings.append(emb)

        if len(self.embeddings) == 0:
            print("⚠️ 資料庫尚無使用者特徵，辨識會失敗。")

        # 建立 Annoy Index
        self.index = AnnoyIndex(512, 'angular')
        for i, e in enumerate(self.embeddings):
            self.index.add_item(i, e)
        self.index.build(INDEX_TREES)
        print(f"✅ Index 已建立，共 {self.index.get_n_items()} 筆")

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
        results = self.detector.detect_faces(rgb_img)
        if not results:
            return None, 0.0
        main = max(results, key=lambda x: x['confidence'])
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

# 全域 recognizer，只初始化一次
recognizer = FaceRecognizer(threshold=0.5)


# =========================== 3. WebSocket Handler ===========================
async def ws_handler(ws, path):
    """
    只處理 path 以 /ws 開頭的 WebSocket 握手 & 訊息收發
    """
    print(f"WebSocket 已連到：path={path}")
    try:
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
                    await ws.send(json.dumps({
                        "status": "fail",
                        "reason": "不支援的 type"
                    }))
                    continue

                if user:
                    resp = {"status": "ok", "user": user, "similarity": round(sim, 4)}
                else:
                    reason = "未檢測到人臉" if sim == 0.0 else "相似度不足"
                    resp = {"status": "fail", "reason": reason, "similarity": round(sim, 4)}
                await ws.send(json.dumps(resp))
                print("回傳結果 →", resp)

            except Exception as je:
                await ws.send(json.dumps({
                    "status": "error",
                    "message": str(je)
                }))
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket 正常關閉")
    except Exception as e:
        print("WebSocket 例外：", e)


# ====================== 4. process_request 用以攔截 HTTP 健康檢查 ======================
async def process_request(path, request_headers):
    """
    如果 path 是 "/" 或 "/healthz" 且 method 為 HEAD/GET，就回 200，
    讓 Render 的 HTTP 健康檢查通過。其餘情況下回 None，表示繼續做 WebSocket 握手檢查。
    """
    # 在 websockets.serve 這邊，process_request 只有 path 和 headers，沒有 method，
    # 但 HTTP HEAD/GET 在 WebSocket 握手時 path 一定不含 "Upgrade: websocket" 之類 header。
    # 我們只要看 path：
    if path == "/" or path == "/healthz":
        # 回傳一個 HTTP 200，body 空：
        return (
            200,                          # status
            [("Content-Type", "text/plain")],  # headers
            b""                          # 空 body
        )
    # 路徑不是 "/"、"/healthz"，就回 None，讓 websockets 判斷是否為升級 WebSocket
    return None


# ===================== 5. main: 只用 websockets.serve 監聽同一個 port =====================
async def main():
    port = int(os.getenv("PORT", "8765"))
    bind_addr = "0.0.0.0"

    # 只呼叫一個 websockets.serve()，把 process_request 參數帶進去：
    server = await websockets.serve(
        ws_handler,
        bind_addr,
        port,
        process_request=process_request
    )
    print(f"✅ Server 已啟動，監聽 {bind_addr}:{port} (同時接受 HTTP HEAD/GET /healthz & WebSocket /ws)")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
