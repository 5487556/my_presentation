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
from supabase import create_client, Client

# =============================================================================
# 1. 全域設定
# =============================================================================
# 你的 Supabase 專案 URL，格式類似：https://<project-ref>.supabase.co
SUPABASE_URL = "https://wiqldwmpszfinwbdegrs.supabase.co"
# 你的 Supabase service_role key（或 anon key，看存取權限）
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndpcWxkd21wc3pmaW53YmRlZ3JzIiwicm9sZSI6InNlcnZpY2NlX3JvbGUiLCJpYXQiOjE3NDg5NTUyNTQsImV4cCI6MjA2NDUzMTI1NH0.5jB5nf7D-OM704ZF29NVVNyEkHtxORx5PXyyLXIshbs"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_PATH = os.getenv('FACE_AUTH_PATH', r'D:\mydev\my_presentation\FaceAuthSystem')
for sub in ('database','models','test_images'):
    os.makedirs(os.path.join(BASE_PATH, sub), exist_ok=True)

ALLOWED_ID_PATTERN = r'^[a-zA-Z0-9_-]{4,20}$'
SAVE_IMAGE_FORMAT = 'png'
MIN_FACE_CONFIDENCE = 0.95
FACE_SIZE = (160,160)
INDEX_TREES = 15

def validate_user_id(uid):
    if not re.match(ALLOWED_ID_PATTERN, uid):
        raise ValueError(f"無效用戶ID: {uid}")
    return uid

def safe_filename(uid):
    return quote(uid, safe='')

# =============================================================================
# 2. FaceRegister（不必改動）
# =============================================================================
class FaceRegister:
    def __init__(self):
        self.detector = MTCNN()
        self.embedder = FaceNet()

    def _process_face(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片不存在：{img_path}")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        if not faces:
            raise ValueError("未檢測到人臉")
        main = max(faces, key=lambda x: x['confidence'])
        if main['confidence'] < MIN_FACE_CONFIDENCE:
            raise ValueError(f"人臉品質不足 ({main['confidence']:.2f})")
        x,y,w,h = (max(0,v) for v in main['box'])
        roi = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return roi, face_img

    def register(self, img_path, user_id, debug=False):
        try:
            uid = validate_user_id(user_id)
            safe_id = safe_filename(uid)
            _, face_img = self._process_face(img_path)
            emb = self.embedder.embeddings([face_img])[0]
            np.save(os.path.join(BASE_PATH,'database',f"{safe_id}.npy"), emb)
            cv2.imwrite(
                os.path.join(BASE_PATH,'database',f"{safe_id}.{SAVE_IMAGE_FORMAT}"),
                cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION,0]
            )
            print(f"✅ 註冊成功：{uid}")
            return True
        except Exception as e:
            print(f"❌ 註冊失敗：{e}")
            return False

# =============================================================================
# 3. FaceRecognizer（新增從 Supabase 下載的功能）
# =============================================================================
class FaceRecognizer:
    def __init__(self, threshold=0.5):
        # GPU 加速
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

        # 載入本地 database
        self.user_ids, self.embeddings = [], []
        db_dir = os.path.join(BASE_PATH,'database')
        for fname in os.listdir(db_dir):
            if fname.endswith('.npy'):
                uid = os.path.splitext(fname)[0]
                emb = np.load(os.path.join(db_dir, fname))
                if emb.shape[0]==512:
                    self.user_ids.append(uid)
                    self.embeddings.append(emb)
        if not self.embeddings:
            print("⚠️ 本地資料庫空，請先註冊用戶")

        self.index = AnnoyIndex(512,'angular')
        for i,e in enumerate(self.embeddings):
            self.index.add_item(i,e)
        self.index.build(INDEX_TREES)
        print(f"✅ Index 建立完成，共 {self.index.get_n_items()} 筆")

    def _recognize_embedding(self, emb):
        idxs, dists = self.index.get_nns_by_vector(emb,3,include_distances=True)
        if not idxs:
            return None,0.0
        sim = 1.0 - (dists[0]**2)/2
        return (self.user_ids[idxs[0]], sim) if sim>=self.threshold else (None,sim)

    def recognize_from_image(self, rgb_img, debug=False):
        faces = self.detector.detect_faces(rgb_img)
        if not faces:
            return None,0.0
        main=max(faces,key=lambda x:x['confidence'])
        if main['confidence']<MIN_FACE_CONFIDENCE:
            return None,main['confidence']
        x,y,w,h=(max(0,v) for v in main['box'])
        roi=rgb_img[y:y+h, x:x+w]
        img=cv2.resize(roi,FACE_SIZE,interpolation=cv2.INTER_LANCZOS4)
        emb=self.embedder.embeddings([img])[0]
        return self._recognize_embedding(emb)

    def recognize_from_file(self, img_path, debug=False):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到圖片：{img_path}")
        bgr=cv2.imread(img_path)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        return self.recognize_from_image(rgb, debug)

    def _download_image_from_supabase(self, bucket, object_name):
        """從 Supabase Storage 下載圖片，回傳 RGB numpy.ndarray"""
        data = supabase.storage.from_(bucket).download(object_name)
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"解碼 Supabase 圖片 {object_name} 失敗")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def recognize_from_supabase(self, bucket, object_name):
        """支援直接傳 bucket, object_name 辨識"""
        rgb = self._download_image_from_supabase(bucket, object_name)
        return self.recognize_from_image(rgb)

# 全域 recognizer
recognizer = FaceRecognizer(threshold=0.5)

# =============================================================================
# 4. 攔截健康檢查
# =============================================================================
async def process_request(path, request_headers):
    if path in ('/','/healthz'):
        return (200,[('Content-Type','text/plain')],b'')
    return None

# =============================================================================
# 5. WebSocket Handler
# =============================================================================
async def handler(websocket):
    await websocket.send(json.dumps({'status':'ready','message':'Server Ready'}))
    async for raw in websocket:
        try:
            msg=json.loads(raw)
            t=msg.get('type','')
            if t=='file':
                user,sim = recognizer.recognize_from_file(msg.get('path',''))
            elif t=='base64':
                b64=msg.get('data','')
                if ',' in b64:b64=b64.split(',',1)[1]
                data=base64.b64decode(b64)
                arr=np.frombuffer(data,np.uint8)
                bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                user,sim=recognizer.recognize_from_image(rgb)
            elif t=='supabase':
                bucket=msg.get('bucket'); obj=msg.get('object')
                user,sim=recognizer.recognize_from_supabase(bucket,obj)
            else:
                await websocket.send(json.dumps({'status':'fail','reason':'不支援的 type'}))
                continue

            if user:
                resp={'status':'ok','user':user,'similarity':round(sim,4)}
            else:
                reason='未檢測到人臉' if sim==0.0 else '相似度不足'
                resp={'status':'fail','reason':reason,'similarity':round(sim,4)}
            await websocket.send(json.dumps(resp))

        except Exception as e:
            await websocket.send(json.dumps({'status':'error','message':str(e)}))

# =============================================================================
# 6. 啟動 Server
# =============================================================================
async def main():
    port=int(os.getenv('PORT','8080'))
    await websockets.serve(handler,'0.0.0.0',port,process_request=process_request)
    await asyncio.Future()

if __name__=='__main__':
    asyncio.run(main())
