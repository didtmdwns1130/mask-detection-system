import cv2
import numpy as np
from pathlib import Path
import shutil
import tempfile

def _clip(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def _nms_xywh(boxes, iou_thr=0.35):
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32)  # [x,y,w,h,score]
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,0]+b[:,2], b[:,1]+b[:,3]
    s = b[:,4]; a = (x2-x1)*(y2-y1); order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h; iou = inter / (a[i] + a[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thr]
    return b[keep, :4].astype(int).tolist()

class FaceDetector:
    def __init__(self, min_size=(80, 80)):
        model_dir = Path(__file__).resolve().parent / "models"
        prototxt_path = model_dir / "deploy.prototxt"

        # fp16 혼용으로 미스매치 나는 경우가 있어 일반본만 강제
        weight_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if not prototxt_path.exists():
            raise FileNotFoundError(f"[FaceDetector] missing prototxt: {prototxt_path}")
        if prototxt_path.stat().st_size == 0:
            raise RuntimeError(f"[FaceDetector] empty prototxt: {prototxt_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"[FaceDetector] missing caffemodel: {weight_path}")

        # 1) 임시폴더 경로 먼저 만든다
        tmp_root = Path(tempfile.gettempdir()) / "fmd_models"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_prototxt = tmp_root / "deploy.prototxt"
        tmp_weights  = tmp_root / weight_path.name

        # 2) 복사한 다음
        if (not tmp_prototxt.exists()) or (tmp_prototxt.stat().st_mtime < prototxt_path.stat().st_mtime):
            shutil.copy2(prototxt_path, tmp_prototxt)
        if (not tmp_weights.exists()) or (tmp_weights.stat().st_mtime < weight_path.stat().st_mtime):
            shutil.copy2(weight_path, tmp_weights)

        # 3) 그 다음에 print (여기서만!)
        print(f"[FMD] prototxt -> {tmp_prototxt}")
        print(f"[FMD] weights  -> {tmp_weights}")

        # 4) 마지막에 로드
        self.net = cv2.dnn.readNetFromCaffe(str(tmp_prototxt), str(tmp_weights))

        self.conf_thr = 0.25
        self.min_size = tuple(min_size)

        cdir = cv2.data.haarcascades
        self.haar = cv2.CascadeClassifier(cdir + "haarcascade_frontalface_alt2.xml")

    def _dnn_detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.net.setInput(blob)
        dets = self.net.forward()  # shape: [1,1,N,7]
        boxes = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.conf_thr:
                continue
            x1 = int(dets[0, 0, i, 3] * W)
            y1 = int(dets[0, 0, i, 4] * H)
            x2 = int(dets[0, 0, i, 5] * W)
            y2 = int(dets[0, 0, i, 6] * H)
            w = x2 - x1; h = y2 - y1
            x1, y1, w, h = _clip(x1, y1, w, h, W, H)
            if w >= self.min_size[0] and h >= self.min_size[1]:
                boxes.append([x1, y1, w, h, conf])
        return _nms_xywh(boxes, 0.35)

    def detect(self, frame_bgr):
        # 1) DNN 시도
        boxes = self._dnn_detect(frame_bgr)
        if boxes:
            return boxes
        # 2) 백업: Haar(가볍게)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.haar.detectMultiScale(gray, 1.05, 4, minSize=self.min_size)
        out = []
        H, W = frame_bgr.shape[:2]
        for (x, y, w, h) in faces:
            x, y, w, h = _clip(x, y, w, h, W, H)
            out.append((x, y, w, h))
        return out
