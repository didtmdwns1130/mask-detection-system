# app.py - 마스크 탐지 서버 파일. 먼저 서버 실행 후 클라 실행을 해야함

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import threading
import time

import inspect

import cv2
import numpy as np
from fastapi import FastAPI, Response

# ---- Basler (있으면 사용, 없으면 웹캠 폴백) ----
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except Exception:
    pylon = None
    PYLON_AVAILABLE = False

# ---- DB 저장 함수 (사용자 제공 모듈) ----
try:
    from db import save_event
except Exception:
    def save_event(*args, **kwargs):
        pass  # DB 모듈 없으면 무시

# =========================
# (옵션) face_mask_detection 모듈
# =========================
try:
    from face_mask_detection import FaceDetector as _FD
except Exception:
    _FD = None
try:
    from face_mask_detection import MaskDetector as _MD
except Exception:
    _MD = None

# 여기 한 줄 추가
print(f"[FMD import] FaceDetector={'OK' if _FD else 'None'}  MaskDetector={'OK' if _MD else 'None'}")


# =========================
# 파라미터
# =========================
WINDOW_NAME = "Mask Check (Basler + face_mask_detection)"
DISPLAY_WIDTH = 960
MIN_FACE_SIZE = (120, 120)

# 오탐 억제 강화
PROB_ON  = 0.85     # 착용으로 굳히는 임계 ↑
PROB_OFF = 0.15     # 미착용으로 굳히는 임계 ↓
N_ON_FRAMES  = 4    # 착용 확정에 연속 프레임 요구
N_OFF_FRAMES = 2    # 미착용 확정도 약간의 연속성 요구
EMA_ALPHA = 0.35    # 확률 반응 속도 ↑ (너무 흔들리지 않게)
SHRINK_FACE_ROI = 0.10  # 하관 중심으로 ROI 좁힘

SHOW_LABEL_TEXT = True
SHOW_DEBUG_OVERLAY = False
PRINT_DEBUG = False

MODEL_PROB_IS_WORN = False  # True: 값=착용확률, False: 값=미착용확률

def dbg(*args):
    return  # 완전 비활성

# =========================
# 폰트(라벨은 서버 내 렌더 불필요, 유지만)
# =========================
from PIL import Image, ImageDraw, ImageFont
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}
def _load_korean_font(font_size=28):
    cached = _FONT_CACHE.get(font_size)
    if cached is not None:
        return cached
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
        Path("C:/Windows/Fonts/batang.ttc"),
    ]
    for fp in candidates:
        if fp.exists():
            try:
                f = ImageFont.truetype(str(fp), font_size)
                _FONT_CACHE[font_size] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _FONT_CACHE[font_size] = f
    return f

def _draw_labels(frame: np.ndarray, annotations: List[Tuple[str, Tuple[int,int,int,int], Tuple[int,int,int]]]) -> np.ndarray:
    if not annotations or not SHOW_LABEL_TEXT:
        return frame
    font = _load_korean_font()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    H, W = frame.shape[:2]
    for text, (x,y,w,h), bgr in annotations:
        try:
            bbox = dr.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = dr.textsize(text, font=font)
        tx = max(8, min(x, W-8-tw))
        ty = y - th - 10
        if ty < 8: ty = min(y + h + 8, H - th - 8)
        pad = 6
        bg = (max(tx-pad,0), max(ty-pad,0), min(tx+tw+pad, W), min(ty+th+pad, H))
        dr.rectangle(bg, fill=(0,0,0))
        dr.text((tx,ty), text, font=font, fill=(int(bgr[2]), int(bgr[1]), int(bgr[0])))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# =========================
# 얼굴 검출
# =========================
class FMDFaceDetector:
    def __init__(self):
        # face_mask_detection.FaceDetector가 있으면 그것만 사용
        if _FD is not None:
            self.det = _FD()
            self.mode = "FaceDetector"
        else:
            # 없으면 OpenCV Haar 폴백
            self.det = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.det.empty():
                raise RuntimeError("No face detector available.")
            self.mode = "OpenCV-Haar"

        dbg(f"[FaceDetector] mode={self.mode}")

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        if self.mode == "FaceDetector":
            # FMD FaceDetector의 출력 형태를 박스 리스트로 표준화
            boxes = self.det.detect(frame_bgr)
            return self._normalize_boxes(boxes, frame_bgr)

        # OpenCV-Haar 폴백
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=6, minSize=MIN_FACE_SIZE
        )
        return self._post_filter_boxes(
            [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces],
            frame_bgr
        )

    @staticmethod
    def _normalize_boxes(boxes, frame):
        out = []
        H, W = frame.shape[:2]
        if boxes is None:
            return out
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x, y, w, h = map(int, b[:4])
            elif isinstance(b, dict) and all(k in b for k in ("x", "y", "w", "h")):
                x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            else:
                continue
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]:
                out.append((x, y, w, h))
        return FMDFaceDetector._post_filter_boxes(out, frame)

    @staticmethod
    def _post_filter_boxes(boxes: List[Tuple[int,int,int,int]], frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        H, W = frame.shape[:2]
        flt = []
        min_area = int(0.015 * W * H)
        for (x, y, w, h) in boxes:
            area = w * h
            ar = w / float(h) if h > 0 else 0
            if area < min_area:
                continue
            if not (0.75 <= ar <= 1.33):
                continue
            flt.append((x, y, w, h))
        return flt

# =========================
# 강건한 하관 가림 휴리스틱
# =========================
# ---- [PATCH 1] _LowerFaceRobust 수정 ----
class _LowerFaceRobust:
    def predict(self, face_bgr: np.ndarray):
        if face_bgr is None or face_bgr.size == 0:
            return {"prob_worn": 0.05}

        h, w = face_bgr.shape[:2]

        # 상/하 분할: 0.55 -> 0.60 (하관을 더 넓게 봄)
        y_mid = int(h * 0.60)
        top = face_bgr[0:y_mid, :]
        bot = face_bgr[y_mid:h, :]
        if top.size == 0 or bot.size == 0:
            return {"prob_worn": 0.05}

        def skin_frac(img):
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
            return float(np.count_nonzero(mask)) / float(img.shape[0]*img.shape[1])

        sf_top = skin_frac(top)
        sf_bot = skin_frac(bot)
        delta  = max(0.0, sf_top - sf_bot)

        # 무채색·비피부 비율: 채도 임계 35 -> 60 (흰/검/회색 마스크 더 잘 잡도록)
        hsv_bot = cv2.cvtColor(bot, cv2.COLOR_BGR2HSV)
        S_bot = hsv_bot[...,1]
        lowS = cv2.inRange(S_bot, 0, 60)
        ycbcr_bot = cv2.cvtColor(bot, cv2.COLOR_BGR2YCrCb)
        skin_mask_bot = cv2.inRange(ycbcr_bot, (0,133,77), (255,173,127))
        nonSkin_lowS = cv2.bitwise_and(lowS, cv2.bitwise_not(skin_mask_bot))
        frac_nonskin_lowS = float(np.count_nonzero(nonSkin_lowS)) / float(bot.shape[0]*bot.shape[1])

        # 엣지(주름) 밀도 완화
        gray_bot = cv2.cvtColor(bot, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_bot, 35, 110)
        edge_density = float(np.count_nonzero(edges)) / float(edges.shape[0]*edges.shape[1])

        hsv_full = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        mean_S = float(np.mean(hsv_full[...,1])) / 255.0
        mean_Y = float(np.mean(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)[...,0]))

        # 규칙 완화
        t1 = (sf_top >= 0.14 and sf_bot <= 0.32 and delta >= 0.07)
        t2 = (delta >= 0.20 and sf_top >= 0.16)
        t3 = (frac_nonskin_lowS >= 0.18)
        t4 = (edge_density >= 0.10 and sf_bot <= 0.32)

        # 극저조도 가드 완화: 과거엔 "둘 다 매우 낮음"이면 강제로 미착용 -> 제거
        grayscale_env  = (mean_S < 0.25 or mean_Y < 60.0)

        if grayscale_env:
            covered = (t1 or t2 or t3 or t4)
        else:
            covered = (t1 or t2 or t3 or t4)

        # 연속 확률 산출 (기본치 상향 + 무채색/하관피부 가중치 ↑)
        score = 0.55
        score += 0.50 * np.clip(delta, 0.0, 0.6) / 0.6
        score += 0.45 * np.clip(frac_nonskin_lowS - 0.10, 0.0, 0.6) / 0.6
        score += 0.25 * np.clip(0.32 - sf_bot, 0.0, 0.32) / 0.32

        if covered:
            score = max(score, 0.72)
        else:
            score = min(score, 0.28)

        return {"prob_worn": float(np.clip(score, 0.0, 1.0))}



# =========================
# FMDMaskInspector 클래스
# =========================
# ==== REPLACE: FMDMaskInspector (모델 전용) ====
class FMDMaskInspector:
    def __init__(self):
        if _MD is None:
            raise RuntimeError("face_mask_detection.MaskDetector 모듈이 필요합니다.")

        # __init__ 시그니처 점검해서 detector_backend 지원 여부만 선택적으로 전달
        sig = inspect.signature(_MD.__init__)
        kwargs = {}
        try:
            sig = inspect.signature(_MD.__init__)
            if "detector_backend" in sig.parameters:
                kwargs["detector_backend"] = "retinaface"
        except Exception:
            # 시그니처 조회 불가한 구현(C 확장 등)에도 안전
            pass

        self.model = _MD(**kwargs)
        self.mode = "MaskDetector"

    def predict_prob(self, face_bgr: np.ndarray) -> float:
        if face_bgr is None or face_bgr.size == 0:
            return 0.0

        # 다양한 구현 포맷 수용
        if hasattr(self.model, "predict"):
            out = self.model.predict(face_bgr)
        elif hasattr(self.model, "infer"):
            out = self.model.infer(face_bgr)
        elif callable(self.model):
            out = self.model(face_bgr)
        else:
            raise RuntimeError("MaskDetector에 호출 가능한 API가 없습니다.")

        # 반환값 파싱
        if isinstance(out, dict):
            if "prob_worn" in out:
                return float(np.clip(out["prob_worn"], 0.0, 1.0))
            if "mask" in out and "prob" in out:
                m, p = bool(out["mask"]), float(out["prob"])
                return float(np.clip(p if m else 1.0 - p, 0.0, 1.0))
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            m, p = bool(out[0]), float(out[1])
            return float(np.clip(p if m else 1.0 - p, 0.0, 1.0))
        
        if isinstance(out, (int, float)):
            p = float(out)
            # 모델이 내놓는 값의 의미를 스위치로 선택
            # MODEL_PROB_IS_WORN = True  -> out == 착용확률
            # MODEL_PROB_IS_WORN = False -> out == 미착용확률
            return float(np.clip(p if MODEL_PROB_IS_WORN else (1.0 - p), 0.0, 1.0))

        raise RuntimeError(f"MaskDetector 결과 형식 인식 실패: {type(out)} -> {out}")



# =========================
# 스무딩 트랙/IOU
# =========================
@dataclass
class Track:
    box: Tuple[int,int,int,int]
    prob_ema: float
    stable_state: Optional[bool] = None
    miss: int = 0
    on_cnt: int = 0
    off_cnt: int = 0

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = aw*ah + bw*bh - inter
    if union <= 0: return 0.0
    return inter / union

# =========================
# DB 로깅
# =========================
_last_save_time = 0.0
SAVE_COOLDOWN_SEC = 3.0

def log_event(mask_on: bool, prob: float, box, frame_bgr=None):
    global _last_save_time
    now = time.time()
    if now - _last_save_time < SAVE_COOLDOWN_SEC:
        return
    _last_save_time = now
    status = "MASK_ON" if mask_on else "NO_MASK"
    try:
        save_event(status=status, prob=float(prob), camera_id="CAM01")
    except Exception as e:
        print(f"[DB][ERR] {e}")

# =========================
# FastAPI 서버 상태 공유
# =========================
app = FastAPI(title="Mask Check Server")

_last_jpeg: bytes = b""
_last_status: Dict[str, object] = {"mask": None, "prob": 0.0, "faces": 0, "ts": 0.0}
_state_lock = threading.Lock()
_running = False
_worker: Optional[threading.Thread] = None

def _update_shared(jpeg: Optional[bytes], mask: Optional[bool], prob: float, faces: int):
    global _last_jpeg, _last_status
    with _state_lock:
        if jpeg is not None:
            _last_jpeg = jpeg
        _last_status = {"mask": mask, "prob": float(prob), "faces": int(faces), "ts": time.time()}

def camera_loop():
    global _running
    _running = True

    # 입력 소스 선택: Basler 우선, 실패 시 웹캠
    cap = None
    cam = None
    converter = None
    use_basler = False

    if PYLON_AVAILABLE:
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            cam.Open()
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            use_basler = True
            print(">> Basler camera opened")
        except Exception as e:
            print(f"! Basler open failed: {e}")

    if not use_basler:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("! camera open failed"); _running = False; return
        print(">> Using default webcam")

    face_det = FMDFaceDetector()
    mask_ins = FMDMaskInspector()

    tracks: List[Track] = []
    MAX_MISS = 5
    IOU_THR = 0.3

    try:
        while _running:
            # ---- 프레임 획득 ----
            if use_basler:
                if not cam.IsGrabbing():
                    time.sleep(0.01); continue
                res = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                if not res.GrabSucceeded():
                    res.Release(); continue
                try:
                    img = converter.Convert(res)
                    frame = img.GetArray()
                finally:
                    res.Release()
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01); continue

            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                continue

            disp_h = int(h * (DISPLAY_WIDTH / w))
            frame = cv2.resize(frame, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_AREA)

            # ---- 얼굴 검출/판정 ----
            boxes = face_det.detect(frame)
            for t in tracks: t.miss += 1
            annotations: List[Tuple[str, Tuple[int,int,int,int], Tuple[int,int,int]]] = []
            new_tracks: List[Track] = []
            any_mask: Optional[bool] = None
            any_prob = 0.0

            for (x,y,wf,hf) in boxes:
                sx, sy = int(wf * SHRINK_FACE_ROI), int(hf * SHRINK_FACE_ROI)
                x1 = max(0, x + sx); y1 = max(0, y + sy)
                x2 = min(frame.shape[1], x + wf - sx); y2 = min(frame.shape[0], y + hf - sy)
                if x2 <= x1 or y2 <= y1:
                    continue

                face_roi = frame[y1:y2, x1:x2]
                p_mask = mask_ins.predict_prob(face_roi)

                # 트랙 매칭
                best_idx, best_iou = -1, 0.0
                for i, t in enumerate(tracks):
                    iouv = iou((x,y,wf,hf), t.box)
                    if iouv > best_iou: best_iou, best_idx = iouv, i

                if best_iou >= IOU_THR and best_idx >= 0:
                    prev = tracks[best_idx].prob_ema
                    prob_ema = EMA_ALPHA * p_mask + (1.0 - EMA_ALPHA) * prev
                    st = tracks[best_idx].stable_state
                    prev_on = tracks[best_idx].on_cnt
                    prev_off = tracks[best_idx].off_cnt
                else:
                    prob_ema = p_mask
                    st = None
                    prev_on = 0
                    prev_off = 0

                score = prob_ema
                proposed_on  = (score >= PROB_ON)
                proposed_off = (score <= PROB_OFF)

                if proposed_on:
                    on_cnt = prev_on + 1; off_cnt = 0
                elif proposed_off:
                    off_cnt = prev_off + 1; on_cnt = 0
                else:
                    on_cnt = 0; off_cnt = 0

                if st is None:
                    st = (on_cnt >= N_ON_FRAMES)
                else:
                    if st and off_cnt >= N_OFF_FRAMES: st = False
                    elif (not st) and on_cnt >= N_ON_FRAMES: st = True

                nt = Track(box=(x,y,wf,hf), prob_ema=float(prob_ema), stable_state=st,
                           miss=0, on_cnt=on_cnt, off_cnt=off_cnt)
                new_tracks.append(nt)

                if (not st and off_cnt == 1) or (st and on_cnt == 1):
                    log_event(st, score, (x, y, wf, hf), frame_bgr=frame)

                color = (0,255,0) if st else (0,0,255)
                if SHOW_LABEL_TEXT:
                    label = "마스크 착용" if st else "마스크 미착용"
                    annotations.append((label, (x,y,wf,hf), color))

                any_mask = st
                any_prob = float(score)

            tracks = [t for t in new_tracks if t.miss <= MAX_MISS]
            if annotations:
                frame = _draw_labels(frame, annotations)

            # ---- JPEG 버퍼/상태 공유 업데이트 ----
            if any_mask is None:
                any_mask = False
                any_prob = 0.0
            ok2, buf = cv2.imencode(".jpg", frame)
            jpeg = buf.tobytes() if ok2 else None
            _update_shared(jpeg, any_mask, any_prob, faces=len(boxes))

            # 속도 조절
            time.sleep(0.03)

    finally:
        if use_basler and cam is not None:
            cam.StopGrabbing(); cam.Close()
        if not use_basler and cap is not None:
            cap.release()

# =========================
# FastAPI 엔드포인트
# =========================
@app.get("/status")
def status():
    with _state_lock:
        return dict(_last_status)

@app.get("/frame.jpg")
def frame():
    with _state_lock:
        data = _last_jpeg
    if not data:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg")

# =========================
# 수명주기 훅: 서버 시작/종료 시 워커 관리
# =========================
@app.on_event("startup")
def _on_start():
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=camera_loop, daemon=True)
    _worker.start()
    print(">> camera worker started")

@app.on_event("shutdown")
def _on_stop():
    global _running
    _running = False
    print(">> shutting down")

# 개발 중 직접 실행:
if __name__ == "__main__":
     import uvicorn
     uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
