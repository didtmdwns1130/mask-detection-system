import cv2
import numpy as np

class MaskInspector:
    def __init__(self):
        pass
    def predict(self, face_bgr):
        return FakeMaskInspector().predict(face_bgr)

class FakeMaskInspector:
    """
    흑백/저채도 카메라 대응.
    - prob_present : 하관에 '마스크 같은' 매끈한 천이 일정 비율 있으면 높게.
    - prob_worn    : 코/입 노출이 적고(central dark 낮음) 가로주름이 있으면 높게.
    """
    def predict(self, face_bgr):
        h, w = face_bgr.shape[:2]
        if h < 60 or w < 60:
            return {"prob_present": 0.5, "prob_worn": 0.5}

        # 하관 중심 띠 (마스크가 있을 영역)
        y1, y2 = int(h*0.45), int(h*0.85)
        x1, x2 = int(w*0.20), int(w*0.80)
        band = face_bgr[y1:y2, x1:x2]
        if band.size == 0:
            return {"prob_present": 0.5, "prob_worn": 0.5}

        blur = cv2.GaussianBlur(band, (5,5), 0)
        g    = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        total = max(1, H*W)

        # ----- (A) '마스크 같은 천' 찾기: 밝기 중간대 + 매끈(저텍스처) -----
        # 매끈함: 라플라시안 분산이 낮음
        lap  = cv2.Laplacian(g, cv2.CV_32F)
        var  = cv2.GaussianBlur(lap*lap, (7,7), 0)
        smooth = (var < 85).astype(np.uint8)  # 매끈한 천
        # 너무 어둡거나 너무 밝은 건 제외
        bright_ok = cv2.inRange(g, 50, 220)   # 중간대
        mask_like = cv2.bitwise_and(smooth*255, bright_ok)
        # 작은 구멍 메우기
        mask_like = cv2.morphologyEx(mask_like, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        coverage = float(np.count_nonzero(mask_like)) / total  # 하관 내 '천' 비율

        # ----- (B) '제대로 착용' 판단용 보조 특징 -----
        # 중앙 어두운 영역(콧구멍/입) 비율: 작을수록 착용
        cc1, cc2 = int(W*0.35), int(W*0.65)
        center   = g[:, cc1:cc2]
        dark_center = float(np.count_nonzero(cv2.inRange(center, 0, 45))) / max(1, center.size)

        # 가로 주름(gy) vs 세로 윤곽(gx)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mean_gx = float(np.mean(np.abs(gx)))
        mean_gy = float(np.mean(np.abs(gy)))
        pleat_score = np.tanh((mean_gy - mean_gx) / 20.0)  # +면 가로주름 우세(마스크)

        # 균일도
        uniform = 1.0 - min(float(np.std(g))/128.0, 1.0)

        # ----- 확률 산출 -----
        # 존재(prob_present): '천' 커버리지 비중을 크게, 보조로 균일도 가산
        p_present = 0.75*coverage + 0.25*uniform

        # 제대로 착용(prob_worn): 존재에 (입/코 노출 없음) + (가로주름) 보정
        p_worn = (
            0.60*p_present +
            0.25*(1.0 - dark_center) +
            0.15*(0.5 + 0.5*pleat_score)
        )

        # 클램프
        p_present = float(np.clip(p_present, 0.0, 1.0))
        p_worn    = float(np.clip(p_worn,    0.0, 1.0))
        return {"prob_present": p_present, "prob_worn": p_worn}

MaskDetector = MaskInspector
