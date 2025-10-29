Mask Detection System 🩺😷

마스크 착용 여부를 실시간으로 감지하는 시스템입니다.  
Python FastAPI 서버와 MFC C++ 클라이언트로 구성되어 있으며,  
Basler 산업용 카메라 또는 웹캠을 통해 얼굴을 인식하고  
마스크 착용 상태를 판단합니다.

---

📁 프로젝트 구조

mask-detection-system/
│
├─ mask client/ # MFC 클라이언트 (C++)
│ └─ MfcTestProject/ # UI, 영상 표시, 서버 통신
│
├─ mask server/pyserver/ # FastAPI 기반 파이썬 서버
│ ├─ app.py # 메인 서버 (실시간 영상 스트리밍 + 감지)
│ ├─ db.py # MySQL DB 로그 관리
│ ├─ face_mask_detection/ # 마스크 감지 모델
│ │ ├─ detector.py
│ │ ├─ inspector.py
│ │ └─ models/
│ │ ├─ deploy.prototxt
│ │ └─ res10_300x300_ssd_iter_140000.caffemodel
│ └─ requirements.txt # Python 패키지 의존성 목록
│
└─ .gitignore # IDE 및 빌드 캐시 무시 규칙
