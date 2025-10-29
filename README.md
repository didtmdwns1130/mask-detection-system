Mask Detection System 🩺😷

마스크 착용 여부를 실시간으로 감지하는 시스템입니다.  
Python FastAPI 서버와 MFC C++ 클라이언트로 구성되어 있으며,  
Basler 산업용 카메라 또는 웹캠을 통해 얼굴을 인식하고  
마스크 착용 상태를 판단합니다.

---

📁 프로젝트 구조
mask-detection-system/
├─ mask client/
│  └─ MfcTestProject/
├─ mask server/
│  └─ pyserver/
│     ├─ app.py
│     ├─ db.py
│     └─ face_mask_detection/
│        ├─ detector.py
│        ├─ inspector.py
│        └─ models/
│           ├─ deploy.prototxt
│           └─ res10_300x300_ssd_iter_140000.caffemodel
├─ requirements.txt
└─ .gitignore


