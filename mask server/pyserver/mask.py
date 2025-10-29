import threading
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
import uvicorn
from pypylon import pylon


app = FastAPI()

_camera_lock = threading.Lock()
_camera: Optional[pylon.InstantCamera] = None
_converter: Optional[pylon.ImageFormatConverter] = None


def _open_camera() -> tuple[Optional[pylon.InstantCamera], Optional[pylon.ImageFormatConverter]]:
    camera: Optional[pylon.InstantCamera] = None
    try:
        device = pylon.TlFactory.GetInstance().CreateFirstDevice()
        camera = pylon.InstantCamera(device)
        camera.Open()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        return camera, converter
    except Exception:
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()
            finally:
                camera = None
        return None, None


def _ensure_camera() -> bool:
    global _camera, _converter
    if _camera is not None and _camera.IsOpen() and _camera.IsGrabbing():
        return True

    if _camera is not None:
        try:
            if _camera.IsGrabbing():
                _camera.StopGrabbing()
            if _camera.IsOpen():
                _camera.Close()
        finally:
            _camera = None
            _converter = None

    camera, converter = _open_camera()
    _camera = camera
    _converter = converter
    return _camera is not None


def _read_frame() -> Optional[np.ndarray]:
    global _camera, _converter
    with _camera_lock:
        if not _ensure_camera():
            return None
        if _camera is None or _converter is None:
            return None

        try:
            res = _camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        except (pylon.TimeoutException, pylon.RuntimeException):
            return None

        if res is None or not res.GrabSucceeded():
            if res is not None:
                res.Release()
            return None

        try:
            image = _converter.Convert(res)
            frame = image.GetArray()
        finally:
            res.Release()

    return frame


@app.on_event("startup")
def startup_event():
    with _camera_lock:
        _ensure_camera()


@app.on_event("shutdown")
def shutdown_event():
    global _camera, _converter
    with _camera_lock:
        if _camera is not None:
            if _camera.IsGrabbing():
                _camera.StopGrabbing()
            if _camera.IsOpen():
                _camera.Close()
            _camera = None
        _converter = None


@app.get("/health")
def health():
    with _camera_lock:
        ok = _camera is not None and _camera.IsOpen() and _camera.IsGrabbing()
    return {"ok": ok}


@app.get("/status")
def status():
    return {"mask": True, "prob": 0.99}


@app.get("/frame.jpg")
def frame():
    frame = _read_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    height, width = frame.shape[:2]
    target_width = 640
    if width != target_width and width != 0:
        target_height = int(height * (target_width / width))
        frame = cv2.resize(frame, (target_width, max(target_height, 1)))

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("mask:app", host="0.0.0.0", port=8000)
