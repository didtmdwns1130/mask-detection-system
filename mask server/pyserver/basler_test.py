from pypylon import pylon
import cv2

# Basler 카메라 객체 생성 및 연결
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# 카메라에서 받아온 Bayer 데이터를 BGR(OpenCV 호환)로 변환하기 위한 설정
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# 최신 프레임만 유지하는 전략으로 연속 캡처 시작
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

window_name = "Basler Camera"
display_width = 960  # 다른 PC에서도 동일한 표시 너비 사용
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    display_height = None  # 첫 프레임으로 비율을 계산해 창 크기 고정

    while camera.IsGrabbing():
        # 지정된 시간 안에 새 프레임을 가져오며 실패 시 예외를 발생시킴
        grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if not grab_result.GrabSucceeded():
            grab_result.Release()
            continue

        try:
            image = converter.Convert(grab_result)
            frame = image.GetArray()
        finally:
            grab_result.Release()

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            continue

        # 최초 한 번 창 크기를 프레임 비율에 맞춰 설정
        if display_height is None:
            display_height = int(height * (display_width / width))
            cv2.resizeWindow(window_name, display_width, display_height)

        # 고정된 창 크기에 맞춰 리사이즈하여 출력
        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break
finally:
    # 예외 발생 여부와 관계없이 자원 정리
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
