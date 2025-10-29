from db import save_event

save_event(
    status="NO_MASK",
    prob=0.87,
    x=100, y=120, w=240, h=240,
    image_path="captures/no_mask_01.jpg",
    camera_id="CAM01",
    extra={"note": "manual test"}
)
