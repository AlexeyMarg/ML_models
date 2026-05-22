from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_seg_dataset" / "data.yaml"

model = YOLO("yolo11n-seg.pt")

model.train(
    data=str(DATA_YAML),
    epochs=300,
    imgsz=960,
    batch=8,
    device="cpu",

    degrees=3,
    translate=0.08,
    scale=0.35,
    fliplr=0.5,
    mosaic=0.3,
    close_mosaic=20,

    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,

    mask_ratio=2,
    overlap_mask=True,
)

best_model = YOLO("runs/segment/train/weights/best.pt")

best_model.predict(
    source="path/to/test_image.jpg",
    imgsz=640,
    conf=0.25,
    save=True,
)