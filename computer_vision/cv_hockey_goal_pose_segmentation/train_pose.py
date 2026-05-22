from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

model.train(
    data="yolo_pose_dataset/data.yaml",
    epochs=200,
    imgsz=960,
    batch=8,
    device="cpu",
    degrees=3,
    translate=0.08,
    scale=0.4,
    fliplr=0.5,
    mosaic=0.3,
    close_mosaic=20,
)

best_model = YOLO("runs/pose/train/weights/best.pt")

best_model.predict(
    source="path/to/test_image.jpg",
    imgsz=640,
    conf=0.25,
    save=True,
)