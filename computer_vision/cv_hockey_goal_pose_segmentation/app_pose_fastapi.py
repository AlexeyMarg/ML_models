from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import io
from contextlib import asynccontextmanager
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "pose" / "train" / "weights" / "best.pt"

POINT_NAMES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully")
    yield
    del yolo_model
    
    
def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def detect_corners(image: np.ndarray):
    results = yolo_model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False,
    )
    result = results[0]

    goals = []
    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or keypoints is None or len(boxes) == 0:
        return goals 

    for obj_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())

        kpts_xy = keypoints.xy[obj_idx].cpu().numpy()
        kpts_conf = None
        if keypoints.conf is not None:
            kpts_conf = keypoints.conf[obj_idx].cpu().numpy()

        corners = []
        for point_idx, (x, y) in enumerate(kpts_xy):
            x, y = int(x), int(y)
            kp_conf = float(kpts_conf[point_idx]) if kpts_conf is not None else 1.0
            corners.append({
                "name": POINT_NAMES[point_idx],
                "x": x,
                "y": y,
                "confidence": kp_conf if kpts_conf is not None else None,
            })

        goals.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": conf,
            "corners": corners,
        })

    return goals


def draw_annotations(image: np.ndarray, goals: list) -> np.ndarray:
    annotated = image.copy()
    for goal in goals:
        x1, y1, x2, y2 = goal["bbox"]
        conf = goal["confidence"]
        corners = goal["corners"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"goal {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for corner in corners:
            if corner.get("confidence") is not None and corner["confidence"] < 0.25:
                continue
            cv2.circle(annotated, (corner["x"], corner["y"]), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                corner["name"],
                (corner["x"] + 6, corner["y"] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
    return annotated


app = FastAPI(title="Hockey Goal Corner Detector", lifespan=lifespan)


@app.post("/coordinates")
async def get_coordinates(file: UploadFile = File(...)):
    image = read_image(file)
    goals = detect_corners(image)
    return {"goals": goals}

@app.post("/annotate")
async def get_annotated_image(file: UploadFile = File(...)):
    image = read_image(file)
    goals = detect_corners(image)
    annotated = draw_annotations(image, goals)

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)