from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import io
from contextlib import asynccontextmanager


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "segment" / "train" / "weights" / "best.pt"

POINT_NAMES = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]

yolo_model = None


def contour_to_quad(polygon: np.ndarray) -> np.ndarray:
    polygon = polygon.astype(np.int32)
    peri = cv2.arcLength(polygon, True)

    for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15]:
        approx = cv2.approxPolyDP(polygon, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.int32,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Segmentation model loaded successfully")
    yield
    del yolo_model


app = FastAPI(title="Hockey Goal Corner Detector (Segmentation)", lifespan=lifespan)


def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def detect_corners(image: np.ndarray) -> list:
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
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return goals
    if masks is None or masks.xy is None:
        return goals

    masks_xy = masks.xy

    for obj_idx, box in enumerate(boxes):
        if obj_idx >= len(masks_xy):
            continue

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())

        # Контур маски
        polygon = masks_xy[obj_idx]
        if polygon is None or len(polygon) < 3:
            continue

        # Получаем 4 угла
        quad = contour_to_quad(polygon)
        quad = order_quad_points(quad)

        corners = []
        for i, (x, y) in enumerate(quad):
            corners.append({
                "name": POINT_NAMES[i],
                "x": int(x),
                "y": int(y),
            })

        goals.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": conf,
            "corners": corners,
        })

    return goals


def draw_annotations(image: np.ndarray, goals: list) -> np.ndarray:
    annotated = image.copy()
    overlay = image.copy()

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

        pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        for corner in corners:
            cv2.circle(annotated, (corner["x"], corner["y"]), 6, (255, 0, 0), -1)
            cv2.putText(
                annotated,
                corner["name"],
                (corner["x"] + 6, corner["y"] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
    alpha = 0.35
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
    return annotated


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
    import uvicorn
    uvicorn.run("seg_server:app", host="0.0.0.0", port=8000, reload=True)