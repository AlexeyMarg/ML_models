from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs" / "segment" / "train" / "weights" / "best.pt"

IMAGE_PATH = BASE_DIR / "yolo_seg_dataset" / "images" / "val" / "f5326829-133.png"

OUTPUT_PATH = BASE_DIR / "seg_result_quad.jpg"


def contour_to_quad(polygon: np.ndarray) -> np.ndarray:
    """
    Преобразует контур маски YOLO segmentation в 4 точки.
    Возвращает массив shape: (4, 2)
    """
    polygon = polygon.astype(np.int32)

    peri = cv2.arcLength(polygon, True)

    for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15]:
        approx = cv2.approxPolyDP(
            polygon,
            eps_factor * peri,
            True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2)

    # fallback: if approxPolyDP didn't return 4 points
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки:
    top_left, top_right, bottom_right, bottom_left
    """
    points = points.astype(np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.int32
    )


def main():
    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No goal detected.")
        return

    if result.masks is None:
        print("No masks detected.")
        return

    boxes = result.boxes
    masks_xy = result.masks.xy

    overlay = image.copy()

    for obj_idx, box in enumerate(boxes):
        if obj_idx >= len(masks_xy):
            continue

        conf = float(box.conf[0].cpu().numpy())

        # bbox
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"goal {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # исходный контур маски
        polygon = masks_xy[obj_idx]

        if polygon is None or len(polygon) < 3:
            continue

        polygon = polygon.astype(np.int32)

        # получаем ровно 4 точки
        quad = contour_to_quad(polygon)
        quad = order_quad_points(quad)

        # заливка четырёхугольника
        cv2.fillPoly(
            overlay,
            [quad],
            (0, 255, 255),
        )

        # контур четырёхугольника
        cv2.polylines(
            image,
            [quad],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

        point_names = [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ]

        for i, (x, y) in enumerate(quad):
            cv2.circle(
                image,
                (int(x), int(y)),
                6,
                (255, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                point_names[i],
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        print(f"Detected goal {obj_idx}: conf={conf:.3f}")
        print("Quad points:")
        for name, (x, y) in zip(point_names, quad):
            print(f"  {name}: ({int(x)}, {int(y)})")

    alpha = 0.35
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    #cv2.imwrite(str(OUTPUT_PATH), image)
    #print(f"Saved result to: {OUTPUT_PATH}")

    cv2.imshow("YOLO Segmentation Quad Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()