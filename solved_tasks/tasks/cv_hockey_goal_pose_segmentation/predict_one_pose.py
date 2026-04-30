from ultralytics import YOLO
from pathlib import Path
import cv2


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs" / "pose" / "train" / "weights" / "best.pt"

IMAGE_PATH = BASE_DIR / "yolo_pose_dataset" / "images" / "val" / "41b9ec4d-132.png"

OUTPUT_PATH = BASE_DIR / "pose_result.jpg"

POINT_NAMES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]


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

    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or keypoints is None or len(boxes) == 0:
        print("No goal detected.")
        return

    for obj_idx, box in enumerate(boxes):
        # bbox в пикселях: x1, y1, x2, y2
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        conf = float(box.conf[0].cpu().numpy())

        # Рисуем bbox
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

        # keypoints в пикселях: shape = [num_objects, 4, 2]
        kpts_xy = keypoints.xy[obj_idx].cpu().numpy()

        # confidence/visibility точек, если доступно
        kpts_conf = None
        if keypoints.conf is not None:
            kpts_conf = keypoints.conf[obj_idx].cpu().numpy()

        for point_idx, (x, y) in enumerate(kpts_xy):
            x, y = int(x), int(y)

            if kpts_conf is not None:
                kp_conf = float(kpts_conf[point_idx])
                if kp_conf < 0.25:
                    continue

            cv2.circle(
                image,
                (x, y),
                5,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                image,
                POINT_NAMES[point_idx],
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    #cv2.imwrite(str(OUTPUT_PATH), image)
    print(f"Saved result to: {OUTPUT_PATH}")

    cv2.imshow("YOLO Pose Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()