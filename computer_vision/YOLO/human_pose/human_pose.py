"""
YOLO Pose Demo: скелет человека на камере, видео или изображении.

Примеры запуска:
    python human_pose.py --source 0 --show
    python human_pose.py --source video.mp4 --save output.mp4
    python human_pose.py --source video.mp4 --save output.mp4 --codec avc1
    python human_pose.py --source photo.jpg --save result.jpg
"""

import cv2
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
import sys

MODEL_NAME = 'yolo26n-pose.pt'
CONFIDENCE_THRESHOLD = 0.5 
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          
    (5, 7), (7, 9), (6, 8), (8, 10),          
    (5, 6), (5, 11), (6, 12), (11, 12),       
    (11, 13), (13, 15), (12, 14), (14, 16)    
]
COLOR_LINE = (255, 255, 0)  
COLOR_POINT = (0, 255, 0)   
COLOR_TEXT = (0, 255, 255) 

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обнаружение скелета человека через YOLO-Pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Источник: '0' (камера), путь к видео или путь к изображению.",
    )
    parser.add_argument(
        "--model", default="yolo26n-pose.pt",
        help="Веса модели: yolov8n-pose.pt / yolo11n-pose.pt / yolo26n-pose.pt и т.д.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Порог уверенности для отрисовки точки.",
    )
    parser.add_argument(
        "--save", default=None,
        help="Путь для сохранения результата (видеофайл или изображение).",
    )
    parser.add_argument(
        "--codec", type=codec_type, default="mp4v",
        help="FourCC-кодек для записи видео (4 символа). "
             "Популярные: mp4v, avc1 (H.264), XVID, MJPG.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Показывать окно с результатом (для headless-режима отключено).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Устройство: 'cpu', 'cuda', '0', 'mps'. По умолчанию — авто.",
    )
    return parser.parse_args()


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def draw_pose(frame, result, conf_threshold: float):
    """Рисует скелет и точки на кадре."""
    if result.keypoints is None or result.keypoints.xy is None:
        return frame

    for person_idx in range(len(result.keypoints.xy)):
        kps = result.keypoints.xy[person_idx].cpu().numpy()
        confs = result.keypoints.conf[person_idx].cpu().numpy()

        for a, b in SKELETON_CONNECTIONS:
            if confs[a] > conf_threshold and confs[b] > conf_threshold:
                cv2.line(
                    frame,
                    (int(kps[a][0]), int(kps[a][1])),
                    (int(kps[b][0]), int(kps[b][1])),
                    COLOR_LINE, 2,
                )

        for i in range(len(kps)):
            if confs[i] > conf_threshold:
                cv2.circle(
                    frame,
                    (int(kps[i][0]), int(kps[i][1])),
                    4, COLOR_POINT, -1,
                )
    return frame


def codec_type(value: str) -> str:
    """Валидатор для --codec: должен быть ровно 4 символа."""
    if len(value) != 4:
        raise argparse.ArgumentTypeError(
            f"Кодек должен быть ровно 4 символа (FourCC), получено: {value!r}"
        )
    return value

def open_source(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            sys.exit(f'Не удалось открыть камеру с индексом {source}.')
        return cap, 'stream', None

    path = Path(source)
    if not path.exists():
        sys.exit(f'Файл не найден: {source}')

    if is_image_path(source):
        img = cv2.imread(str(path))
        if img is None:
            sys.exit(f'Не удалось прочитать изображение: {source}')
        return None, 'image', img

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f'Не удалось открыть видеофайл: {source}')
    return cap, 'video', None


def make_writer(path: str, codec: str, fps: float, size: tuple):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        sys.exit(
            f"Не удалось открыть VideoWriter с кодеком '{codec}' "
            f"для файла {path}. Попробуйте другой --codec (например, mp4v, MJPG)."
        )
    return writer


def process_image(model, img, args):
    result = model(img, conf=args.conf, device=args.device, verbose=False)[0]
    out = draw_pose(img.copy(), result, args.conf)

    if args.save:
        cv2.imwrite(args.save, out)
        print(f"Сохранено: {args.save}")
    if args.show:
        cv2.imshow("YOLO Pose", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def process_stream(cap, model, args, is_camera: bool):
    writer = None
    if args.save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = make_writer(args.save, args.codec, fps_src, (w, h))
        print(f"Пишу видео: {args.save} (кодек: {args.codec}, {w}x{h} @ {fps_src:.1f} fps)")

    prev_t = time.time()
    frame_count = 0
    print("Обработка... Нажмите 'q' для остановки (если окно открыто).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, conf=args.conf, device=args.device, verbose=False)[0]
        frame = draw_pose(frame, result, args.conf)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("YOLO Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if is_camera is False and frame_count % 50 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Видео сохранено: {args.save}")
    if args.show:
        cv2.destroyAllWindows()
    


if __name__ == "__main__":
    args = parse_args()
    print(f"Загрузка модели: {args.model}")
    model = YOLO(args.model)
    cap, kind, img = open_source(args.source)
    
    if kind == "image":
        process_image(model, img, args)
    else:
        process_stream(cap, model, args, is_camera=args.source.isdigit())
    
    
