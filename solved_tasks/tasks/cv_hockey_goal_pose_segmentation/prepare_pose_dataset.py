from pathlib import Path
import random
import shutil
import argparse


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_seg_label_file(label_path: Path):
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"{label_path}: empty label file")

    lines = text.splitlines()

    for line_idx, line in enumerate(lines, start=1):
        parts = line.strip().split()

        # YOLO segmentation:
        # class x1 y1 x2 y2 x3 y3 ...
        if len(parts) < 7:
            raise ValueError(
                f"{label_path}: line {line_idx}: too few values. "
                f"Expected class + at least 3 polygon points, got {len(parts)} values"
            )

        if (len(parts) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}: line {line_idx}: invalid polygon. "
                f"Number of coordinates after class must be even"
            )

        values = [float(x) for x in parts]

        class_id = int(values[0])
        coords = values[1:]

        if class_id != 0:
            raise ValueError(
                f"{label_path}: line {line_idx}: expected class 0, got {class_id}"
            )

        for coord in coords:
            if not (0.0 <= coord <= 1.0):
                raise ValueError(
                    f"{label_path}: line {line_idx}: coordinate out of range [0, 1]: {coord}"
                )


def prepare_dataset(src_dir: Path, dst_dir: Path, train_ratio: float, seed: int):
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )

    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    valid_pairs = []
    missing_labels = []

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        read_seg_label_file(label_path)
        valid_pairs.append((image_path, label_path))

    if missing_labels:
        print("WARNING: images without labels:")
        for name in missing_labels[:20]:
            print(f"  - {name}")
        if len(missing_labels) > 20:
            print(f"  ... and {len(missing_labels) - 20} more")

    if not valid_pairs:
        raise RuntimeError("No valid image-label pairs found.")

    random.seed(seed)
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * train_ratio)

    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    if not val_pairs:
        raise RuntimeError(
            "Validation set is empty. Add more images or reduce --train-ratio."
        )

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        out_images_dir = dst_dir / "images" / split_name
        out_labels_dir = dst_dir / "labels" / split_name

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in pairs:
            shutil.copy2(image_path, out_images_dir / image_path.name)
            shutil.copy2(label_path, out_labels_dir / label_path.name)

    data_yaml = f"""path: {dst_dir.resolve()}

train: images/train
val: images/val

names:
  0: goal
"""

    (dst_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print("YOLO segmentation dataset prepared successfully.")
    print(f"Source: {src_dir.resolve()}")
    print(f"Output: {dst_dir.resolve()}")
    print(f"Total labeled images: {len(valid_pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print(f"YAML: {dst_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        required=True,
        help="Source folder with images/ and labels/"
    )

    parser.add_argument(
        "--dst",
        required=True,
        help="Output YOLO segmentation dataset folder"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    prepare_dataset(
        src_dir=Path(args.src),
        dst_dir=Path(args.dst),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()