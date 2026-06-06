#!/usr/bin/env python3
"""Prepare BDD100K clear-daytime subset for Stage 1 finetuning.

File này có 3 phần cốt lõi:
1. Đọc annotation JSON của BDD100K và lọc ảnh ban ngày, thời tiết đẹp.
2. Map class BDD100K về 6 class mục tiêu và chuyển bbox `box2d` sang dạng YOLO.
3. Chọn tối đa N ảnh, chia train/val và ghi dataset YOLO.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import (
    PreparedSample,
    clean_output_dir,
    clamp_box,
    export_yolo_dataset,
    find_image,
    make_yolo_dirs,
    normalize_weather,
    read_image_size,
    split_train_val,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--timeofday", default="daytime")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC VÀ LỌC BDD100K
# BDD100K detection label là JSON list. Mỗi frame có `name`, `attributes`
# và danh sách `labels`; bbox nằm trong `label["box2d"]`.
# ---------------------------------------------------------------------------


def is_selected_frame(frame: dict, weather: str, timeofday: str) -> bool:
    attributes = frame.get("attributes", {})
    frame_weather = normalize_weather(attributes.get("weather"))
    frame_timeofday = str(attributes.get("timeofday", "")).lower()
    return frame_weather == normalize_weather(weather) and frame_timeofday == timeofday.lower()


def load_bdd_samples(images_dir: Path, labels_json: Path, weather: str, timeofday: str) -> list[PreparedSample]:
    frames = json.loads(labels_json.read_text(encoding="utf-8"))
    samples: list[PreparedSample] = []
    skipped_no_image = 0

    for frame in frames:
        if not is_selected_frame(frame, weather, timeofday):
            continue

        image_path = find_image(images_dir, frame["name"])
        if image_path is None:
            skipped_no_image += 1
            continue
        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        width, height = image_size

        # BDD box2d dùng tọa độ tuyệt đối: x1, y1, x2, y2.
        boxes = []
        for label in frame.get("labels", []):
            box2d = label.get("box2d")
            if not box2d:
                continue
            box = clamp_box(
                label.get("category"),
                box2d["x1"],
                box2d["y1"],
                box2d["x2"],
                box2d["y2"],
                width,
                height,
            )
            if box is not None:
                boxes.append(box)

        if boxes:
            samples.append(
                PreparedSample(
                    image=image_path,
                    boxes=tuple(boxes),
                    split_key="clear",
                    source="bdd100k",
                    weather="clear",
                )
            )

    if skipped_no_image:
        print(f"Skipped frames without image file: {skipped_no_image}")
    return samples


# ---------------------------------------------------------------------------
# PHẦN 2: CHỌN SUBSET VÀ CHIA TRAIN/VAL
# Stage 1 chỉ cần tập clear-daytime để thích ứng miền dữ liệu. Nếu dữ liệu nhiều,
# script random chọn tối đa --max-images bằng seed cố định để tái lập.
# ---------------------------------------------------------------------------


def select_subset(samples: list[PreparedSample], max_images: int, seed: int) -> list[PreparedSample]:
    if max_images <= 0 or len(samples) <= max_images:
        return samples
    rng = random.Random(seed)
    selected = rng.sample(samples, max_images)
    return sorted(selected, key=lambda sample: str(sample.image))


# ---------------------------------------------------------------------------
# PHẦN 3: XUẤT DATASET YOLO
# Output gồm images/train, labels/train, images/val, labels/val, manifest.csv
# và dataset.yaml để dùng trực tiếp với YOLOv8.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=("train", "val"))

    samples = load_bdd_samples(images_dir, args.labels_json.resolve(), args.weather, args.timeofday)
    samples = select_subset(samples, args.max_images, args.seed)
    if not samples:
        raise RuntimeError("No BDD100K samples matched the requested filters.")

    assignments = split_train_val(samples, args.seed, args.train_ratio)
    export_yolo_dataset(
        samples=samples,
        assignments=assignments,
        raw_root=images_dir,
        output_dir=output_dir,
        imgsz=args.imgsz,
        splits=("train", "val"),
    )


if __name__ == "__main__":
    main()
