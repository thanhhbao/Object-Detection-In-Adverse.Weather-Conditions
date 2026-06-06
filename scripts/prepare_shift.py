#!/usr/bin/env python3
"""Prepare SHIFT synthetic adverse-weather subset for Stage 2 finetuning.

File này có 3 phần cốt lõi:
1. Đọc annotation SHIFT ở dạng BDD/Scalabel JSON hoặc COCO JSON.
2. Lọc các mẫu thời tiết nhân tạo cần dùng, map class về 6 class mục tiêu.
3. Chọn tối đa N ảnh, chia train/val theo weather và ghi dataset YOLO.
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
    infer_weather_from_path,
    make_yolo_dirs,
    normalize_weather,
    read_image_size,
    stratified_train_val,
)


def get_box2d_value(box2d: dict, *names: str) -> float | None:
    for name in names:
        if name in box2d:
            return box2d[name]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--weather", default="fog,rain")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC ANNOTATION SHIFT
# SHIFT thường được chia sẻ theo format gần BDD/Scalabel; một số bản export có
# thể là COCO JSON. Script tự nhận dạng bằng cấu trúc JSON bên ngoài.
# ---------------------------------------------------------------------------


def selected_weather_values(raw: str) -> set[str]:
    return {normalize_weather(item) for item in raw.split(",") if item.strip()}


def weather_from_frame(frame: dict, image_path: Path) -> str:
    attributes = frame.get("attributes", {})
    candidates = [
        attributes.get("weather"),
        attributes.get("condition"),
        attributes.get("domain"),
        frame.get("weather"),
        frame.get("condition"),
    ]
    for candidate in candidates:
        weather = normalize_weather(candidate)
        if weather != "unknown":
            return weather
    return infer_weather_from_path(image_path)


def load_bdd_like_shift(images_dir: Path, frames: list[dict], allowed_weather: set[str]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for frame in frames:
        image_name = frame.get("name") or frame.get("image") or frame.get("file_name")
        if not image_name:
            continue
        image_path = find_image(images_dir, image_name)
        if image_path is None:
            continue
        weather = weather_from_frame(frame, image_path)
        if weather not in allowed_weather:
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        width, height = image_size

        boxes = []
        for label in frame.get("labels", []):
            box2d = label.get("box2d") or label.get("box")
            if not box2d:
                continue
            box = clamp_box(
                label.get("category") or label.get("label"),
                get_box2d_value(box2d, "x1", "xmin"),
                get_box2d_value(box2d, "y1", "ymin"),
                get_box2d_value(box2d, "x2", "xmax"),
                get_box2d_value(box2d, "y2", "ymax"),
                width,
                height,
            )
            if box is not None:
                boxes.append(box)

        if boxes:
            samples.append(PreparedSample(image_path, tuple(boxes), weather, "shift", weather))
    return samples


def load_coco_shift(images_dir: Path, coco: dict, allowed_weather: set[str]) -> list[PreparedSample]:
    category_by_id = {item["id"]: item["name"] for item in coco.get("categories", [])}
    image_by_id = {item["id"]: item for item in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    samples: list[PreparedSample] = []
    for image_id, image_info in image_by_id.items():
        image_path = find_image(images_dir, image_info["file_name"])
        if image_path is None:
            continue
        weather = normalize_weather(
            image_info.get("weather") or image_info.get("condition") or infer_weather_from_path(image_path)
        )
        if weather not in allowed_weather:
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        width = int(image_info.get("width") or image_size[0])
        height = int(image_info.get("height") or image_size[1])
        boxes = []
        for annotation in annotations_by_image.get(image_id, []):
            x, y, w, h = annotation["bbox"]
            box = clamp_box(category_by_id.get(annotation["category_id"]), x, y, x + w, y + h, width, height)
            if box is not None:
                boxes.append(box)
        if boxes:
            samples.append(PreparedSample(image_path, tuple(boxes), weather, "shift", weather))
    return samples


def load_shift_samples(images_dir: Path, annotations_json: Path, allowed_weather: set[str]) -> list[PreparedSample]:
    payload = json.loads(annotations_json.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return load_bdd_like_shift(images_dir, payload, allowed_weather)
    if isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload):
        return load_coco_shift(images_dir, payload, allowed_weather)
    raise ValueError("Unsupported SHIFT annotation format. Expected BDD-like list or COCO dict.")


# ---------------------------------------------------------------------------
# PHẦN 2: CHỌN SUBSET
# Stage 2 dùng khoảng 5.000 ảnh fog/rain giả lập. Nếu JSON nhiều hơn, lấy mẫu
# ngẫu nhiên bằng seed cố định để lần chạy sau vẫn tạo cùng subset.
# ---------------------------------------------------------------------------


def select_subset(samples: list[PreparedSample], max_images: int, seed: int) -> list[PreparedSample]:
    if max_images <= 0 or len(samples) <= max_images:
        return samples
    rng = random.Random(seed)
    selected = rng.sample(samples, max_images)
    return sorted(selected, key=lambda sample: str(sample.image))


# ---------------------------------------------------------------------------
# PHẦN 3: XUẤT DATASET YOLO
# Chia train/val theo weather để fog và rain không bị lệch quá nhiều giữa hai
# split, sau đó ghi ảnh, label, manifest.csv và dataset.yaml.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=("train", "val"))

    samples = load_shift_samples(images_dir, args.annotations_json.resolve(), selected_weather_values(args.weather))
    samples = select_subset(samples, args.max_images, args.seed)
    if not samples:
        raise RuntimeError("No SHIFT samples matched the requested filters.")

    assignments = stratified_train_val(samples, args.seed, args.train_ratio, stratify_attr="weather")
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
