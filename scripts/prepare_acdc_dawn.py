#!/usr/bin/env python3
"""Merge ACDC and DAWN into one real adverse-weather YOLO dataset.

File này có 3 phần cốt lõi:
1. Đọc ACDC và DAWN từ annotation gốc, rồi map class về 6 class mục tiêu.
2. Gộp hai nguồn dữ liệu và giữ nhãn weather: fog, rain, snow, sand.
3. Chia stratified 70/15/15 theo weather và ghi dataset YOLO cho Stage 3/4.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import (
    IMAGE_SUFFIXES,
    PreparedSample,
    clean_output_dir,
    clamp_box,
    export_yolo_dataset,
    find_image,
    infer_weather_from_path,
    make_yolo_dirs,
    normalize_weather,
    read_image_size,
    stratified_split,
)


def get_box2d_value(box2d: dict, *names: str) -> float | None:
    for name in names:
        if name in box2d:
            return box2d[name]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-images-dir", type=Path, required=True)
    parser.add_argument("--acdc-annotations-json", type=Path, required=True)
    parser.add_argument("--dawn-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weather", default="fog,rain,snow,sand")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def selected_weather_values(raw: str) -> set[str]:
    return {normalize_weather(item) for item in raw.split(",") if item.strip()}


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC ACDC
# Script ưu tiên ACDC object detection ở dạng COCO JSON. Nếu JSON là list frame
# kiểu BDD/Scalabel, script cũng có thể đọc `box2d`.
# ---------------------------------------------------------------------------


def weather_from_record(record: dict, image_path: Path) -> str:
    attributes = record.get("attributes", {})
    weather = normalize_weather(
        record.get("weather")
        or record.get("condition")
        or attributes.get("weather")
        or attributes.get("condition")
    )
    return weather if weather != "unknown" else infer_weather_from_path(image_path)


def load_acdc_coco(images_dir: Path, payload: dict, allowed_weather: set[str]) -> list[PreparedSample]:
    category_by_id = {item["id"]: item["name"] for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in payload.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    samples: list[PreparedSample] = []
    for image_info in payload.get("images", []):
        image_path = find_image(images_dir, image_info["file_name"])
        if image_path is None:
            continue
        weather = weather_from_record(image_info, image_path)
        if weather not in allowed_weather:
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        width = int(image_info.get("width") or image_size[0])
        height = int(image_info.get("height") or image_size[1])

        boxes = []
        for annotation in annotations_by_image.get(image_info["id"], []):
            x, y, w, h = annotation["bbox"]
            box = clamp_box(category_by_id.get(annotation["category_id"]), x, y, x + w, y + h, width, height)
            if box is not None:
                boxes.append(box)
        if boxes:
            samples.append(PreparedSample(image_path, tuple(boxes), weather, "acdc", weather))
    return samples


def load_acdc_bdd_like(images_dir: Path, frames: list[dict], allowed_weather: set[str]) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for frame in frames:
        image_name = frame.get("name") or frame.get("image") or frame.get("file_name")
        if not image_name:
            continue
        image_path = find_image(images_dir, image_name)
        if image_path is None:
            continue
        weather = weather_from_record(frame, image_path)
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
            samples.append(PreparedSample(image_path, tuple(boxes), weather, "acdc", weather))
    return samples


def load_acdc_samples(images_dir: Path, annotations_json: Path, allowed_weather: set[str]) -> list[PreparedSample]:
    payload = json.loads(annotations_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload):
        return load_acdc_coco(images_dir, payload, allowed_weather)
    if isinstance(payload, list):
        return load_acdc_bdd_like(images_dir, payload, allowed_weather)
    raise ValueError("Unsupported ACDC annotation format. Expected COCO dict or BDD-like list.")


# ---------------------------------------------------------------------------
# PHẦN 2: ĐỌC DAWN
# DAWN thường dùng Pascal VOC XML. Mỗi ảnh được ghép với XML cùng tên hoặc XML có
# cùng stem ở thư mục khác.
# ---------------------------------------------------------------------------


def parse_voc_boxes(xml_path: Path, width: int, height: int):
    boxes = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        box = clamp_box(
            obj.findtext("name"),
            float(bbox.findtext("xmin", "0")),
            float(bbox.findtext("ymin", "0")),
            float(bbox.findtext("xmax", "0")),
            float(bbox.findtext("ymax", "0")),
            width,
            height,
        )
        if box is not None:
            boxes.append(box)
    return tuple(boxes)


def load_dawn_samples(raw_dir: Path, allowed_weather: set[str]) -> list[PreparedSample]:
    xml_by_stem: dict[str, list[Path]] = {}
    for xml_path in raw_dir.rglob("*.xml"):
        xml_by_stem.setdefault(xml_path.stem, []).append(xml_path)

    samples: list[PreparedSample] = []
    for image_path in sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        same_dir = image_path.with_suffix(".xml")
        candidates = [same_dir] if same_dir.exists() else xml_by_stem.get(image_path.stem, [])
        if len(candidates) != 1:
            continue

        weather = infer_weather_from_path(image_path.relative_to(raw_dir))
        if weather not in allowed_weather:
            continue

        image_size = read_image_size(image_path)
        if image_size is None:
            continue
        width, height = image_size
        boxes = parse_voc_boxes(candidates[0], width, height)
        if boxes:
            samples.append(PreparedSample(image_path, boxes, weather, "dawn", weather))
    return samples


# ---------------------------------------------------------------------------
# PHẦN 3: GỘP, CHIA SPLIT VÀ XUẤT YOLO
# ACDC và DAWN được gộp trước rồi mới chia split, để train/val/test có phân phối
# weather gần nhau nhất có thể trên toàn bộ dataset thực tế.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    allowed_weather = selected_weather_values(args.weather)

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=("train", "val", "test"))

    acdc_samples = load_acdc_samples(
        args.acdc_images_dir.resolve(), args.acdc_annotations_json.resolve(), allowed_weather
    )
    dawn_samples = load_dawn_samples(args.dawn_raw_dir.resolve(), allowed_weather)
    samples = acdc_samples + dawn_samples
    if not samples:
        raise RuntimeError("No ACDC/DAWN samples matched the requested filters.")

    print(f"ACDC samples: {len(acdc_samples)}")
    print(f"DAWN samples: {len(dawn_samples)}")
    assignments = stratified_split(samples, args.seed, train_ratio=0.70, val_ratio=0.15, stratify_attr="weather")

    common_root = Path("/")
    export_yolo_dataset(
        samples=samples,
        assignments=assignments,
        raw_root=common_root,
        output_dir=output_dir,
        imgsz=args.imgsz,
        splits=("train", "val", "test"),
    )


if __name__ == "__main__":
    main()
