"""Shared helpers for converting driving datasets to YOLO format."""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

CLASS_ALIASES = {
    "person": "person",
    "pedestrian": "person",
    "rider": "person",
    "other person": "person",
    "people": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}

WEATHER_ALIASES = {
    "clear": "clear",
    "clear-daytime": "clear",
    "daytime-clear": "clear",
    "partly cloudy": "clear",
    "overcast": "clear",
    "fog": "fog",
    "foggy": "fog",
    "rain": "rain",
    "rainy": "rain",
    "snow": "snow",
    "snowy": "snow",
    "sand": "sand",
    "sandstorm": "sand",
    "dust": "sand",
    "night": "night",
}


@dataclass(frozen=True)
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class PreparedSample:
    image: Path
    boxes: tuple[Box, ...]
    split_key: str
    source: str
    weather: str


def normalize_class(name: str | None) -> str | None:
    if not name:
        return None
    return CLASS_ALIASES.get(name.strip().lower())


def normalize_weather(value: str | None) -> str:
    if not value:
        return "unknown"
    text = value.strip().lower().replace("_", " ").replace("-", " ")
    return WEATHER_ALIASES.get(text, text)


def infer_weather_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for keyword in ("fog", "foggy", "rain", "rainy", "snow", "snowy", "sand", "dust"):
        if any(keyword in part for part in parts):
            return normalize_weather(keyword)
    return "unknown"


def find_image(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.rglob(Path(name).name))
    if len(matches) == 1:
        return matches[0]
    return None


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def clamp_box(
    class_name: str | None, xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> Box | None:
    target = normalize_class(class_name)
    if target is None:
        return None
    if xmin is None or ymin is None or xmax is None or ymax is None:
        return None
    xmin = max(0.0, min(float(xmin), width - 1))
    ymin = max(0.0, min(float(ymin), height - 1))
    xmax = max(0.0, min(float(xmax), width))
    ymax = max(0.0, min(float(ymax), height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(TARGET_CLASSES.index(target), xmin, ymin, xmax, ymax)


def clean_output_dir(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    elif any(output_dir.glob("images/*/*")):
        raise RuntimeError(
            f"{output_dir} already contains processed images. Re-run with --clean "
            "to avoid mixing old and new splits."
        )


def make_yolo_dirs(output_dir: Path, splits: Iterable[str] = ("train", "val", "test")) -> None:
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def stratified_split(
    samples: list[PreparedSample],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    stratify_attr: str = "weather",
) -> dict[int, str]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[getattr(sample, stratify_attr)].append(index)

    assignments: dict[int, str] = {}
    for key, indexes in sorted(groups.items()):
        rng.shuffle(indexes)
        n = len(indexes)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for position, sample_index in enumerate(indexes):
            split = "train" if position < n_train else "val" if position < n_train + n_val else "test"
            assignments[sample_index] = split
        print(f"{key}: total={n}, train={n_train}, val={n_val}, test={n-n_train-n_val}")
    return assignments


def split_train_val(samples: list[PreparedSample], seed: int, train_ratio: float = 0.80) -> dict[int, str]:
    rng = random.Random(seed)
    indexes = list(range(len(samples)))
    rng.shuffle(indexes)
    n_train = int(len(indexes) * train_ratio)
    assignments: dict[int, str] = {}
    for position, sample_index in enumerate(indexes):
        assignments[sample_index] = "train" if position < n_train else "val"
    print(f"total={len(samples)}, train={n_train}, val={len(samples)-n_train}")
    return assignments


def stratified_train_val(
    samples: list[PreparedSample],
    seed: int,
    train_ratio: float = 0.80,
    stratify_attr: str = "weather",
) -> dict[int, str]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[getattr(sample, stratify_attr)].append(index)

    assignments: dict[int, str] = {}
    for key, indexes in sorted(groups.items()):
        rng.shuffle(indexes)
        n_train = int(len(indexes) * train_ratio)
        for position, sample_index in enumerate(indexes):
            assignments[sample_index] = "train" if position < n_train else "val"
        print(f"{key}: total={len(indexes)}, train={n_train}, val={len(indexes)-n_train}")
    return assignments


def letterbox(image, boxes: tuple[Box, ...], size: int):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        size - new_height - top,
        left,
        size - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    transformed = tuple(
        Box(box.class_id, box.xmin * scale + left, box.ymin * scale + top, box.xmax * scale + left, box.ymax * scale + top)
        for box in boxes
    )
    return canvas, transformed


def write_yolo_label(path: Path, boxes: tuple[Box, ...], size: int) -> None:
    lines = []
    for box in boxes:
        x_center = ((box.xmin + box.xmax) / 2) / size
        y_center = ((box.ymin + box.ymax) / 2) / size
        width = (box.xmax - box.xmin) / size
        height = (box.ymax - box.ymin) / size
        lines.append(f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(output_dir: Path, splits: Iterable[str] = ("train", "val", "test")) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
    }
    if "test" in set(splits):
        payload["test"] = "images/test"
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def export_yolo_dataset(
    samples: list[PreparedSample],
    assignments: dict[int, str],
    raw_root: Path,
    output_dir: Path,
    imgsz: int,
    splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    manifest_rows = []
    counts: Counter = Counter()

    for index, sample in enumerate(samples):
        split = assignments[index]
        image = cv2.imread(str(sample.image))
        if image is None:
            print(f"SKIP: cannot read {sample.image}")
            continue
        image, boxes = letterbox(image, sample.boxes, imgsz)
        relative = sample.image.relative_to(raw_root) if sample.image.is_relative_to(raw_root) else sample.image.name
        digest = hashlib.sha1(str(relative).encode()).hexdigest()[:10]
        stem = f"{sample.source}_{sample.weather}_{sample.image.stem}_{digest}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_label(label_path, boxes, imgsz)
        counts[split] += 1
        manifest_rows.append(
            [split, sample.source, sample.weather, str(sample.image), str(image_path), len(boxes)]
        )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "source", "weather", "source_image", "output_image", "objects"])
        writer.writerows(manifest_rows)
    write_dataset_yaml(output_dir, splits=splits)
    print(f"Created {dict(counts)} at {output_dir}")
