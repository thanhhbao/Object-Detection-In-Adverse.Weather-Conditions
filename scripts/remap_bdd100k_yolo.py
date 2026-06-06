#!/usr/bin/env python3
"""Remap a ready-made BDD100K YOLO dataset to the 6 target classes.

File này có 3 phần cốt lõi:
1. Đọc dataset YOLO gốc có cấu trúc `train/val/test/images|labels`.
2. Remap 10 class BDD100K-YOLO về 6 class mục tiêu, bỏ các object không dùng.
3. Copy ảnh và ghi label mới vào dataset YOLO sạch cho Stage 1.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import IMAGE_SUFFIXES, TARGET_CLASSES, clean_output_dir, make_yolo_dirs, write_dataset_yaml

# Class order của Kaggle `a7madmostafa/bdd100k-yolo`.
# Nguồn dataset ghi: 0 person, 1 rider, 2 car, 3 bus, 4 truck, 5 bike,
# 6 motor, 7 traffic light, 8 traffic sign, 9 train.
SOURCE_TO_TARGET = {
    0: 0,  # person -> person
    1: 0,  # rider -> person
    2: 2,  # car -> car
    3: 4,  # bus -> bus
    4: 5,  # truck -> truck
    5: 1,  # bike -> bicycle
    6: 3,  # motor -> motorcycle
    # 7 traffic light -> bỏ
    # 8 traffic sign  -> bỏ
    # 9 train         -> bỏ
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=5000)
    parser.add_argument("--max-val-images", type=int, default=1000)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC DATASET YOLO GỐC
# Dataset bạn tải có dạng:
# bdd100k_yolo_raw/{train,val,test}/images và labels. Với mỗi ảnh, label YOLO
# cùng tên nằm trong thư mục labels tương ứng.
# ---------------------------------------------------------------------------


def list_image_label_pairs(raw_dir: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = raw_dir / split / "images"
    label_dir = raw_dir / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return []

    pairs = []
    for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def select_subset(pairs: list[tuple[Path, Path]], max_items: int, seed: int) -> list[tuple[Path, Path]]:
    if max_items <= 0 or len(pairs) <= max_items:
        return pairs
    rng = random.Random(seed)
    selected = rng.sample(pairs, max_items)
    return sorted(selected, key=lambda item: str(item[0]))


# ---------------------------------------------------------------------------
# PHẦN 2: REMAP LABEL
# File label YOLO gốc có dòng: class_id x_center y_center width height.
# Script chỉ đổi class_id theo SOURCE_TO_TARGET và bỏ dòng thuộc class không dùng.
# ---------------------------------------------------------------------------


def remap_label_file(label_path: Path) -> tuple[list[str], Counter]:
    output_lines: list[str] = []
    removed: Counter = Counter()

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        source_id = int(float(parts[0]))
        target_id = SOURCE_TO_TARGET.get(source_id)
        if target_id is None:
            removed[source_id] += 1
            continue

        output_lines.append(" ".join([str(target_id), *parts[1:]]))
    return output_lines, removed


# ---------------------------------------------------------------------------
# PHẦN 3: GHI DATASET YOLO MỚI
# Ảnh có thể được copy thật sang output hoặc symlink để tiết kiệm dung lượng.
# Label mới luôn được ghi lại với 6 class mục tiêu và data.yaml mới.
# ---------------------------------------------------------------------------


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def export_split(
    pairs: list[tuple[Path, Path]],
    split: str,
    output_dir: Path,
    copy_images: bool,
) -> tuple[int, Counter, list[list[str]]]:
    removed_total: Counter = Counter()
    manifest_rows: list[list[str]] = []
    kept = 0

    for image_path, label_path in pairs:
        lines, removed = remap_label_file(label_path)
        removed_total.update(removed)
        if not lines:
            continue

        out_image = output_dir / "images" / split / image_path.name
        out_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        link_or_copy_image(image_path, out_image, copy_images)
        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

        kept += 1
        manifest_rows.append([split, str(image_path), str(out_image), str(out_label), str(len(lines))])

    return kept, removed_total, manifest_rows


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    splits = ("train", "val", "test") if args.include_test else ("train", "val")

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=splits)

    train_pairs = select_subset(list_image_label_pairs(raw_dir, "train"), args.max_train_images, args.seed)
    val_pairs = select_subset(list_image_label_pairs(raw_dir, "val"), args.max_val_images, args.seed)
    split_pairs = {"train": train_pairs, "val": val_pairs}
    if args.include_test:
        split_pairs["test"] = list_image_label_pairs(raw_dir, "test")

    all_manifest_rows: list[list[str]] = []
    all_removed: Counter = Counter()
    for split, pairs in split_pairs.items():
        kept, removed, rows = export_split(pairs, split, output_dir, args.copy_images)
        all_manifest_rows.extend(rows)
        all_removed.update(removed)
        print(f"{split}: input={len(pairs)}, kept={kept}")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "source_image", "output_image", "output_label", "objects"])
        writer.writerows(all_manifest_rows)

    write_dataset_yaml(output_dir, splits=splits)
    print(f"classes: {TARGET_CLASSES}")
    print(f"removed source class counts: {dict(all_removed)}")
    print(f"saved: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
