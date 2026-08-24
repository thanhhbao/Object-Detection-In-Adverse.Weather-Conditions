#!/usr/bin/env python3
"""Chuẩn bị XWOD (đã ở dạng YOLO) về 6 lớp của đề tài + nhãn weather.

XWOD có sẵn train/valid/test (images|labels), thứ tự lớp:
  0 person, 1 car, 2 truck, 3 motorcycle, 4 bus, 5 bike
Đề tài dùng:
  0 person, 1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck
=> remap theo TÊN: {0:0, 1:2, 2:5, 3:3, 4:4, 5:1}

Nhãn thời tiết nằm trong tên file (vd fog_test_00077, heavy_rain_train_..). Script
trích weather từ tên, chuẩn hóa về fog/rain/snow/sand (heavy_rain->rain, dust->sand),
ghi manifest.csv để evaluate_by_weather chạy được, và (tùy chọn) lọc thời tiết.

Ví dụ:
  python scripts/prepare_xwod.py --src /content/XWOD/dataset --dst /content/xwod_6cls_yolo
  # giữ tất cả thời tiết:
  python scripts/prepare_xwod.py --src ... --dst ... --weather all
  # chia lại 6-1-3 (train-val-test) thay vì dùng split gốc:
  python scripts/prepare_xwod.py --src ... --dst ... --resplit --train-ratio 0.6 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REMAP = {0: 0, 1: 2, 2: 5, 3: 3, 4: 4, 5: 1}  # XWOD id -> project id
PROJECT_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
WEATHER_NORMALIZE = {"heavy_rain": "rain", "rain": "rain", "fog": "fog", "snow": "snow",
                     "dust": "sand", "sand": "sand", "flooding": "flooding",
                     "tornado": "tornado", "wildfire": "wildfire"}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Thư mục XWOD/dataset")
    parser.add_argument("--dst", type=Path, required=True, help="Thư mục đầu ra YOLO")
    parser.add_argument("--weather", default="fog,rain,snow,sand",
                        help="Danh sách thời tiết giữ lại (đã chuẩn hóa), hoặc 'all'")
    parser.add_argument("--copy-images", action="store_true", help="Copy ảnh thay vì symlink")
    parser.add_argument("--resplit", action="store_true",
                        help="Bỏ split gốc, gộp tất cả và chia lại theo --train-ratio/--val-ratio")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def weather_from_name(stem: str, split: str) -> str:
    raw = re.split(rf"_(?:train|valid|test)_", stem)[0]
    return WEATHER_NORMALIZE.get(raw.lower(), raw.lower())


def remap_label(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        new_id = REMAP.get(int(float(parts[0])))
        if new_id is None:
            continue
        out.append(" ".join([str(new_id), *parts[1:]]))
    return out


def _write_sample(image_path: Path, label_path: Path | None, dst_split: str,
                  dst: Path, copy_images: bool, weather: str,
                  manifest_rows: list, counts: dict) -> None:
    lines = remap_label(label_path.read_text(encoding="utf-8")) if (label_path and label_path.exists()) else []
    out_img = dst / "images" / dst_split / image_path.name
    out_lbl = dst / "labels" / dst_split / f"{image_path.stem}.txt"
    (dst / "images" / dst_split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
    if out_img.exists() or out_img.is_symlink():
        out_img.unlink()
    if copy_images:
        import shutil
        shutil.copy2(image_path, out_img)
    else:
        os.symlink(image_path, out_img)
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    counts[f"{dst_split}/{weather}"] = counts.get(f"{dst_split}/{weather}", 0) + 1
    manifest_rows.append([dst_split, weather, str(out_img), str(out_lbl), len(lines)])


def main() -> None:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    keep = None if args.weather.strip().lower() == "all" else {
        w.strip().lower() for w in args.weather.split(",") if w.strip()}

    manifest_rows: list = []
    counts: dict[str, int] = {}

    if args.resplit:
        # Gộp tất cả ảnh từ mọi split gốc, sau đó chia lại theo tỉ lệ stratified by weather
        all_items: list[tuple[Path, Path | None, str]] = []  # (img, lbl, weather)
        for src_split in SPLIT_MAP:
            img_dir = src / src_split / "images"
            lbl_dir = src / src_split / "labels"
            if not img_dir.exists():
                continue
            for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                weather = weather_from_name(image_path.stem, src_split)
                if keep is not None and weather not in keep:
                    continue
                label_path = lbl_dir / f"{image_path.stem}.txt"
                all_items.append((image_path, label_path if label_path.exists() else None, weather))

        # Stratified split by weather
        rng = random.Random(args.seed)
        by_weather: dict[str, list] = {}
        for item in all_items:
            by_weather.setdefault(item[2], []).append(item)

        assignments: dict[int, str] = {}
        indexed = list(enumerate(all_items))
        idx_map = {id(item): i for i, item in indexed}

        for weather, group in sorted(by_weather.items()):
            rng.shuffle(group)
            n = len(group)
            n_train = int(n * args.train_ratio)
            n_val   = int(n * args.val_ratio)
            for j, item in enumerate(group):
                if j < n_train:
                    split = "train"
                elif j < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                assignments[idx_map[id(item)]] = split
            print(f"{weather}: total={n}, train={n_train}, val={n_val}, test={n-n_train-n_val}")

        for i, (image_path, label_path, weather) in indexed:
            _write_sample(image_path, label_path, assignments[i], dst,
                          args.copy_images, weather, manifest_rows, counts)
    else:
        # Dùng split gốc của XWOD
        for src_split, dst_split in SPLIT_MAP.items():
            img_dir = src / src_split / "images"
            lbl_dir = src / src_split / "labels"
            if not img_dir.exists():
                continue
            for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                weather = weather_from_name(image_path.stem, src_split)
                if keep is not None and weather not in keep:
                    continue
                label_path = lbl_dir / f"{image_path.stem}.txt"
                _write_sample(image_path, label_path if label_path.exists() else None,
                              dst_split, dst, args.copy_images, weather, manifest_rows, counts)

    # dataset.yaml theo 6 lớp của đề tài
    payload = {"path": str(dst), "train": "images/train", "val": "images/val",
               "test": "images/test", "names": {i: n for i, n in enumerate(PROJECT_NAMES)}}
    (dst / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                                      encoding="utf-8")
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "weather", "image", "label", "objects"])
        writer.writerows(manifest_rows)

    print(f"\nTổng ảnh: {len(manifest_rows)}")
    print("Phân bố split/weather:", dict(sorted(counts.items())))
    print(f"Đã ghi: {dst}/dataset.yaml + manifest.csv")


if __name__ == "__main__":
    main()
