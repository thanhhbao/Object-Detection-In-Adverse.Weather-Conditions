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
"""

from __future__ import annotations

import argparse
import csv
import os
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


def main() -> None:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    keep = None if args.weather.strip().lower() == "all" else {
        w.strip().lower() for w in args.weather.split(",") if w.strip()}

    manifest_rows = []
    counts: dict[str, int] = {}
    for src_split, dst_split in SPLIT_MAP.items():
        img_dir = src / src_split / "images"
        lbl_dir = src / src_split / "labels"
        if not img_dir.exists():
            continue
        (dst / "images" / dst_split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / dst_split).mkdir(parents=True, exist_ok=True)

        for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            weather = weather_from_name(image_path.stem, src_split)
            if keep is not None and weather not in keep:
                continue
            label_path = lbl_dir / f"{image_path.stem}.txt"
            lines = remap_label(label_path.read_text(encoding="utf-8")) if label_path.exists() else []

            out_img = dst / "images" / dst_split / image_path.name
            out_lbl = dst / "labels" / dst_split / f"{image_path.stem}.txt"
            if out_img.exists() or out_img.is_symlink():
                out_img.unlink()
            if args.copy_images:
                import shutil
                shutil.copy2(image_path, out_img)
            else:
                os.symlink(image_path, out_img)
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts[f"{dst_split}/{weather}"] = counts.get(f"{dst_split}/{weather}", 0) + 1
            manifest_rows.append([dst_split, weather, str(out_img), str(out_lbl), len(lines)])

    # dataset.yaml theo 6 lớp của đề tài
    payload = {"path": str(dst), "train": "images/train", "val": "images/val",
               "test": "images/test", "names": {i: n for i, n in enumerate(PROJECT_NAMES)}}
    (dst / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                                      encoding="utf-8")
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "weather", "image", "label", "objects"])
        writer.writerows(manifest_rows)

    print(f"Tổng ảnh: {len(manifest_rows)}")
    print("Phân bố split/weather:", dict(sorted(counts.items())))
    print(f"Đã ghi: {dst}/dataset.yaml + manifest.csv")


if __name__ == "__main__":
    main()
