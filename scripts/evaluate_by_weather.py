#!/usr/bin/env python3
"""Evaluate one Ultralytics run separately for each weather condition.

Đề tài tập trung vào "thời tiết bất lợi", nên một con số mAP tổng là chưa đủ:
fog/rain/sand/snow/night có độ khó rất khác nhau. Script này đọc `manifest.csv`
do `scripts/prepare_dawn.py` sinh ra (có cột `weather`), tách split được chọn
thành từng nhóm thời tiết, rồi chạy Ultralytics validation trên mỗi nhóm.

Kết quả lưu `{split}_metrics_by_weather.json` trong thư mục run để báo cáo bảng
mAP theo điều kiện thời tiết.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from ultralytics import YOLO

from dawn_ablation.common import (
    experiment_checkpoint,
    experiment_run_dir,
    load_experiment_config,
    register_custom_modules,
    resolve_from_root,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override manifest.csv path (default: <dataset_root>/manifest.csv).",
    )
    return parser.parse_args()


def group_images_by_weather(manifest: Path, split: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                groups[row["weather"]].append(row["output_image"])
    return dict(sorted(groups.items()))


def main() -> None:
    args = parse_args()
    register_custom_modules()
    config = load_experiment_config(args.config)

    checkpoint = Path(args.weights) if args.weights else experiment_checkpoint(config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    data_yaml = resolve_from_root(config["data"])
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path", data_yaml.parent))

    manifest = Path(args.manifest) if args.manifest else dataset_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing manifest with weather labels: {manifest}. "
            "Per-weather evaluation needs a DAWN dataset prepared by prepare_dawn.py."
        )

    groups = group_images_by_weather(manifest, args.split)
    if not groups:
        raise RuntimeError(f"No images found for split '{args.split}' in {manifest}")

    model = YOLO(str(checkpoint))
    work_dir = experiment_run_dir(config) / f"{args.split}_weather_eval"
    work_dir.mkdir(parents=True, exist_ok=True)

    per_weather = {}
    for weather, images in groups.items():
        # Ultralytics reads a .txt list of image paths; labels are resolved by
        # swapping images/ -> labels/, which matches the prepared layout.
        listing = work_dir / f"{weather}.txt"
        listing.write_text("\n".join(images) + "\n", encoding="utf-8")

        subset_yaml = work_dir / f"{weather}.yaml"
        subset_yaml.write_text(
            yaml.safe_dump(
                {"path": str(dataset_root), args.split: str(listing), "names": data["names"]},
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        metrics = model.val(
            data=str(subset_yaml),
            split=args.split,
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=config["device"],
            conf=config.get("conf", 0.001),
            iou=config.get("iou", 0.7),
            project=str(work_dir),
            name=weather,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        per_weather[weather] = {
            "images": len(images),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        print(
            f"{weather:>8}: {len(images):4d} imgs | "
            f"mAP50 {per_weather[weather]['map50']:.4f} | "
            f"mAP50-95 {per_weather[weather]['map50_95']:.4f}"
        )

    result = {
        "run": config["name"],
        "split": args.split,
        "weights": str(checkpoint),
        "by_weather": per_weather,
    }
    output = experiment_run_dir(config) / f"{args.split}_metrics_by_weather.json"
    write_json(output, result)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
