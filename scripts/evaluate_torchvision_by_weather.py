#!/usr/bin/env python3
"""Per-weather evaluation for a Faster R-CNN run (TorchVision branch).

Đối ứng của `scripts/evaluate_by_weather.py` (vốn dành cho YOLO/Ultralytics) cho
nhánh 2-stage, để bảng so sánh theo thời tiết phủ đủ cả 4 model.

Đọc `manifest.csv` của DAWN, tách split được chọn theo weather, rồi chạy cùng
hàm `evaluate_detector` trên từng nhóm. Xuất `{split}_metrics_by_weather.json`
cùng cấu trúc với bản YOLO (khóa `by_weather`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from torch.utils.data import DataLoader

from dawn_ablation.common import (
    experiment_checkpoint,
    experiment_run_dir,
    load_experiment_config,
    resolve_from_root,
    write_json,
)
from dawn_ablation.torchvision_detection import (
    YoloDetectionDataset,
    build_fasterrcnn,
    collate_fn,
    evaluate_detector,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def group_images_by_weather(manifest: Path, split: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        image_column = next((name for name in ("image", "output_image") if name in fields), None)
        if image_column is None or "weather" not in fields or "split" not in fields:
            raise KeyError(f"manifest.csv must have split/weather and an image column; got {fields}")
        for row in reader:
            if row["split"] == split:
                groups[row["weather"]].append(row[image_column])
    return dict(sorted(groups.items()))


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(config.get("device", "0"))

    checkpoint_path = Path(args.weights) if args.weights else experiment_checkpoint(config, "best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    data_yaml = resolve_from_root(config["data"])
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path", data_yaml.parent))
    manifest = Path(args.manifest) if args.manifest else dataset_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest with weather labels: {manifest}")

    groups = group_images_by_weather(manifest, args.split)
    if not groups:
        raise RuntimeError(f"No images for split '{args.split}' in {manifest}")

    model = build_fasterrcnn(int(config["num_classes"]), int(config["imgsz"]), coco_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device)

    per_weather = {}
    for weather, images in groups.items():
        dataset = YoloDetectionDataset(images=images)
        loader = DataLoader(
            dataset, batch_size=int(config["batch"]), shuffle=False,
            num_workers=2, collate_fn=collate_fn,
        )
        metrics = evaluate_detector(model, loader, device)
        per_weather[weather] = {
            "images": len(images),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
        }
        print(
            f"{weather:>8}: {len(images):4d} imgs | "
            f"mAP50 {metrics['map50']:.4f} | mAP50-95 {metrics['map50_95']:.4f}"
        )

    result = {
        "run": config["name"],
        "split": args.split,
        "weights": str(checkpoint_path),
        "by_weather": per_weather,
    }
    output = experiment_run_dir(config) / f"{args.split}_metrics_by_weather.json"
    write_json(output, result)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
