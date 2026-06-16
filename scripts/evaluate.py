#!/usr/bin/env python3
"""Evaluate one trained Ultralytics run from the same YAML config used for training.

File này có 3 phần cốt lõi:
1. Đọc config thí nghiệm và tìm checkpoint `runs/<name>/weights/best.pt`.
2. Chạy Ultralytics validation trên split được chọn: thường là `val` hoặc `test`.
3. Lưu metrics chuẩn hóa thành JSON để sau này gom bảng so sánh.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--name", default=None, help="Optional evaluation folder name.")
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        help="Override the run name (config name), e.g. a per-seed run folder.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def first_image_from_split(data_yaml: Path, split: str) -> Path | None:
    """Find one image from the dataset split for a small batch-1 latency check."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    split_value = data.get(split)
    if not split_value:
        return None

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path

    if split_path.is_file() and split_path.suffix.lower() in IMAGE_SUFFIXES:
        return split_path

    if split_path.is_dir():
        for path in sorted(split_path.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                return path

    return None


def benchmark_batch1(model: YOLO, image: Path | None, config: dict[str, Any]) -> tuple[float | None, float | None]:
    if image is None:
        return None, None

    kwargs = dict(imgsz=config["imgsz"], device=config["device"], verbose=False)
    for _ in range(config.get("benchmark_warmup", 10)):
        model.predict(str(image), **kwargs)

    synchronize(str(config["device"]))
    start = time.perf_counter()
    iterations = config.get("benchmark_iterations", 50)
    for _ in range(iterations):
        model.predict(str(image), **kwargs)
    synchronize(str(config["device"]))

    latency_ms = (time.perf_counter() - start) * 1000 / iterations
    return latency_ms, 1000 / latency_ms


def per_class_ap(model: YOLO, metrics: Any) -> dict[str, dict[str, float | None]]:
    """Per-class AP so small/rare classes (bicycle, motorcycle, bus) stay visible.

    DAWN is small and imbalanced, so an aggregate mAP can hide weak classes.
    `box.maps` is mAP50-95 indexed by class id; `box.ap50` is mAP50 aligned to
    `box.ap_class_index` (only classes that appear in the split)."""
    names = model.names
    maps = metrics.box.maps
    map50_by_id = {
        int(class_id): float(metrics.box.ap50[position])
        for position, class_id in enumerate(metrics.box.ap_class_index)
    }
    per_class: dict[str, dict[str, float | None]] = {}
    for class_id, name in names.items():
        index = int(class_id)
        per_class[name] = {
            "map50": map50_by_id.get(index),
            "map50_95": float(maps[index]) if index < len(maps) else None,
        }
    return per_class


def main() -> None:
    args = parse_args()
    # Register custom modules (e.g. CBAMResearch) so ablation checkpoints that
    # contain them can be deserialized by Ultralytics' loader.
    register_custom_modules()
    config = load_experiment_config(args.config)
    if args.run_name:
        config["name"] = args.run_name
    checkpoint = Path(args.weights) if args.weights else experiment_checkpoint(config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = YOLO(str(checkpoint))
    data_yaml = resolve_from_root(config["data"])
    eval_name = args.name or f"{args.split}_eval"

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        conf=config.get("conf", 0.001),
        iou=config.get("iou", 0.7),
        project=str(experiment_run_dir(config)),
        name=eval_name,
        exist_ok=True,
        plots=True,
    )

    image = first_image_from_split(data_yaml, args.split)
    latency_ms, fps = benchmark_batch1(model, image, config)
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    per_class = per_class_ap(model, metrics)

    result = {
        "run": config["name"],
        "config": args.config,
        "weights": str(checkpoint),
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "ultralytics_inference_ms": float(metrics.speed["inference"]),
        "ultralytics_inference_fps": 1000 / float(metrics.speed["inference"]),
        "inference_ms_batch1": latency_ms,
        "fps_batch1": fps,
        "parameters": int(parameters),
        "per_class": per_class,
    }

    output = experiment_run_dir(config) / f"{args.split}_metrics.json"
    write_json(output, result)
    print(result)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
