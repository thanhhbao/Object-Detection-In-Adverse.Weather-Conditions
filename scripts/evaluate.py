#!/usr/bin/env python3
"""Evaluate best checkpoint on the held-out test split and benchmark latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from ultralytics import YOLO

from dawn_ablation.common import load_config, register_custom_modules, resolve_from_root, variant_paths, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "cbam"), required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if str(device).startswith("0") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(model: YOLO, image: Path, config: dict) -> tuple[float, float]:
    kwargs = dict(imgsz=config["imgsz"], device=config["device"], verbose=False)
    for _ in range(config["benchmark_warmup"]):
        model.predict(str(image), **kwargs)
    synchronize(str(config["device"]))
    start = time.perf_counter()
    for _ in range(config["benchmark_iterations"]):
        model.predict(str(image), **kwargs)
    synchronize(str(config["device"]))
    elapsed = time.perf_counter() - start
    milliseconds = elapsed * 1000 / config["benchmark_iterations"]
    return milliseconds, 1000 / milliseconds


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_root(args.config))
    register_custom_modules()
    _, run_dir = variant_paths(args.variant, config)
    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Train {args.variant} first; missing {checkpoint}")

    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(resolve_from_root(config["data"])),
        split="test",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        conf=config["conf"],
        iou=config["iou"],
        project=str(run_dir),
        name="test",
        exist_ok=True,
        plots=True,
    )
    data_root = resolve_from_root(config["data"]).parent
    test_image = next((data_root / "images" / "test").glob("*"))
    latency_ms, fps = benchmark(model, test_image, config)
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    result = {
        "variant": args.variant,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "ultralytics_inference_ms": float(metrics.speed["inference"]),
        "ultralytics_inference_fps": 1000 / float(metrics.speed["inference"]),
        "inference_ms_batch1": latency_ms,
        "fps_batch1": fps,
        "parameters": parameters,
    }
    write_json(run_dir / "test_metrics.json", result)
    print(result)


if __name__ == "__main__":
    main()
