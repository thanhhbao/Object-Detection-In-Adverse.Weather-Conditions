#!/usr/bin/env python3
"""Evaluate one trained Faster R-CNN run and write metrics in the shared format.

Song song với `scripts/evaluate.py` (dành cho YOLO) nhưng cho nhánh TorchVision.
Xuất ra cùng các khóa metric (`precision`, `recall`, `map50`, `map50_95`,
`inference_ms_batch1`, `fps_batch1`, `parameters`) để `collect_results.py` và
`compare_results.py` gom chung một bảng với các model YOLO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
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
    benchmark_batch1,
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
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        help="Override the run name (config name), e.g. a per-seed run folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.run_name:
        config["name"] = args.run_name
    device = resolve_device(config.get("device", "0"))

    checkpoint_path = Path(args.weights) if args.weights else experiment_checkpoint(config, "best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_fasterrcnn(int(config["num_classes"]), int(config["imgsz"]), coco_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device)

    data_yaml = resolve_from_root(config["data"])
    dataset = YoloDetectionDataset(data_yaml, args.split)
    loader = DataLoader(
        dataset, batch_size=int(config["batch"]), shuffle=False,
        num_workers=int(config.get("workers", 4)), collate_fn=collate_fn,
    )

    metrics = evaluate_detector(model, loader, device)
    sample_image, _ = dataset[0]
    latency_ms, fps = benchmark_batch1(
        model, sample_image, device,
        warmup=int(config.get("benchmark_warmup", 10)),
        iterations=int(config.get("benchmark_iterations", 50)),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())

    result = {
        "run": config["name"],
        "config": args.config,
        "weights": str(checkpoint_path),
        "split": args.split,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "ultralytics_inference_ms": None,
        "ultralytics_inference_fps": None,
        "inference_ms_batch1": latency_ms,
        "fps_batch1": fps,
        "parameters": int(parameters),
        "per_class_map50_95": metrics["per_class_map50_95"],
    }

    output = experiment_run_dir(config) / f"{args.split}_metrics.json"
    write_json(output, result)
    print(result)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
