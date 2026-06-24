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

import matplotlib
import torch
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    collect_predictions,
    confusion_matrix_counts,
    metrics_from_predictions,
    pr_curve_data,
    resolve_device,
)


def plot_confusion_matrix(cm, class_names, output):
    labels = class_names + ["background"]
    normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Confusion Matrix (IoU 0.5, conf 0.25)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if normalized[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_pr_curves(pr_data, class_names, output):
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, (recalls, precisions, ap) in sorted(pr_data.items()):
        ax.plot(recalls, precisions, label=f"{class_names[cls]} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("PR Curve (IoU 0.5)"); ax.legend(loc="lower left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


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
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    class_names = [names[i] for i in range(len(names))]
    dataset = YoloDetectionDataset(data_yaml, args.split)
    loader = DataLoader(
        dataset, batch_size=int(config["batch"]), shuffle=False,
        num_workers=int(config.get("workers", 4)), collate_fn=collate_fn,
    )

    # Suy luận một lần, dùng lại cho metric + confusion matrix + PR curve.
    predictions, targets = collect_predictions(model, loader, device)
    metrics = metrics_from_predictions(predictions, targets)

    run_dir = experiment_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix_counts(predictions, targets, len(class_names))
    pr = pr_curve_data(predictions, targets, len(class_names))
    plot_confusion_matrix(cm, class_names, run_dir / f"{args.split}_confusion_matrix.png")
    plot_pr_curves(pr, class_names, run_dir / f"{args.split}_pr_curve.png")
    print(f"Saved confusion matrix + PR curve to {run_dir}")

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
