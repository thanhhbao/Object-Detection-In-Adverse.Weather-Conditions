#!/usr/bin/env python3
"""Evaluate an ablation checkpoint (baseline or +SE) on a dataset split.

Registers the custom attention modules first, so +SE checkpoints load correctly.
Prints the standard Ultralytics metrics (P, R, mAP50, mAP50-95) per class.

Examples:
  # +SE on DAWN (zero-shot generalization)
  python scripts/eval_ablation.py \\
    --weights /workspace/runs/improved_yolov8n_se_xwod_v3/weights/best.pt \\
    --data /workspace/datasets/dawn_6cls_yolo/dataset.yaml --split val \\
    --name eval_dawn_se

  # baseline on XWOD test
  python scripts/eval_ablation.py \\
    --weights /workspace/runs/baseline_via_trainpy_v2/weights/best.pt \\
    --data configs/xwod.yaml --split test --name eval_xwod_test_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.common import register_custom_modules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="/workspace/runs/evals")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name, exist_ok=True,
    )
    box = metrics.box
    print(f"\n=== {args.name} ({args.split}) ===")
    print(f"mAP50:    {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    print(f"Precision:{box.mp:.4f}")
    print(f"Recall:   {box.mr:.4f}")


if __name__ == "__main__":
    main()
