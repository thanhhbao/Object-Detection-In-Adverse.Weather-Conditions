#!/usr/bin/env python3
"""Train an Ultralytics model from a custom architecture YAML with partial pretrained loading.

Used for the +SE ablation: build YOLOv8n+SE from configs/yolov8n_se.yaml, transfer the
matching weights from the Phase-1 checkpoint (stage1_bdd30k_yolov8n), and fine-tune on XWOD.

Weight transfer pairs NON-attention layers of the pretrained model with the non-attention
layers of the target model in order, and skips the inserted attention block(s). This handles
the head index shift caused by inserting SE after SPPF — a plain name-based load would miss
the whole head.

Example:
  python scripts/train.py --data configs/xwod.yaml --cfg configs/yolov8n_se.yaml \\
      --weights /content/workspace/runs/stage1_bdd30k_yolov8n/weights/best.pt \\
      --name improved_yolov8n_se_xwod
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.common import register_custom_modules  # noqa: E402


def transfer_pretrained(target_dm, weights: str) -> dict:
    """Copy matching weights from `weights` into target DetectionModel, skipping attention.

    Returns a stats dict: loaded tensors, skipped attention layers, paired/total layers.
    """
    from ultralytics import YOLO

    from dawn_ablation.attention import CBAMResearch, SEResearch
    ATTN = (SEResearch, CBAMResearch)

    src_dm = YOLO(weights).model
    src_non_attn = [l for l in src_dm.model if not isinstance(l, ATTN)]
    tgt_layers = list(target_dm.model)

    new_state: dict = {}
    loaded = skipped = paired = 0
    si = 0
    for ti, tl in enumerate(tgt_layers):
        if isinstance(tl, ATTN):
            skipped += 1
            continue
        if si >= len(src_non_attn):
            break
        sl = src_non_attn[si]; si += 1; paired += 1
        sl_sd = sl.state_dict()
        for k, v in tl.state_dict().items():
            if k in sl_sd and sl_sd[k].shape == v.shape:
                new_state[f"model.{ti}.{k}"] = sl_sd[k]
                loaded += 1

    target_dm.load_state_dict(new_state, strict=False)
    return {
        "loaded_tensors": loaded,
        "skipped_attention_layers": skipped,
        "paired_layers": paired,
        "source_non_attention_layers": len(src_non_attn),
        "target_total_tensors": len(target_dm.state_dict()),
    }


def build_transferred_model(cfg: str, weights: str):
    """Build the model from cfg, transfer pretrained weights, return a trainable YOLO.

    Saves the transferred model to a temp checkpoint and reloads it so Ultralytics treats
    it as a normal pretrained model for training (robust across versions).
    """
    from ultralytics import YOLO

    model = YOLO(cfg)
    stats = transfer_pretrained(model.model, weights)
    print("Weight transfer:", stats)

    tmp = tempfile.mktemp(suffix="_se_init.pt")
    torch.save({"model": model.model, "train_args": {}, "epoch": -1}, tmp)
    return YOLO(tmp), stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train custom-arch model with partial pretrained loading")
    ap.add_argument("--data", required=True, help="Ultralytics data YAML (e.g. configs/xwod.yaml)")
    ap.add_argument("--cfg", required=True, help="Model architecture YAML (e.g. configs/yolov8n_se.yaml)")
    ap.add_argument("--weights", required=True, help="Pretrained checkpoint (stage1 best.pt)")
    ap.add_argument("--name", required=True, help="Run name")
    ap.add_argument("--project", default="/content/workspace/runs")
    # Hyperparameters mirror the baseline XWOD stage-2 run for a fair ablation.
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    register_custom_modules()

    model, _ = build_transferred_model(args.cfg, args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        cos_lr=True,
        deterministic=True,
        amp=True,
        pretrained=False,     # weights already transferred
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
