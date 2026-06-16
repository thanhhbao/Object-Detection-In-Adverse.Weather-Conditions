#!/usr/bin/env python3
"""Train an Ultralytics detector from one YAML config.

File này có 3 phần cốt lõi:
1. Đọc config trong `configs/ultralytics/` và ghép với config common.
2. Tạo YOLO model từ official pretrained hoặc checkpoint stage trước.
3. Train hoặc resume, rồi lưu toàn bộ output vào thư mục project/name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO

from dawn_ablation.common import (
    experiment_checkpoint,
    load_experiment_config,
    resolve_from_root,
)

TRAIN_KEYS = {
    "epochs", "imgsz", "batch", "workers", "device", "optimizer", "lr0", "lrf",
    "weight_decay", "patience", "cos_lr", "deterministic", "amp", "seed", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "close_mosaic", "cache",
}


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC CONFIG
# Mỗi file YAML trong configs/ultralytics/ là một thí nghiệm đầy đủ: model,
# dataset, project/name và hyperparameters. Script không hard-code đường dẫn.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override seed for multi-seed runs.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_config_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else resolve_from_root(path))


def load_train_config(args: argparse.Namespace) -> dict:
    config = load_experiment_config(args.config)
    if args.weights:
        config["model"] = args.weights
    if args.name:
        config["name"] = args.name
    if args.seed is not None:
        config["seed"] = args.seed
    return config


# ---------------------------------------------------------------------------
# PHẦN 2: TẠO MODEL
# Config này chỉ dành cho Ultralytics models như YOLO/RT-DETR. Faster R-CNN dùng
# `scripts/train_torchvision.py` sau này.
# ---------------------------------------------------------------------------


def build_model(config: dict, resume: bool) -> YOLO:
    if resume:
        checkpoint = experiment_checkpoint(config, "last.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint}")
        return YOLO(str(checkpoint))
    return YOLO(str(config["model"]))


# ---------------------------------------------------------------------------
# PHẦN 3: TRAIN HOẶC RESUME
# Train mới dùng toàn bộ hyperparameters trong config. Resume thì Ultralytics tự
# đọc lại epoch, optimizer và model state từ `last.pt`.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_train_config(args)
    model = build_model(config, args.resume)

    if args.resume:
        model.train(resume=True)
        return

    train_args = {key: config[key] for key in TRAIN_KEYS if key in config}
    train_args.update(
        data=resolve_config_path(config["data"]),
        project=resolve_config_path(config["project"]),
        name=config["name"],
        exist_ok=config.get("exist_ok", True),
        pretrained=config.get("pretrained", True),
        verbose=config.get("verbose", True),
    )
    model.train(**train_args)


if __name__ == "__main__":
    main()
