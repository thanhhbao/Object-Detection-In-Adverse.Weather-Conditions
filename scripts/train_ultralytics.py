#!/usr/bin/env python3
"""Train any Ultralytics detector from a YAML config.

File này có 3 phần cốt lõi:
1. Đọc config YAML: model, dataset path, project/name và hyperparameters.
2. Tạo YOLO model từ checkpoint hoặc model name, ví dụ `yolov8n.pt`.
3. Train hoặc resume, rồi lưu toàn bộ output vào thư mục project/name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO

from dawn_ablation.common import load_config, resolve_from_root

TRAIN_KEYS = {
    "epochs", "imgsz", "batch", "workers", "device", "optimizer", "lr0", "lrf",
    "weight_decay", "patience", "cos_lr", "deterministic", "amp", "seed", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "close_mosaic", "cache",
}


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC CONFIG
# Config chứa mọi thứ cần chỉnh sau này: model, data, epochs, batch, lr...
# Nhờ vậy trên Colab chỉ cần sửa YAML thay vì sửa code Python.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_config_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else resolve_from_root(path))


# ---------------------------------------------------------------------------
# PHẦN 2: TẠO MODEL
# `model` có thể là pretrained official như `yolov8n.pt`, `yolo11n.pt`,
# hoặc checkpoint tự train như `/content/.../yolov8n_bdd.pt`.
# ---------------------------------------------------------------------------


def build_model(config: dict, resume: bool) -> YOLO:
    if resume:
        checkpoint = Path(config["project"]) / config["name"] / "weights" / "last.pt"
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
    config = load_config(resolve_from_root(args.config))
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

