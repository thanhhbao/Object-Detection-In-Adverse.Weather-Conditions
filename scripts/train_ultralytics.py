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

import torch
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
    parser.add_argument("--weights", default=None,
                        help="Override config['model']: a .pt checkpoint loads the full model; "
                             "a .yaml creates a fresh architecture. "
                             "For P2/custom arch experiments use --init-weights instead.")
    parser.add_argument("--init-weights", default=None,
                        help="Partial weight transfer: build model from config['model'] (YAML), "
                             "then copy matching layers from this checkpoint. "
                             "Non-matching layers (new arch branches) stay randomly initialised. "
                             "Transfers layers with identical name AND shape; logs all decisions.")
    parser.add_argument("--max-transfer-layer", type=int, default=None,
                        help="Hard cap on layer index to transfer (inclusive). "
                             "Default None = transfer all matching layers. "
                             "Set to 15 for P2/custom-head models to prevent accidental "
                             "transfer of re-indexed PAN layers.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--data", default=None,
                        help="Override the dataset YAML path (e.g. Phase 2 merged dataset).")
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
    if args.data:
        config["data"] = resolve_config_path(args.data)
    if args.seed is not None:
        config["seed"] = args.seed
    return config


# ---------------------------------------------------------------------------
# Partial weight transfer for custom architectures (E2: P2 head, etc.)
# ---------------------------------------------------------------------------


def _load_init_weights(model: YOLO, ckpt_path: Path, max_layer: int | None = None) -> None:
    """Copy matching layers from ckpt_path into model; enforce layer allowlist; log every decision.

    Matching criterion:
      1. Key exists in source checkpoint.
      2. Layer index (from "model.<idx>.*") is within the allowlist (0..max_layer).
      3. Tensor shapes are identical.

    Using an explicit allowlist (not just shape matching) prevents accidental transfer
    of re-indexed PAN layers that happen to share the same shape as new P2 layers.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Ultralytics saves the full model object under "model" key
    src_obj = ckpt.get("model", ckpt)
    if hasattr(src_obj, "state_dict"):
        src_sd: dict = src_obj.float().state_dict()
    elif isinstance(src_obj, dict):
        src_sd = src_obj
    else:
        raise ValueError(f"Cannot read state dict from {ckpt_path}")

    dst_sd = model.model.state_dict()
    allowed = set(range(max_layer + 1)) if max_layer is not None else None

    transferred: dict = {}
    skip_missing: list[str] = []
    skip_layer: list[str] = []   # outside allowlist
    skip_shape: list[str] = []

    for key, dst_tensor in dst_sd.items():
        if key not in src_sd:
            skip_missing.append(key)
            continue

        # Enforce layer allowlist before checking shape
        parts = key.split(".")
        if allowed is not None and len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            if int(parts[1]) not in allowed:
                skip_layer.append(key)
                continue

        if src_sd[key].shape != dst_tensor.shape:
            skip_shape.append(
                f"{key}  src={tuple(src_sd[key].shape)} dst={tuple(dst_tensor.shape)}"
            )
            continue

        transferred[key] = src_sd[key]

    dst_sd.update(transferred)
    model.model.load_state_dict(dst_sd, strict=True)

    # Report which layer indices actually received weights (Upsample/Concat have none)
    transferred_layers: set[int] = set()
    for k in transferred:
        parts = k.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            transferred_layers.add(int(parts[1]))

    # Hard assert: no layer outside allowlist was transferred
    if allowed is not None:
        leaked = transferred_layers - allowed
        assert not leaked, f"BUG: layers {sorted(leaked)} transferred despite allowlist {max_layer}"

    print(f"\n[init_weights] Source checkpoint     : {ckpt_path}")
    print(f"[init_weights] Layer allowlist (max) : {max_layer if max_layer is not None else 'all'}")
    print(f"[init_weights] Tensors transferred   : {len(transferred)}/{len(dst_sd)}")
    print(f"[init_weights] Skipped – outside cap : {len(skip_layer)}")
    print(f"[init_weights] Skipped – shape       : {len(skip_shape)}")
    print(f"[init_weights] Skipped – missing     : {len(skip_missing)}")
    # Upsample/Concat have no params → appear in allowlist but not in transferred_layers; that's correct
    print(f"[init_weights] Layer indices w/ params: {sorted(transferred_layers)}")
    if skip_shape:
        print("[init_weights] Shape mismatches (first 5):")
        for line in skip_shape[:5]:
            print(f"               {line}")


# ---------------------------------------------------------------------------
# PHẦN 2: TẠO MODEL
# Config này chỉ dành cho Ultralytics models như YOLO/RT-DETR. Faster R-CNN dùng
# `scripts/train_torchvision.py` sau này.
# ---------------------------------------------------------------------------


def build_model(
    config: dict,
    resume: bool,
    init_weights: str | None = None,
    max_transfer_layer: int | None = None,
) -> YOLO:
    if resume:
        checkpoint = experiment_checkpoint(config, "last.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint}")
        return YOLO(str(checkpoint))
    model = YOLO(str(config["model"]))
    if init_weights:
        _load_init_weights(model, Path(init_weights), max_layer=max_transfer_layer)
    return model


# ---------------------------------------------------------------------------
# PHẦN 3: TRAIN HOẶC RESUME
# Train mới dùng toàn bộ hyperparameters trong config. Resume thì Ultralytics tự
# đọc lại epoch, optimizer và model state từ `last.pt`.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_train_config(args)
    model = build_model(
        config,
        args.resume,
        init_weights=args.init_weights,
        max_transfer_layer=args.max_transfer_layer,
    )

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
