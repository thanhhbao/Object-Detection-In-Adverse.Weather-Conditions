#!/usr/bin/env python3
"""Train a custom YOLOv8 ablation model such as YOLOv8n + CBAM.

File này có 3 phần cốt lõi:
1. Đọc ablation config, rồi kế thừa dataset/hyperparameters từ Stage 2 baseline.
2. Tạo model kiến trúc mới và copy trọng số từ checkpoint BDD sang các layer khớp.
3. Train hoặc resume, lưu kết quả vào cùng thư mục `project/name`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO

from dawn_ablation.attention import CBAMResearch
from dawn_ablation.common import (
    experiment_checkpoint,
    experiment_run_dir,
    load_experiment_config,
    register_custom_modules,
    resolve_from_root,
    write_json,
)

TRAIN_KEYS = {
    "seed", "epochs", "imgsz", "batch", "workers", "device", "optimizer", "lr0",
    "lrf", "weight_decay", "patience", "cos_lr", "deterministic", "amp", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "close_mosaic", "cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override seed for multi-seed runs.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_ablation_config(args: argparse.Namespace) -> dict:
    path = args.config
    config = load_experiment_config(path)
    if config.get("variant") != "cbam":
        raise NotImplementedError(
            "scripts/train.py currently implements only the CBAM architecture ablation."
        )
    if args.weights:
        config["model"] = args.weights
    if args.name:
        config["name"] = args.name
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def copy_pretrained_by_layer(target: YOLO, source: YOLO) -> tuple[int, int]:
    """Copy compatible source layers and leave inserted CBAM layers randomly initialized."""
    source_layers = list(source.model.model)
    copied_tensors = 0
    source_index = 0

    for target_layer in target.model.model:
        if isinstance(target_layer, CBAMResearch):
            continue

        source_layer = source_layers[source_index]
        source_index += 1

        if target_layer.__class__.__name__ != source_layer.__class__.__name__:
            raise RuntimeError(
                f"Pretrained layer mismatch: target={target_layer.__class__.__name__}, "
                f"source={source_layer.__class__.__name__}"
            )

        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target_layer.load_state_dict(compatible, strict=False)
        copied_tensors += len(compatible)

    if source_index != len(source_layers):
        raise RuntimeError(f"Only consumed {source_index}/{len(source_layers)} source layers")
    return copied_tensors, source_index


def main() -> None:
    args = parse_args()
    register_custom_modules()
    config = load_ablation_config(args)
    run_dir = experiment_run_dir(config)

    if args.resume:
        checkpoint = experiment_checkpoint(config, "last.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return

    model = YOLO(str(resolve_from_root(config["model_yaml"])))
    source = YOLO(str(config["model"]))
    copied, layers = copy_pretrained_by_layer(model, source)
    print(f"Transferred {copied} tensors across {layers} non-CBAM layers.")

    write_json(
        run_dir / "pretrained_transfer.json",
        {
            "variant": config["variant"],
            "source": str(config["model"]),
            "model_yaml": str(resolve_from_root(config["model_yaml"])),
            "copied_tensors": copied,
            "non_cbam_layers": layers,
        },
    )

    train_args = {key: config[key] for key in TRAIN_KEYS if key in config}
    train_args.update(
        data=str(resolve_from_root(config["data"])),
        project=str(resolve_from_root(config["project"])),
        name=config["name"],
        exist_ok=config.get("exist_ok", True),
        pretrained=True,
        verbose=config.get("verbose", True),
    )
    model.train(**train_args)


if __name__ == "__main__":
    main()
