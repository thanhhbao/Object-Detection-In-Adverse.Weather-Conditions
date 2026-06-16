#!/usr/bin/env python3
"""Train a TorchVision Faster R-CNN detector from one YAML config.

File này song song với `scripts/train_ultralytics.py` nhưng cho nhánh 2-stage
(Faster R-CNN). Ba phần cốt lõi:
1. Đọc config trong `configs/torchvision/` và ghép với config common.
2. Tạo Faster R-CNN (COCO pretrained ở Stage 1, hoặc nạp checkpoint Stage 1 ở
   Stage 2) rồi thay detection head cho 6 class + background.
3. Train với SGD, chọn `best.pth` theo mAP50-95 trên split val, hỗ trợ resume.
"""

from __future__ import annotations

import argparse
import math
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
    build_fasterrcnn,
    collate_fn,
    evaluate_detector,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override seed for multi-seed runs.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHẦN 1 + 2: CONFIG VÀ MODEL
# Stage 1: config["model"] == "fasterrcnn_resnet50_fpn" -> dùng COCO weights.
# Stage 2: from_run resolve config["model"] thành .../weights/best.pth -> nạp lại.
# ---------------------------------------------------------------------------


def build_model_from_config(config: dict, device: torch.device) -> torch.nn.Module:
    num_classes = int(config["num_classes"])
    imgsz = int(config["imgsz"])
    model_value = str(config["model"])
    is_checkpoint = model_value.endswith((".pth", ".pt"))

    model = build_fasterrcnn(num_classes, imgsz, coco_pretrained=not is_checkpoint)
    if is_checkpoint:
        checkpoint = torch.load(model_value, map_location="cpu")
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        print(f"Loaded Stage 1 weights from {model_value}")
    return model.to(device)


def make_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_yaml = resolve_from_root(config["data"])
    batch = int(config["batch"])
    workers = int(config.get("workers", 4))

    train_set = YoloDetectionDataset(data_yaml, "train")
    val_set = YoloDetectionDataset(data_yaml, "val")
    print(f"train images: {len(train_set)} | val images: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=True, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch, shuffle=False, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# PHẦN 3: TRAIN
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, scaler, device, amp) -> float:
    model.train()
    running = 0.0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.item())
    return running / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.weights:
        config["model"] = args.weights
    if args.name:
        config["name"] = args.name
    if args.seed is not None:
        config["seed"] = args.seed

    device = resolve_device(config.get("device", "0"))
    torch.manual_seed(int(config.get("seed", 42)))
    print(f"device: {device}")

    run_dir = experiment_run_dir(config)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = experiment_checkpoint(config, "best.pth")
    last_path = experiment_checkpoint(config, "last.pth")

    train_loader, val_loader = make_loaders(config)
    model = build_model_from_config(config, device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(config["lr0"]),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.0005)),
    )

    epochs = int(config["epochs"])
    if config.get("cos_lr", True):
        eta_min = float(config["lr0"]) * float(config.get("lrf", 0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        scheduler = None

    amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    patience = int(config.get("patience", 10))

    start_epoch = 0
    best_map = -1.0
    epochs_without_improve = 0

    # Resume: nạp lại model/optimizer/scheduler từ last.pth.
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_map = checkpoint.get("best_map", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp)
        if scheduler:
            scheduler.step()

        metrics = evaluate_detector(model, val_loader, device)
        current_map = metrics["map50_95"]
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"epoch {epoch + 1}/{epochs} | loss {train_loss:.4f} | "
            f"mAP50 {metrics['map50']:.4f} | mAP50-95 {current_map:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_map": best_map,
            "config_name": config["name"],
        }
        torch.save(checkpoint, last_path)

        if current_map > best_map:
            best_map = current_map
            checkpoint["best_map"] = best_map
            torch.save(checkpoint, best_path)
            epochs_without_improve = 0
            print(f"  ↳ new best mAP50-95 {best_map:.4f} -> saved {best_path.name}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience}).")
                break

    write_json(run_dir / "train_history.json", {"history": history, "best_map50_95": best_map})
    print(f"Done. Best mAP50-95 = {best_map:.4f}. Weights in {weights_dir}")


if __name__ == "__main__":
    main()
