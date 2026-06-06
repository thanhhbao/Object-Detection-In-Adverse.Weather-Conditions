#!/usr/bin/env python3
"""Train YOLOv8 baseline hoặc YOLOv8-CBAM cho ablation study.

File này có 3 phần cốt lõi:
1. Đọc cấu hình: chọn train `baseline` hay `cbam`, đọc file config YAML.
2. Nạp pretrained: copy trọng số từ `yolov8n.pt` sang model đang train.
3. Huấn luyện/resume: chạy train mới hoặc tiếp tục từ `last.pt` trên Colab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ROOT là thư mục gốc project. Khi chạy từ Colab hoặc terminal, dòng này giúp
# Python import được package local trong thư mục src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO

from dawn_ablation.attention import CBAMResearch
from dawn_ablation.common import (
    load_config,
    register_custom_modules,
    resolve_from_root,
    variant_paths,
    write_json,
)

# Chỉ các key trong danh sách này mới được truyền vào YOLO.train().
# Làm vậy để config có thể chứa thêm thông tin khác như `data`, `project`,
# `pretrained` mà không gây lỗi tham số lạ cho Ultralytics.
TRAIN_KEYS = {
    "seed", "epochs", "imgsz", "batch", "workers", "device", "optimizer", "lr0",
    "lrf", "weight_decay", "patience", "cos_lr", "deterministic", "amp", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "close_mosaic",
}


# ---------------------------------------------------------------------------
# PHẦN 1: ĐỌC CẤU HÌNH THÍ NGHIỆM
# Người dùng chọn variant bằng --variant:
#   baseline -> YOLOv8n gốc
#   cbam     -> YOLOv8n có thêm CBAM ở Neck
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "cbam"), required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="Resume this variant from last.pt."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PHẦN 2: NẠP PRETRAINED CÔNG BẰNG
# CBAM có thêm vài layer mới so với YOLOv8n gốc, nên không thể load checkpoint
# trực tiếp theo kiểu thông thường. Hàm dưới copy từng layer giống nhau, còn
# layer CBAM thì bỏ qua để nó tự học từ đầu.
# ---------------------------------------------------------------------------


def copy_pretrained_by_layer(target: YOLO, source: YOLO) -> tuple[int, int]:
    """Copy matching source layers, skipping inserted CBAM modules in target."""
    source_layers = list(source.model.model)
    copied_tensors = 0
    source_index = 0

    for target_layer in target.model.model:
        # CBAM là layer mới thêm vào model cải tiến, checkpoint YOLOv8n gốc
        # không có trọng số tương ứng nên không copy.
        if isinstance(target_layer, CBAMResearch):
            continue

        source_layer = source_layers[source_index]
        source_index += 1

        # Nếu thứ tự layer bị lệch, dừng ngay để tránh copy sai trọng số.
        if target_layer.__class__.__name__ != source_layer.__class__.__name__:
            raise RuntimeError(
                f"Pretrained layer mismatch: target={target_layer.__class__.__name__}, "
                f"source={source_layer.__class__.__name__}"
            )

        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()

        # Chỉ copy tensor có cùng tên và cùng shape. Các tensor không khớp sẽ
        # được giữ nguyên theo khởi tạo ban đầu của model đích.
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target_layer.load_state_dict(compatible, strict=False)
        copied_tensors += len(compatible)

    if source_index != len(source_layers):
        raise RuntimeError(f"Only consumed {source_index}/{len(source_layers)} source layers")
    return copied_tensors, source_index


# ---------------------------------------------------------------------------
# PHẦN 3: HUẤN LUYỆN HOẶC RESUME
# Nếu train mới: tạo model, copy pretrained, rồi gọi YOLO.train().
# Nếu Colab bị ngắt: chạy lại với --resume để tiếp tục từ last.pt.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_root(args.config))
    register_custom_modules()
    model_yaml, run_dir = variant_paths(args.variant, config)
    pretrained = config["pretrained"]

    # Resume không cần tạo model YAML hay copy pretrained nữa. Ultralytics sẽ đọc
    # lại model, optimizer state và epoch từ checkpoint last.pt.
    if args.resume:
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return

    # Train mới: tạo đúng kiến trúc theo variant, rồi lấy yolov8n.pt làm nguồn
    # pretrained để transfer sang model vừa tạo.
    model = YOLO(str(model_yaml))
    source = YOLO(pretrained)
    copied, layers = copy_pretrained_by_layer(model, source)
    print(f"Transferred {copied} tensors across {layers} non-CBAM layers.")
    write_json(
        run_dir / "pretrained_transfer.json",
        {"variant": args.variant, "copied_tensors": copied, "non_cbam_layers": layers},
    )

    # Ultralytics dựng lại model trong trainer; đặt ckpt truthy để trainer nhận
    # model đã được copy trọng số ở trên làm nguồn weights.
    model.ckpt = {"source": pretrained, "custom_transfer": True}
    model.ckpt_path = str(pretrained)

    # Dùng cùng một file config cho baseline và CBAM để so sánh công bằng.
    train_args = {key: config[key] for key in TRAIN_KEYS if key in config}
    train_args.update(
        data=str(resolve_from_root(config["data"])),
        project=str(resolve_from_root(config["project"])),
        name=args.variant,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    model.train(**train_args)


if __name__ == "__main__":
    main()
