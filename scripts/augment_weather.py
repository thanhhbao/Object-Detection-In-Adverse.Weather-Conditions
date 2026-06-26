#!/usr/bin/env python3
"""Tăng cường dữ liệu thời tiết (offline) cho một dataset YOLO.

Sinh thêm các phiên bản sương mù / mưa / tuyết / tối-sáng cho ảnh TRAIN bằng
Albumentations, giúp khắc phục dữ liệu thời tiết xấu ít. Các phép biến đổi đều là
quang học (không đổi hình học) nên hộp bao GIỮ NGUYÊN — chỉ cần chép lại nhãn.

Kết quả: dataset mới = train (gốc + tăng cường) + val/test (giữ nguyên), kèm
dataset.yaml để train như bình thường. Đây là hiện thực của ablation "weather aug".

Ví dụ:
  python scripts/augment_weather.py --src /content/dawn_6cls_yolo \\
      --dst /content/dawn_6cls_yolo_waug --copies 2 --seed 42
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import albumentations as A
import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Thư mục dataset YOLO gốc")
    parser.add_argument("--dst", type=Path, required=True, help="Thư mục dataset đầu ra")
    parser.add_argument("--copies", type=int, default=2, help="Số ảnh tăng cường mỗi ảnh train")
    parser.add_argument("--split", default="train", help="Split để tăng cường (mặc định train)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_transform(seed: int) -> A.Compose:
    """Mỗi lần áp dụng chọn ngẫu nhiên một kiểu thời tiết xấu, kèm nhiễu nhẹ."""
    # Dùng tham số mặc định để tương thích mọi phiên bản Albumentations
    # (tên tham số chi tiết thay đổi giữa các bản; mặc định luôn hợp lệ).
    return A.Compose([
        A.OneOf([
            A.RandomFog(p=1.0),
            A.RandomRain(p=1.0),
            A.RandomSnow(p=1.0),
            A.RandomShadow(p=1.0),
        ], p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussNoise()], p=0.3),
    ])


def label_dir_of(images_split_dir: Path) -> Path:
    parts = list(images_split_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"; break
    return Path(*parts)


def copy_tree(src_dir: Path, dst_dir: Path) -> None:
    if src_dir.exists():
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    import random
    random.seed(args.seed)

    # Sao chép nguyên dataset gốc (giữ val/test + train gốc).
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        copy_tree(src / sub, dst / sub)

    src_images = src / "images" / args.split
    dst_images = dst / "images" / args.split
    dst_labels = dst / "labels" / args.split
    src_labels = label_dir_of(src_images)

    images = sorted(p for p in src_images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"Không thấy ảnh trong {src_images}")

    transform = build_transform(args.seed)
    created = 0
    for image_path in images:
        label_path = src_labels / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for k in range(args.copies):
            augmented = transform(image=image)["image"]
            stem = f"{image_path.stem}_waug{k}"
            cv2.imwrite(str(dst_images / f"{stem}{image_path.suffix}"), augmented,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Nhãn giữ nguyên (biến đổi quang học không làm dời hộp bao).
            if label_path.exists():
                shutil.copy2(label_path, dst_labels / f"{stem}.txt")
            else:
                (dst_labels / f"{stem}.txt").write_text("", encoding="utf-8")
            created += 1

    # dataset.yaml: cập nhật path, giữ names/splits từ gốc.
    data = yaml.safe_load((src / "dataset.yaml").read_text(encoding="utf-8"))
    data["path"] = str(dst)
    (dst / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    total = len(list((dst_images).rglob("*")))
    print(f"Đã tạo {created} ảnh tăng cường. Train tổng cộng ~{total} ảnh.")
    print(f"Dataset mới: {dst}/dataset.yaml")


if __name__ == "__main__":
    main()
