#!/usr/bin/env python3
"""Prepare ACDC dataset for YOLO training.

ACDC cung cấp panoptic segmentation theo COCO panoptic format (JSON + PNG mask).
Script này convert mask → tight bbox → YOLO format, áp dụng class mapping sau:

  person     → person (0)
  rider      → person (0)   # consistent với BDD100K mapping
  car        → car    (2)
  truck      → truck  (5)
  bus        → bus    (4)
  motorcycle → motorcycle (3)
  bicycle    → bicycle (1)
  train / traffic light / sign / stuff → ignore

ACDC structure (mặc định):
  <raw-dir>/
    rgb_anon/
      fog/train/<city>/<img>.png
      night/train/<city>/<img>.png
      rain/train/<city>/<img>.png
      snow/train/<city>/<img>.png
    gt/
      panoptic/
        fog/train/panoptic_fog_train.json
              └── <city>/<img>_gt_panoptic.png
        night/train/panoptic_night_train.json
        rain/train/panoptic_rain_train.json
        snow/train/panoptic_snow_train.json
      (val/test cùng cấu trúc)

Ví dụ:
  python scripts/prepare_acdc.py \\
    --raw-dir /content/acdc \\
    --output-dir /content/acdc_6cls_yolo \\
    --imgsz 640 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import (
    Box,
    PreparedSample,
    TARGET_CLASSES,
    clean_output_dir,
    clamp_box,
    export_yolo_dataset,
    make_yolo_dirs,
    stratified_split,
)

# ---------------------------------------------------------------------------
# Cityscapes category_id → class name (chỉ các class cần thiết)
# ---------------------------------------------------------------------------

# Cityscapes trainId mapping (dùng trong ACDC panoptic JSON)
# category_id trong JSON có thể là trainId hoặc id tùy version ACDC
# Script hỗ trợ cả hai qua dict này (key = category_id từ JSON)
CITYSCAPES_ID_TO_NAME: dict[int, str | None] = {
    # things — các class quan tâm
    24: "person",
    25: "rider",     # → person qua CLASS_ALIASES
    26: "car",
    27: "truck",
    28: "bus",
    31: "train",     # → None (ignore)
    32: "motorcycle",
    33: "bicycle",
    # stuff và classes khác → None (ignore)
}

WEATHER_CONDITIONS = ["fog", "night", "rain", "snow"]


# ---------------------------------------------------------------------------
# Panoptic mask decoder
# ---------------------------------------------------------------------------


def decode_panoptic_mask(mask_path: Path) -> np.ndarray:
    """Đọc COCO panoptic PNG và trả về array segment_id per pixel (H, W)."""
    mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.int32)
    # COCO panoptic encoding: segment_id = R + G*256 + B*65536
    return mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 65536


def segment_to_bbox(seg_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tính tight bbox (x1, y1, x2, y2) từ boolean mask."""
    rows = np.any(seg_mask, axis=1)
    cols = np.any(seg_mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x1, y1, x2 + 1, y2 + 1  # x2/y2 exclusive


# ---------------------------------------------------------------------------
# Đọc một split của một điều kiện thời tiết
# ---------------------------------------------------------------------------


def load_acdc_split(
    raw_dir: Path,
    condition: str,
    split: str,
) -> list[PreparedSample]:
    """Load tất cả ảnh từ một condition/split, convert panoptic → bbox."""
    panoptic_json = (
        raw_dir / "gt" / "panoptic" / condition / split
        / f"panoptic_{condition}_{split}.json"
    )
    if not panoptic_json.exists():
        print(f"  [SKIP] không tìm thấy JSON: {panoptic_json}")
        return []

    data = json.loads(panoptic_json.read_text(encoding="utf-8"))

    # Xây category_id lookup từ JSON (nếu có categories block)
    cat_lookup: dict[int, str | None] = dict(CITYSCAPES_ID_TO_NAME)
    if "categories" in data:
        for cat in data["categories"]:
            cid = cat["id"]
            name = cat.get("name", "").lower().strip()
            if cid not in cat_lookup:
                cat_lookup[cid] = name if name in {c for c in TARGET_CLASSES} | {"rider", "train"} else None

    mask_root = raw_dir / "gt" / "panoptic" / condition / split
    img_root = raw_dir / "rgb_anon" / condition / split

    samples: list[PreparedSample] = []
    skipped = {"no_image": 0, "no_mask": 0, "no_boxes": 0}

    for ann in data.get("annotations", []):
        # Tìm file ảnh
        file_name = ann.get("file_name", "")  # e.g. "GOPR0351_frame_000001_rgb_anon.png"
        img_stem = Path(file_name).stem.replace("_rgb_anon", "")
        # Tìm ảnh trong cây thư mục (có thể có subfolder city)
        img_candidates = list(img_root.rglob(f"*{img_stem}*.png"))
        if not img_candidates:
            skipped["no_image"] += 1
            continue
        image_path = img_candidates[0]

        # Tìm mask PNG (tên tương ứng với panoptic_file_name trong ann)
        mask_file = ann.get("panoptic_file_name", "")
        if not mask_file:
            # fallback: tên ảnh nhưng đổi suffix
            mask_file = Path(file_name).stem + "_gt_panoptic.png"
        mask_candidates = list(mask_root.rglob(f"*{Path(mask_file).stem}*"))
        if not mask_candidates:
            skipped["no_mask"] += 1
            continue
        mask_path = mask_candidates[0]

        try:
            seg_map = decode_panoptic_mask(mask_path)
            img_h, img_w = seg_map.shape
        except Exception:
            skipped["no_mask"] += 1
            continue

        boxes: list[Box] = []
        for seg_info in ann.get("segments_info", []):
            cat_id = seg_info.get("category_id")
            seg_id = seg_info.get("id")
            class_name = cat_lookup.get(cat_id)
            if class_name is None:
                continue

            pixel_mask = seg_map == seg_id
            bbox = segment_to_bbox(pixel_mask)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            box = clamp_box(class_name, x1, y1, x2, y2, img_w, img_h)
            if box is not None:
                boxes.append(box)

        if not boxes:
            skipped["no_boxes"] += 1
            continue

        samples.append(
            PreparedSample(
                image=image_path,
                boxes=tuple(boxes),
                split_key=f"{condition}_{split}",
                source="acdc",
                weather=condition,
            )
        )

    if any(skipped.values()):
        print(f"  [{condition}/{split}] skipped: {skipped}")
    return samples


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Root ACDC directory (chứa rgb_anon/ và gt/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions", type=str, default="all",
        help="Comma-separated: fog,rain,snow,night hoặc 'all'",
    )
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated list of ACDC splits to load (default: train,val,test)",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    conditions = WEATHER_CONDITIONS if args.conditions == "all" else [
        c.strip() for c in args.conditions.split(",")
    ]
    acdc_splits = [s.strip() for s in args.splits.split(",")]

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=("train", "val", "test"))

    all_samples: list[PreparedSample] = []
    for condition in conditions:
        for acdc_split in acdc_splits:
            print(f"Loading {condition}/{acdc_split}...")
            samples = load_acdc_split(raw_dir, condition, acdc_split)
            print(f"  → {len(samples)} ảnh hợp lệ")
            all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("Không tìm thấy ảnh nào. Kiểm tra --raw-dir và cấu trúc thư mục ACDC.")

    print(f"\nTổng: {len(all_samples)} ảnh. Đang chia train/val/test (stratified theo weather)...")
    assignments = stratified_split(
        all_samples,
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        stratify_attr="weather",
    )

    export_yolo_dataset(
        samples=all_samples,
        assignments=assignments,
        raw_root=raw_dir,
        output_dir=output_dir,
        imgsz=args.imgsz,
        splits=("train", "val", "test"),
    )


if __name__ == "__main__":
    main()
