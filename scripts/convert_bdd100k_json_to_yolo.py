#!/usr/bin/env python3
"""Convert official BDD100K JSON labels -> 6-class YOLO format.

The Kaggle mirror used by prepare_bdd100k_yolo.py ships pre-made 10-class YOLO
label files. The official BDD100K release does not: it ships raw images plus a
single JSON per split. This script bridges that gap so the full 70K train split
can be turned into a retrieval pool.

Source layout (official BDD100K release):
  <images-root>/images/100k/{train,val}/*.jpg
  <labels-root>/labels/bdd100k_labels_images_{train,val}.json

Output layout (project 6-class YOLO):
  <dst>/images/<split>/*.jpg    (symlink or copy)
  <dst>/labels/<split>/*.txt
  <dst>/dataset.yaml

Usage:
  python3 scripts/convert_bdd100k_json_to_yolo.py \
    --images-root /workspace/bdd100k_raw/bdd100k/bdd100k \
    --labels-root /workspace/bdd100k_raw/bdd100k_labels_release/bdd100k \
    --dst /workspace/datasets_noleak/bdd100k_6cls_full_yolo \
    --splits train --mode symlink --clean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Class mapping: BDD100K JSON category -> project 6-class
# ---------------------------------------------------------------------------
# Target: person(0) bicycle(1) car(2) motorcycle(3) bus(4) truck(5)
# Mirrors REMAP in prepare_bdd100k_yolo.py, which folds "rider" into person.
CATEGORY_MAP: dict[str, int] = {
    "person": 0,
    "rider": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "bus": 4,
    "truck": 5,
}
# Skipped: traffic light, traffic sign, train, drivable area, lane

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# BDD100K is uniformly 1280x720; verified against a sample before conversion.
DEFAULT_W, DEFAULT_H = 1280, 720


def verify_image_size(img_dir: Path, expect_w: int, expect_h: int, n: int = 5) -> None:
    """Fail loudly if the sampled images are not the assumed resolution."""
    try:
        from PIL import Image
    except ImportError:
        print("  WARNING: Pillow unavailable, skipping resolution check.")
        return

    samples = []
    for p in sorted(img_dir.rglob("*.jpg")):
        samples.append(p)
        if len(samples) >= n:
            break
    if not samples:
        raise RuntimeError(f"No .jpg found under {img_dir}")

    for p in samples:
        with Image.open(p) as im:
            if im.size != (expect_w, expect_h):
                raise RuntimeError(
                    f"Unexpected image size {im.size} for {p.name}; "
                    f"expected ({expect_w}, {expect_h}). "
                    f"Re-run with --img-width/--img-height."
                )
    print(f"  Resolution check: {len(samples)} samples all {expect_w}x{expect_h}")


def convert_split(
    split: str,
    images_root: Path,
    labels_root: Path,
    dst: Path,
    mode: str,
    img_w: int,
    img_h: int,
    drop_empty: bool,
) -> dict:
    src_img_dir = images_root / "images" / "100k" / split
    json_path = labels_root / "labels" / f"bdd100k_labels_images_{split}.json"

    if not src_img_dir.is_dir():
        raise RuntimeError(f"Image dir not found: {src_img_dir}")
    if not json_path.is_file():
        raise RuntimeError(f"Label JSON not found: {json_path}")

    print(f"\n[{split}] images: {src_img_dir}")
    print(f"[{split}] labels: {json_path}")
    verify_image_size(src_img_dir, img_w, img_h)

    # Some BDD100K releases nest images in subdirectories under the split
    # dir, so index every .jpg by basename rather than assuming a flat layout.
    print("  Indexing images (recursive)...")
    index: dict[str, Path] = {}
    dupes = 0
    for p in src_img_dir.rglob("*.jpg"):
        if p.name in index:
            dupes += 1
            continue
        index[p.name] = p
    print(f"  Indexed {len(index)} images" + (f" ({dupes} duplicate names ignored)" if dupes else ""))

    print(f"  Loading JSON ({json_path.stat().st_size / 1e6:.0f} MB)...")
    records = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"  Records: {len(records)}")

    dst_img_dir = dst / "images" / split
    dst_lbl_dir = dst / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "records": len(records),
        "images_indexed": len(index),
        "written": 0,
        "missing_image": 0,
        "empty_dropped": 0,
        "boxes": 0,
        "boxes_skipped_category": 0,
        "boxes_skipped_degenerate": 0,
        "per_class": {name: 0 for name in CLASS_NAMES},
    }

    for rec in records:
        name = rec.get("name")
        if not name:
            continue
        src_img = index.get(name)
        if src_img is None:
            stats["missing_image"] += 1
            continue

        lines: list[str] = []
        for lab in rec.get("labels") or []:
            box = lab.get("box2d")
            if not box:
                continue  # drivable area / lane use poly2d
            cat = lab.get("category")
            cls = CATEGORY_MAP.get(cat)
            if cls is None:
                stats["boxes_skipped_category"] += 1
                continue

            x1 = max(0.0, min(float(box["x1"]), img_w))
            x2 = max(0.0, min(float(box["x2"]), img_w))
            y1 = max(0.0, min(float(box["y1"]), img_h))
            y2 = max(0.0, min(float(box["y2"]), img_h))
            if x2 <= x1 or y2 <= y1:
                stats["boxes_skipped_degenerate"] += 1
                continue

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["boxes"] += 1
            stats["per_class"][CLASS_NAMES[cls]] += 1

        if not lines and drop_empty:
            stats["empty_dropped"] += 1
            continue

        stem = Path(name).stem
        (dst_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        dst_img = dst_img_dir / name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if mode == "copy":
            shutil.copy2(src_img, dst_img)
        else:
            os.symlink(src_img.resolve(), dst_img)
        stats["written"] += 1

    print(f"  Written: {stats['written']}  boxes: {stats['boxes']}")
    if stats["missing_image"]:
        print(f"  Missing images: {stats['missing_image']}")
    if stats["empty_dropped"]:
        print(f"  Dropped (no 6-class object): {stats['empty_dropped']}")
    print("  Per-class boxes: " + "  ".join(
        f"{k}={v}" for k, v in stats["per_class"].items()))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert official BDD100K JSON labels to 6-class YOLO")
    ap.add_argument("--images-root", type=Path, required=True,
                    help="Dir containing images/100k/<split>/")
    ap.add_argument("--labels-root", type=Path, required=True,
                    help="Dir containing labels/bdd100k_labels_images_<split>.json")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "val"])
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--img-width", type=int, default=DEFAULT_W)
    ap.add_argument("--img-height", type=int, default=DEFAULT_H)
    ap.add_argument("--drop-empty", action="store_true",
                    help="Skip images with no 6-class object")
    ap.add_argument("--clean", action="store_true",
                    help="Remove --dst before writing")
    args = ap.parse_args()

    if args.clean and args.dst.exists():
        print(f"Cleaning {args.dst}")
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in args.splits:
        all_stats[split] = convert_split(
            split, args.images_root, args.labels_root, args.dst,
            args.mode, args.img_width, args.img_height, args.drop_empty,
        )

    yaml_lines = [f"path: {args.dst.resolve()}"]
    for split in args.splits:
        key = "train" if split == "train" else "val"
        yaml_lines.append(f"{key}: images/{split}")
    if "val" not in args.splits:
        yaml_lines.append("val: images/train  # no val converted")
    yaml_lines.append(f"nc: {len(CLASS_NAMES)}")
    yaml_lines.append("names:")
    for i, n in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {n}")
    (args.dst / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    (args.dst / "conversion_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")

    print(f"\nDone -> {args.dst}")
    for split, s in all_stats.items():
        print(f"  {split}: {s['written']} images, {s['boxes']} boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
