#!/usr/bin/env python3
"""Remap BDD100K YOLO (10-class Kaggle) → 6-class project format with condition-aware sampling.

Download source: https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo
Requires: original BDD100K JSON labels for weather metadata (condition-aware mode).

Usage (condition-aware, requires JSON):
  python3 scripts/prepare_bdd100k_yolo.py \
    --src /workspace/datasets/bdd100k_yolo_raw \
    --json /workspace/datasets/bdd100k_raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --dst /workspace/datasets/bdd100k_6cls_30k_yolo \
    --subset-size 30000 --seed 42

Usage (random subset, no JSON needed):
  python3 scripts/prepare_bdd100k_yolo.py \
    --src /workspace/datasets/bdd100k_yolo_raw \
    --dst /workspace/datasets/bdd100k_6cls_30k_yolo \
    --subset-size 30000 --seed 42 --subset-mode random

Usage (full train set):
  python3 scripts/prepare_bdd100k_yolo.py \
    --src /workspace/datasets/bdd100k_yolo_raw \
    --dst /workspace/datasets/bdd100k_6cls_full_yolo \
    --full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Class mapping: BDD100K 10-class → project 6-class
# ---------------------------------------------------------------------------

# Source (Kaggle YOLO): person(0) rider(1) car(2) bus(3) truck(4)
#                       bike(5) motor(6) traffic light(7) traffic sign(8) train(9)
# Target: person(0) bicycle(1) car(2) motorcycle(3) bus(4) truck(5)
REMAP: dict[int, int] = {0: 0, 1: 0, 2: 2, 3: 4, 4: 5, 5: 1, 6: 3}
# 7, 8, 9 → skipped

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# ---------------------------------------------------------------------------
# Condition-aware sampling
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["foggy", "snowy", "rainy", "dawn_dusk", "night", "other"]
CAPS: dict[str, int | None] = {
    "foggy":     None,
    "snowy":     5000,
    "rainy":     5000,
    "dawn_dusk": 5000,
    "night":     6000,
    "other":     None,
}


def get_condition(weather: str, timeofday: str) -> str:
    w, t = weather.lower().strip(), timeofday.lower().strip()
    if w == "foggy":      return "foggy"
    if w == "snowy":      return "snowy"
    if w == "rainy":      return "rainy"
    if t == "dawn/dusk":  return "dawn_dusk"
    if t == "night":      return "night"
    return "other"


def condition_aware_subset(
    available: list[str],
    name_to_cond: dict[str, str],
    subset_size: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    pools: dict[str, list[str]] = {c: [] for c in CONDITION_ORDER}
    for name in available:
        pools[name_to_cond.get(name, "other")].append(name)

    print(f"\n{'Condition':<12} {'Available':>10} {'Cap':>8} {'Selected':>10}")
    print("-" * 44)

    selected: list[str] = []
    for cond in CONDITION_ORDER[:-1]:
        pool = pools[cond]
        rng.shuffle(pool)
        cap = CAPS[cond]
        take = pool if cap is None else pool[:cap]
        selected.extend(take)
        print(f"{cond:<12} {len(pool):>10} {str(cap) if cap else 'all':>8} {len(take):>10}")

    remaining = subset_size - len(selected)
    other = pools["other"]
    rng.shuffle(other)
    take_other = other[:max(0, remaining)]
    selected.extend(take_other)
    print(f"{'other':<12} {len(other):>10} {'fill':>8} {len(take_other):>10}")
    print(f"\nTotal selected: {len(selected)}/{subset_size}\n")
    return selected


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------


def convert_label(src: Path, dst: Path) -> bool:
    """Remap class IDs, skip ignored classes. Returns False if no boxes remain."""
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid in REMAP:
            lines.append(f"{REMAP[cid]} {' '.join(parts[1:])}")
    if not lines:
        return False
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------------


def process_split(
    names: list[str],
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst: Path,
    split: str,
    copy_images: bool,
) -> int:
    out_i = dst / "images" / split
    out_l = dst / "labels" / split
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)

    written = 0
    for name in names:
        src_lbl = src_lbl_dir / (Path(name).stem + ".txt")
        src_img = src_img_dir / name
        if not src_lbl.exists() or not src_img.exists():
            continue
        dst_lbl = out_l / src_lbl.name
        if not convert_label(src_lbl, dst_lbl):
            dst_lbl.unlink(missing_ok=True)
            continue
        dst_img = out_i / name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            os.symlink(src_img.resolve(), dst_img)
        written += 1

    print(f"  {split}: {written} images written")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Remap BDD100K YOLO (Kaggle 10-class) → 6-class project format"
    )
    ap.add_argument("--src", type=Path, required=True,
                    help="Root of Kaggle BDD100K YOLO dataset (has train/val/test subdirs)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--json", type=Path, default=None,
                    help="BDD100K train JSON labels (for condition-aware sampling)")
    ap.add_argument("--subset-size", type=int, default=30000)
    ap.add_argument("--subset-mode", choices=["condition_aware", "random"],
                    default="condition_aware")
    ap.add_argument("--full", action="store_true",
                    help="Use full train set (ignore --subset-size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy images instead of symlink (uses more disk)")
    ap.add_argument("--clean", action="store_true",
                    help="Remove --dst if it already exists")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.dst.resolve()

    if args.clean and dst.exists():
        print(f"Removing {dst} ...")
        shutil.rmtree(dst)

    src_train_img = args.src / "train" / "images"
    src_train_lbl = args.src / "train" / "labels"
    src_val_img   = args.src / "val" / "images"
    src_val_lbl   = args.src / "val" / "labels"

    # Build train names list
    all_train = [p.name for p in sorted(src_train_img.glob("*.jpg"))]

    if args.full:
        train_names = all_train
        print(f"Using full train set: {len(train_names)} images")
    elif args.subset_mode == "condition_aware" and args.json:
        print("Loading JSON metadata for condition-aware sampling...")
        data = json.loads(args.json.read_text(encoding="utf-8"))
        name_to_cond = {
            f["name"]: get_condition(
                f.get("attributes", {}).get("weather", ""),
                f.get("attributes", {}).get("timeofday", ""),
            )
            for f in data
        }
        available = [n for n in all_train if (src_train_img / n).exists()]
        train_names = condition_aware_subset(
            available, name_to_cond, args.subset_size, args.seed
        )
    else:
        if args.subset_mode == "condition_aware" and not args.json:
            print("Warning: --json not provided, falling back to random sampling.")
        rng = random.Random(args.seed)
        pool = list(all_train)
        rng.shuffle(pool)
        train_names = pool[:args.subset_size]
        print(f"Random subset: {len(train_names)} images\n")

    val_names = [p.name for p in sorted(src_val_img.glob("*.jpg"))]

    print("Processing splits...")
    process_split(train_names, src_train_img, src_train_lbl, dst, "train", args.copy_images)
    process_split(val_names,   src_val_img,   src_val_lbl,   dst, "val",   args.copy_images)

    (dst / "dataset.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnc: 6\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"\nDone: {dst}")


if __name__ == "__main__":
    main()
