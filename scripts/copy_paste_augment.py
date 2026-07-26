#!/usr/bin/env python3
"""Copy-paste augmentation for rare/small classes (experiment #3).

Takes an already-built merged train set (e.g. phase2_merged_yolo_v3) and produces a NEW
dataset = base (unchanged) + extra augmented TRAIN images. Each augmented image is a base
train image onto which 1-3 real crops of rare/small instances (bicycle, motorcycle, bus,
small person) have been pasted, with matching YOLO labels appended.

Directly targets the dominant error found by error_analysis (missed small/rare objects,
esp. ACDC). Validation/test are copied unchanged.

Guards: pasted boxes must not overlap existing boxes (IoU < --max-iou), stay inside the
image with a margin, and the source crop must be small/medium and a valid size.

Example:
  python scripts/copy_paste_augment.py \\
    --base-root /workspace/datasets/phase2_merged_yolo_v3 \\
    --out-root  /workspace/datasets/phase2_merged_yolo_v5_copypaste \\
    --n-aug-images 3000 --max-per-image 3 --seed 42 --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes to build the paste bank from + sampling weight (rare classes weighted higher).
PASTE_WEIGHTS = {"bicycle": 3, "motorcycle": 3, "bus": 3, "person": 1}
PERSON_MAX_AREA = 96 * 96          # person crops only if small/medium
MIN_SIDE = 12                       # px, reject tiny/degenerate crops
MAX_BOX_FRAC = 0.35                 # reject crops larger than this fraction of the image


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def list_pairs(root: Path, split: str):
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    pairs = []
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMAGE_SUFFIXES:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def read_label(lbl: Path):
    """Return list of (cls, cx, cy, w, h) normalized."""
    out = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
    return out


def source_of(stem: str) -> str:
    for s in ("xwod", "acdc", "bdd"):
        if stem.startswith(s + "_"):
            return s
    return "other"


def place(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst) if mode == "copy" else os.symlink(src.resolve(), dst)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", type=Path, required=True, help="Existing merged dataset (e.g. v3)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-aug-images", type=int, default=3000)
    ap.add_argument("--max-per-image", type=int, default=3)
    ap.add_argument("--min-per-image", type=int, default=1)
    ap.add_argument("--max-iou", type=float, default=0.20, help="Reject paste if IoU with any box exceeds this")
    ap.add_argument("--scale-range", type=float, nargs=2, default=(0.8, 1.2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--clean", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.out_root.resolve()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.clean and out.exists():
        print(f"Removing {out} ...")
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    base_boxes = Counter()          # per class, base
    base_size = Counter()           # per size bucket, base

    # ── 1. Copy base train + val unchanged ──
    train_pairs = list_pairs(args.base_root.resolve(), "train")
    val_pairs = list_pairs(args.base_root.resolve(), "val")
    for img, lbl in train_pairs:
        place(img, out / "images" / "train" / img.name, args.mode)
        place(lbl, out / "labels" / "train" / lbl.name, args.mode)
        manifest.append({"kind": "base", "split": "train", "image": str(out / "images" / "train" / img.name)})
    for img, lbl in val_pairs:
        place(img, out / "images" / "val" / img.name, args.mode)
        place(lbl, out / "labels" / "val" / lbl.name, args.mode)
    print(f"Base copied: {len(train_pairs)} train, {len(val_pairs)} val")

    # count base boxes / sizes (need image sizes → read label + one imread per image with instances)
    label_cache = {}
    for img, lbl in train_pairs:
        rows = read_label(lbl)
        label_cache[img] = rows
        for cls, cx, cy, w, h in rows:
            base_boxes[TARGET_CLASSES[cls]] += 1

    # ── 2. Build paste bank: crop rare/small instances from base train ──
    bank = {c: [] for c in PASTE_WEIGHTS}      # class name -> list of crop images (BGR)
    skipped = Counter()
    for img, lbl in train_pairs:
        rows = label_cache[img]
        if not any(TARGET_CLASSES[r[0]] in PASTE_WEIGHTS for r in rows):
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        for cls, cx, cy, w, h in rows:
            name = TARGET_CLASSES[cls]
            # absolute box + size bucket (for base_size)
            bw, bh = w * W, h * H
            base_size[size_bucket(bw * bh)] += 1
            if name not in PASTE_WEIGHTS:
                continue
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
            cw, ch = x2 - x1, y2 - y1
            if min(cw, ch) < MIN_SIDE:
                skipped["crop_too_small"] += 1; continue
            if cw > MAX_BOX_FRAC * W or ch > MAX_BOX_FRAC * H:
                skipped["crop_too_large"] += 1; continue
            if name == "person" and (cw * ch) >= PERSON_MAX_AREA:
                skipped["person_not_small"] += 1; continue
            crop = im[y1:y2, x1:x2].copy()
            if crop.size == 0:
                skipped["crop_invalid"] += 1; continue
            bank[name].append((crop, source_of(img.stem)))
    bank_counts = {c: len(v) for c, v in bank.items()}
    print(f"Instance bank: {bank_counts}")
    if sum(bank_counts.values()) == 0:
        raise RuntimeError("Empty paste bank — no valid rare/small instances found.")

    classes_avail = [c for c in PASTE_WEIGHTS if bank[c]]
    weights = [PASTE_WEIGHTS[c] for c in classes_avail]

    # ── 3. Generate augmented images ──
    pasted_per_class = Counter()
    pasted_size = Counter()
    pasted_source = Counter()
    n_created = 0
    bg_pool = [img for img, _ in train_pairs]

    attempts = 0
    while n_created < args.n_aug_images and attempts < args.n_aug_images * 3:
        attempts += 1
        bg_path = rng.choice(bg_pool)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        existing = []
        for cls, cx, cy, w, h in label_cache[bg_path]:
            existing.append((int((cx - w / 2) * W), int((cy - h / 2) * H),
                             int((cx + w / 2) * W), int((cy + h / 2) * H)))
        new_lines = []
        n_target = rng.randint(args.min_per_image, args.max_per_image)
        pasted_here = 0
        for _ in range(n_target):
            cname = rng.choices(classes_avail, weights=weights, k=1)[0]
            crop, src = rng.choice(bank[cname])
            ch0, cw0 = crop.shape[:2]
            s = rng.uniform(*args.scale_range)
            nw, nh = max(MIN_SIDE, int(cw0 * s)), max(MIN_SIDE, int(ch0 * s))
            if nw >= W - 4 or nh >= H - 4:
                continue
            placed = False
            for _try in range(10):
                px = rng.randint(2, W - nw - 2)
                py = rng.randint(2, H - nh - 2)
                nb = (px, py, px + nw, py + nh)
                if all(iou(nb, e) < args.max_iou for e in existing):
                    scaled = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    bg[py:py + nh, px:px + nw] = scaled
                    existing.append(nb)
                    cls_id = TARGET_CLASSES.index(cname)
                    new_lines.append(f"{cls_id} {(px + nw / 2) / W:.6f} {(py + nh / 2) / H:.6f} "
                                     f"{nw / W:.6f} {nh / H:.6f}")
                    pasted_per_class[cname] += 1
                    pasted_size[size_bucket(nw * nh)] += 1
                    pasted_source[src] += 1
                    pasted_here += 1
                    placed = True
                    break
            if not placed:
                skipped["overlap"] += 1
        if pasted_here == 0:
            skipped["no_paste"] += 1
            continue
        # save composite + label (base label lines + pasted lines)
        stem = f"cp_{n_created:05d}_{bg_path.stem}"
        out_img = out / "images" / "train" / f"{stem}.jpg"
        out_lbl = out / "labels" / "train" / f"{stem}.txt"
        cv2.imwrite(str(out_img), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        orig = (args.base_root / "labels" / "train" / f"{bg_path.stem}.txt").read_text(encoding="utf-8").rstrip("\n")
        out_lbl.write_text((orig + "\n" if orig else "") + "\n".join(new_lines) + "\n", encoding="utf-8")
        manifest.append({"kind": "copypaste", "split": "train", "image": str(out_img)})
        n_created += 1
        if n_created % 500 == 0:
            print(f"  ...{n_created}/{args.n_aug_images} augmented")

    # ── 4. dataset.yaml ──
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/val\nnc: 6\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES)), encoding="utf-8")

    # ── 5. manifest.csv ──
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["kind", "split", "image"]); wr.writeheader(); wr.writerows(manifest)

    # ── 6. stats + copy_paste_stats ──
    after_boxes = Counter(base_boxes)
    for c, n in pasted_per_class.items():
        after_boxes[c] += n
    after_size = Counter(base_size)
    for s, n in pasted_size.items():
        after_size[s] += n

    cp_stats = {
        "seed": args.seed,
        "base_train_images": len(train_pairs),
        "augmented_images": n_created,
        "train_images_after": len(train_pairs) + n_created,
        "instance_bank_counts": bank_counts,
        "pasted_instances_per_class": dict(pasted_per_class),
        "pasted_by_source": dict(pasted_source),
        "pasted_by_size": dict(pasted_size),
        "box_count_before": {c: base_boxes.get(c, 0) for c in TARGET_CLASSES},
        "box_count_after": {c: after_boxes.get(c, 0) for c in TARGET_CLASSES},
        "size_before": {s: base_size.get(s, 0) for s in ("small", "medium", "large")},
        "size_after": {s: after_size.get(s, 0) for s in ("small", "medium", "large")},
        "skipped": dict(skipped),
    }
    (out / "copy_paste_stats.json").write_text(json.dumps(cp_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({
        "base": str(args.base_root), "train_total": len(train_pairs) + n_created,
        "val_total": len(val_pairs), "copy_paste": cp_stats}, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── report ──
    print("\n" + "=" * 52)
    print("Copy-paste dataset built:", out)
    print("=" * 52)
    print(f"Base train {len(train_pairs)} + augmented {n_created} = {len(train_pairs)+n_created}")
    print(f"Pasted instances: {dict(pasted_per_class)}  (by source: {dict(pasted_source)})")
    print(f"\n{'Class':<12}{'Before':>10}{'After':>10}")
    for c in TARGET_CLASSES:
        print(f"{c:<12}{base_boxes.get(c,0):>10}{after_boxes.get(c,0):>10}")
    print(f"\n{'Size':<8}{'Before':>10}{'After':>10}")
    for s in ("small", "medium", "large"):
        print(f"{s:<8}{base_size.get(s,0):>10}{after_size.get(s,0):>10}")
    print(f"\nSkipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
