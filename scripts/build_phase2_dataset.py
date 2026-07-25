#!/usr/bin/env python3
"""Build the Phase 2 merged training dataset for the final best model.

Merges TRAIN data from:
  1. XWOD train            (all)
  2. ACDC train            (all, optional — only if --acdc-root exists)
  3. BDD30K train replay   (a fraction, default 30%, to fight catastrophic forgetting)

Validation = XWOD val (the main val set). Test is intentionally left out of the merged
dataset to avoid leakage — final evaluation is done separately on XWOD test, DAWN val,
and ACDC val/test. In dataset.yaml, `test` points at `images/val` so Ultralytics has a
valid path, but real testing uses the held-out sets.

All source datasets are already in the fixed 6-class project order, so labels are copied
as-is (no remapping):
  0 person  1 bicycle  2 car  3 motorcycle  4 bus  5 truck

Filenames are prefixed (xwod_ / acdc_ / bdd_) to avoid collisions when merging.

Example:
  python scripts/build_phase2_dataset.py \\
    --xwod-root /content/workspace/datasets/xwod_6cls_yolo \\
    --bdd-root  /content/workspace/datasets/bdd100k_6cls_30k_yolo \\
    --acdc-root /content/workspace/datasets/acdc_6cls_yolo \\
    --out-root  /content/workspace/datasets/phase2_merged_yolo \\
    --bdd-replay-ratio 0.3 --seed 42 --mode symlink --clean
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

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Rare-class oversampling: a train image is duplicated by the LARGEST multiplier among
# the rare classes it contains. Images that only contain car/person/truck are untouched.
RARE_MULTIPLIERS = {"bicycle": 2, "motorcycle": 3, "bus": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs for a split, skipping images with no label.

    Only pairs whose label FILE exists are returned (empty label files — valid
    background negatives — are kept).
    """
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue  # skip samples without a label
        pairs.append((img, lbl))
    return pairs


def count_boxes(label_path: Path, counter: Counter) -> int:
    """Add per-class box counts from a YOLO label file. Returns number of boxes."""
    n = 0
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if 0 <= cid < len(TARGET_CLASSES):
                counter[TARGET_CLASSES[cid]] += 1
                n += 1
    except OSError:
        pass
    return n


def place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def classes_in_label(label_path: Path) -> set[str]:
    """Return the set of target-class names present in a YOLO label file."""
    present: set[str] = set()
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if 0 <= cid < len(TARGET_CLASSES):
                present.add(TARGET_CLASSES[cid])
    except OSError:
        pass
    return present


def oversample_rare(out_img_dir: Path, out_lbl_dir: Path, mode: str,
                    manifest_rows: list[dict]) -> tuple[int, Counter, dict]:
    """Duplicate already-written TRAIN images that contain rare classes.

    Multiplier per image = max RARE_MULTIPLIERS over the rare classes it contains
    (1 if none). Creates (mult-1) extra copies suffixed _osK.
    Returns (duplicate_images, duplicate_boxes_per_class, multiplier_distribution).
    """
    dup_images = 0
    dup_boxes: Counter = Counter()
    mult_dist: Counter = Counter()
    for lbl in sorted(out_lbl_dir.glob("*.txt")):
        present = classes_in_label(lbl)
        mult = max((RARE_MULTIPLIERS[c] for c in present if c in RARE_MULTIPLIERS), default=1)
        mult_dist[mult] += 1
        if mult <= 1:
            continue
        imgs = list(out_img_dir.glob(f"{lbl.stem}.*"))
        if not imgs:
            continue
        img = imgs[0]
        box_counter: Counter = Counter()
        count_boxes(lbl, box_counter)
        for k in range(1, mult):
            dup_img = out_img_dir / f"{lbl.stem}_os{k}{img.suffix.lower()}"
            dup_lbl = out_lbl_dir / f"{lbl.stem}_os{k}.txt"
            place(img, dup_img, mode)
            place(lbl, dup_lbl, mode)
            dup_images += 1
            for c, n in box_counter.items():
                dup_boxes[c] += n
            manifest_rows.append({
                "source_dataset": "oversample", "split": "train",
                "image_path": str(img), "label_path": str(lbl),
                "new_image": str(dup_img), "new_label": str(dup_lbl),
            })
    return dup_images, dup_boxes, dict(sorted(mult_dist.items()))


def write_split(
    pairs: list[tuple[Path, Path]],
    prefix: str,
    source_name: str,
    split_label: str,
    out_img_dir: Path,
    out_lbl_dir: Path,
    mode: str,
    manifest_rows: list[dict],
    class_counter: Counter,
) -> int:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img, lbl in pairs:
        new_stem = f"{prefix}{img.stem}"
        new_img = out_img_dir / f"{new_stem}{img.suffix.lower()}"
        new_lbl = out_lbl_dir / f"{new_stem}.txt"
        place(img, new_img, mode)
        place(lbl, new_lbl, mode)
        count_boxes(lbl, class_counter)
        manifest_rows.append({
            "source_dataset": source_name,
            "split": split_label,
            "image_path": str(img),
            "label_path": str(lbl),
            "new_image": str(new_img),
            "new_label": str(new_lbl),
        })
        written += 1
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Phase 2 merged dataset (XWOD + ACDC + BDD replay)")
    ap.add_argument("--xwod-root", type=Path, required=True)
    ap.add_argument("--bdd-root", type=Path, required=True)
    ap.add_argument("--acdc-root", type=Path, default=None, help="Optional — skipped if missing")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--bdd-replay-ratio", type=float, default=0.3,
                    help="Fraction of BDD30K train to replay (default 0.3). NOTE: 0.3*30000=9000 "
                         "can dominate the merge — prefer --bdd-replay-images for a small minority.")
    ap.add_argument("--bdd-replay-images", type=int, default=None,
                    help="Absolute number of BDD replay images (overrides --bdd-replay-ratio). "
                         "Recommended ~2000-2500 so BDD stays a minority.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--oversample-rare", action="store_true",
                    help="Duplicate train images containing rare classes "
                         "(bicycle x2, motorcycle x3, bus x3). Does not touch val/test.")
    ap.add_argument("--clean", action="store_true", help="Remove --out-root if it exists")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_root.resolve()
    rng = random.Random(args.seed)

    if args.clean and out.exists():
        print(f"Removing existing {out} ...")
        shutil.rmtree(out)

    out_train_img = out / "images" / "train"
    out_train_lbl = out / "labels" / "train"
    out_val_img = out / "images" / "val"
    out_val_lbl = out / "labels" / "val"

    manifest_rows: list[dict] = []
    class_counter: Counter = Counter()
    images_per_source: dict[str, int] = {}

    # ── 1. XWOD train (all) ──
    xwod_train = list_pairs(args.xwod_root.resolve(), "train")
    images_per_source["xwod_train"] = write_split(
        xwod_train, "xwod_", "xwod", "train",
        out_train_img, out_train_lbl, args.mode, manifest_rows, class_counter,
    )

    # ── 2. ACDC train (optional) ──
    acdc_train: list[tuple[Path, Path]] = []
    if args.acdc_root is not None and (args.acdc_root / "images" / "train").exists():
        acdc_train = list_pairs(args.acdc_root.resolve(), "train")
        images_per_source["acdc_train"] = write_split(
            acdc_train, "acdc_", "acdc", "train",
            out_train_img, out_train_lbl, args.mode, manifest_rows, class_counter,
        )
    else:
        images_per_source["acdc_train"] = 0
        print("ACDC not found — building XWOD + BDD replay only.")

    # ── 3. BDD30K train replay (anti-forgetting minority) ──
    # Replay is meant to be a SMALL minority so it doesn't drown the adverse-weather
    # data. Prefer an absolute count (--bdd-replay-images); the ratio is a fraction of
    # BDD30K and can easily overshoot (0.3 * 30000 = 9000, which dominates the merge).
    bdd_all = list_pairs(args.bdd_root.resolve(), "train")
    rng.shuffle(bdd_all)
    if args.bdd_replay_images is not None:
        n_replay = args.bdd_replay_images
    else:
        n_replay = int(round(len(bdd_all) * args.bdd_replay_ratio))
    n_replay = min(n_replay, len(bdd_all))
    bdd_replay = bdd_all[:n_replay]
    images_per_source["bdd_replay"] = write_split(
        bdd_replay, "bdd_", "bdd", "train",
        out_train_img, out_train_lbl, args.mode, manifest_rows, class_counter,
    )

    adverse = images_per_source["xwod_train"] + images_per_source["acdc_train"]
    if images_per_source["bdd_replay"] > adverse:
        print(f"\n  [WARN] BDD replay ({images_per_source['bdd_replay']}) exceeds adverse-weather "
              f"images ({adverse}). Clear-weather BDD dominates the merge and will dilute the "
              f"adverse-weather signal. Use a smaller --bdd-replay-images (e.g. ~2000-2500).\n")

    train_class_counter = Counter(class_counter)  # base train boxes (before oversampling)
    base_train_total = total_images = (images_per_source["xwod_train"]
                                       + images_per_source["acdc_train"]
                                       + images_per_source["bdd_replay"])

    # ── Rare-class oversampling (train only) ──
    dup_images = 0
    dup_boxes: Counter = Counter()
    mult_dist: dict = {}
    if args.oversample_rare:
        dup_images, dup_boxes, mult_dist = oversample_rare(
            out_train_img, out_train_lbl, args.mode, manifest_rows)
        print(f"Oversampling rare classes {RARE_MULTIPLIERS}: +{dup_images} duplicate images.")

    after_class_counter = train_class_counter + dup_boxes
    after_train_total = base_train_total + dup_images

    # ── Validation = XWOD val ──
    val_counter: Counter = Counter()
    xwod_val = list_pairs(args.xwod_root.resolve(), "val")
    images_per_source["xwod_val"] = write_split(
        xwod_val, "xwod_", "xwod", "val",
        out_val_img, out_val_lbl, args.mode, manifest_rows, val_counter,
    )

    # ── dataset.yaml ──
    yaml_text = (
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"   # placeholder — real testing uses held-out XWOD/DAWN/ACDC
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES))
    )
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # ── manifest.csv ──
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "split", "image_path", "label_path", "new_image", "new_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── stats.json ──
    stats = {
        "seed": args.seed,
        "mode": args.mode,
        "bdd_replay_ratio": args.bdd_replay_ratio,
        "images_per_source": images_per_source,
        "oversample_rare": {
            "enabled": args.oversample_rare,
            "multipliers": RARE_MULTIPLIERS if args.oversample_rare else {},
            "base_train_images": base_train_total,
            "duplicate_images": dup_images,
            "train_images_after": after_train_total,
            "multiplier_distribution": mult_dist,
        },
        "train_total": after_train_total,
        "val_total": images_per_source["xwod_val"],
        "train_boxes_per_class_before": {c: train_class_counter.get(c, 0) for c in TARGET_CLASSES},
        "train_boxes_per_class_after": {c: after_class_counter.get(c, 0) for c in TARGET_CLASSES},
        "val_boxes_per_class": {c: val_counter.get(c, 0) for c in TARGET_CLASSES},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Report ──
    print("\n" + "=" * 52)
    print("Phase 2 merged dataset built:", out)
    print("=" * 52)
    print(f"{'Source':<16}{'Split':<8}{'Images':>10}")
    print("-" * 34)
    print(f"{'XWOD':<16}{'train':<8}{images_per_source['xwod_train']:>10}")
    print(f"{'ACDC':<16}{'train':<8}{images_per_source['acdc_train']:>10}")
    print(f"{'BDD replay':<16}{'train':<8}{images_per_source['bdd_replay']:>10}")
    print("-" * 34)
    print(f"{'BASE TRAIN':<16}{'':<8}{base_train_total:>10}")
    if args.oversample_rare:
        print(f"{'+ duplicates':<16}{'':<8}{dup_images:>10}   dist={mult_dist}")
        print(f"{'TRAIN AFTER':<16}{'':<8}{after_train_total:>10}")
    print(f"{'XWOD':<16}{'val':<8}{images_per_source['xwod_val']:>10}")

    print(f"\n{'Class':<14}{'Boxes before':>14}{'Boxes after':>14}{'Val boxes':>12}")
    print("-" * 54)
    for c in TARGET_CLASSES:
        print(f"{c:<14}{train_class_counter.get(c, 0):>14}{after_class_counter.get(c, 0):>14}"
              f"{val_counter.get(c, 0):>12}")

    print(f"\ndataset.yaml + manifest.csv + stats.json → {out}")


if __name__ == "__main__":
    main()
