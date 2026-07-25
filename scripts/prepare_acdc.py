#!/usr/bin/env python3
"""Prepare ACDC dataset for YOLO training (6-class project format).

Iterates over the RGB images and reads the matching ground-truth mask per image.
ACDC train + val are used (test has no public GT). ACDC's own train/val split is
PRESERVED (train→train, val→val) — no re-stratification.

Supported raw layout (auto-detected):
  <raw-dir>/rgb_anon/{fog,night,rain,snow}/{train,val}/<sequence>/*_rgb_anon.png
  <raw-dir>/gt_panoptic/{fog,night,rain,snow}/{train,val}/<sequence>/*_gt_panoptic.png
  (also supports gt/panoptic/... , and *_gt_instanceIds.png / *_gt_labelIds.png)

Label decoding priority (per image):
  1. *_gt_instanceIds.png  — Cityscapes instance ids (id = labelId*1000 + inst). JSON-free.
  2. *_gt_panoptic.png + COCO panoptic JSON (segments_info)  — if a JSON is present.
  3. *_gt_panoptic.png alone — decoded as instance ids (best-effort, warns once).

Class mapping (Cityscapes labelId → project class):
  24 person, 25 rider→person, 26 car, 27 truck, 28 bus, 31 train→ignore,
  32 motorcycle, 33 bicycle.  Fixed order: person,bicycle,car,motorcycle,bus,truck.

Example:
  python scripts/prepare_acdc.py \\
    --raw-dir /workspace/datasets/acdc_raw \\
    --output-dir /workspace/datasets/acdc_6cls_yolo \\
    --imgsz 640 --seed 42 --clean
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.data_prep import (  # noqa: E402
    Box,
    PreparedSample,
    TARGET_CLASSES,
    clamp_box,
    clean_output_dir,
    export_yolo_dataset,
    make_yolo_dirs,
    normalize_class,
)

# Cityscapes labelId → name (things we care about; others ignored)
CITYSCAPES_ID_TO_NAME: dict[int, str] = {
    24: "person", 25: "rider", 26: "car", 27: "truck",
    28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle",
}
WEATHER_CONDITIONS = ["fog", "night", "rain", "snow"]
_WARNED = {"panoptic_guess": False}


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------


def _read_id_map(mask_path: Path) -> np.ndarray:
    """Read a mask PNG as a 2D integer id map (H, W)."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:  # RGB-encoded id: r + g*256 + b*65536
        arr = arr.astype(np.int64)
        return arr[:, :, 0] + arr[:, :, 1] * 256 + arr[:, :, 2] * 65536
    return arr.astype(np.int64)


def segment_to_bbox(seg_mask: np.ndarray):
    rows = np.any(seg_mask, axis=1)
    cols = np.any(seg_mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x1, y1, x2 + 1, y2 + 1


def boxes_from_instance_ids(id_map: np.ndarray) -> list[Box]:
    """Cityscapes instance ids: id = labelId*1000 + inst (things), or labelId (stuff)."""
    h, w = id_map.shape
    boxes: list[Box] = []
    for sid in np.unique(id_map):
        sid = int(sid)
        label_id = sid // 1000 if sid >= 1000 else sid
        name = CITYSCAPES_ID_TO_NAME.get(label_id)
        if name is None:
            continue
        bbox = segment_to_bbox(id_map == sid)
        if bbox is None:
            continue
        box = clamp_box(name, *bbox, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def boxes_from_panoptic_json(id_map: np.ndarray, segments, cat_lookup) -> list[Box]:
    """COCO panoptic: sequential segment ids in PNG, class from JSON segments_info."""
    h, w = id_map.shape
    boxes: list[Box] = []
    for seg in segments:
        name = cat_lookup.get(seg.get("category_id"))
        if name is None:
            continue
        bbox = segment_to_bbox(id_map == seg.get("id"))
        if bbox is None:
            continue
        box = clamp_box(name, *bbox, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


# ---------------------------------------------------------------------------
# Locate GT + optional panoptic JSON
# ---------------------------------------------------------------------------


def find_gt_root(raw_dir: Path) -> Path | None:
    for cand in (raw_dir / "gt_panoptic", raw_dir / "gt" / "panoptic", raw_dir / "gt"):
        if cand.exists():
            return cand
    return None


_JSON_CACHE: dict = {}


def build_json_lookup(gt_root: Path, split: str):
    """Index COCO panoptic segments by image stem for a whole split.

    ACDC ships one JSON per split at the gt root (e.g. gt_panoptic/train_gt_panoptic.json)
    covering all weather conditions. Older layouts keep a JSON per condition/split dir —
    both are supported. Result is cached per (gt_root, split).
    """
    cache_key = (str(gt_root), split)
    if cache_key in _JSON_CACHE:
        return _JSON_CACHE[cache_key]

    lookup: dict[str, list] = {}
    cats: dict[int, str | None] = dict(CITYSCAPES_ID_TO_NAME)

    # Root-level split JSON (standard ACDC), then per-condition/split dirs (fallback).
    jsons = list(gt_root.glob(f"*{split}*panoptic*.json")) + list(gt_root.glob(f"{split}_*.json"))
    if not jsons:
        for cond in WEATHER_CONDITIONS:
            sub = gt_root / cond / split
            if sub.exists():
                jsons += list(sub.rglob("*.json"))

    for jp in dict.fromkeys(jsons):  # dedup, keep order
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for cat in data.get("categories", []):
            cid = cat.get("id")
            nm = str(cat.get("name", "")).lower().strip()
            if cid not in cats:
                cats[cid] = nm if (normalize_class(nm) or nm in {"rider", "train"}) else None
        for ann in data.get("annotations", []):
            key = Path(ann.get("file_name", "")).stem.replace("_rgb_anon", "").replace("_gt_panoptic", "")
            if key:
                lookup[key] = ann.get("segments_info", [])

    _JSON_CACHE[cache_key] = (lookup, cats)
    return lookup, cats


# ---------------------------------------------------------------------------
# Load one condition/split
# ---------------------------------------------------------------------------


def load_split(raw_dir: Path, gt_root: Path, condition: str, split: str) -> list[PreparedSample]:
    img_root = raw_dir / "rgb_anon" / condition / split
    gt_split = gt_root / condition / split
    if not img_root.exists():
        print(f"  [SKIP] no RGB dir: {img_root}")
        return []

    json_lookup, cat_lookup = build_json_lookup(gt_root, split)
    samples: list[PreparedSample] = []
    skipped = Counter()

    for img in sorted(img_root.rglob("*_rgb_anon.png")):
        seq = img.parent.name
        base = img.name.replace("_rgb_anon.png", "")
        key = img.stem.replace("_rgb_anon", "")

        inst = gt_split / seq / f"{base}_gt_instanceIds.png"
        pano = gt_split / seq / f"{base}_gt_panoptic.png"

        boxes: list[Box] = []
        if inst.exists():                                  # 1. instance ids (best)
            boxes = boxes_from_instance_ids(_read_id_map(inst))
        elif pano.exists() and key in json_lookup:         # 2. panoptic + JSON
            boxes = boxes_from_panoptic_json(_read_id_map(pano), json_lookup[key], cat_lookup)
        elif pano.exists():                                # 3. panoptic alone (best-effort)
            if not _WARNED["panoptic_guess"]:
                print("  [WARN] no instanceIds/JSON — decoding *_gt_panoptic.png as instance ids. "
                      "Verify the box counts below look sane.")
                _WARNED["panoptic_guess"] = True
            boxes = boxes_from_instance_ids(_read_id_map(pano))
        else:
            skipped["no_mask"] += 1
            continue

        if not boxes:
            skipped["no_boxes"] += 1
            continue

        samples.append(PreparedSample(
            image=img, boxes=tuple(boxes), split_key=split, source="acdc", weather=condition,
        ))

    if skipped:
        print(f"  [{condition}/{split}] skipped: {dict(skipped)}")
    return samples


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True, help="ACDC root (has rgb_anon/ and gt_panoptic/)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", type=str, default="all", help="fog,rain,snow,night or 'all'")
    p.add_argument("--splits", type=str, default="train,val", help="ACDC splits to use (default train,val)")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    gt_root = find_gt_root(raw_dir)
    if gt_root is None:
        raise RuntimeError(f"No GT dir found under {raw_dir} (looked for gt_panoptic/, gt/panoptic/, gt/).")
    print(f"GT root: {gt_root}")

    conditions = WEATHER_CONDITIONS if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]

    clean_output_dir(output_dir, args.clean)
    make_yolo_dirs(output_dir, splits=tuple(splits))

    all_samples: list[PreparedSample] = []
    assignments: dict[int, str] = {}
    for condition in conditions:
        for split in splits:
            print(f"Loading {condition}/{split}...")
            found = load_split(raw_dir, gt_root, condition, split)
            print(f"  → {len(found)} images with boxes")
            for s in found:
                assignments[len(all_samples)] = split   # preserve ACDC split
                all_samples.append(s)

    if not all_samples:
        raise RuntimeError(
            "No labelled images found. Check --raw-dir and that GT masks/JSON exist.\n"
            "Run the diagnostic in docs/PHASE2_DATASET.md / ask if box decoding failed."
        )

    # Sanity report: boxes per class + images per split
    class_counter: Counter = Counter()
    split_counter: Counter = Counter()
    for i, s in enumerate(all_samples):
        split_counter[assignments[i]] += 1
        for b in s.boxes:
            class_counter[TARGET_CLASSES[b.class_id]] += 1
    print(f"\nImages per split: {dict(split_counter)}")
    print("Boxes per class:")
    for c in TARGET_CLASSES:
        print(f"  {c:<12}{class_counter.get(c, 0)}")

    export_yolo_dataset(
        samples=all_samples,
        assignments=assignments,
        raw_root=raw_dir,
        output_dir=output_dir,
        imgsz=args.imgsz,
        splits=tuple(splits),
    )


if __name__ == "__main__":
    main()
