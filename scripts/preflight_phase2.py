#!/usr/bin/env python3
"""Preflight checks for the Phase 2 RT-DETR run — no GPU, no training.

Verifies the resolved config, checkpoint, dataset, class order, and the
merged-dataset stats/manifest are consistent BEFORE any GPU time is spent.

Usage:
  python scripts/preflight_phase2.py \\
    --config configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.common import load_experiment_config, resolve_from_root  # noqa: E402

EXPECTED_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
EXPECTED_VAL_COUNT = 1001
EXPECTED_STATS = {
    "xwod_train": 6006,
    "acdc_train": 1182,
    "bdd_replay": 2000,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 preflight checks (no GPU, no training)")
    ap.add_argument("--config", required=True)
    return ap.parse_args()


class Check:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, message: str) -> None:
        print(f"  [OK]   {message}")

    def fail(self, message: str) -> None:
        self.failed = True
        print(f"  [FAIL] {message}")

    def warn(self, message: str) -> None:
        print(f"  [WARN] {message}")


def load_dataset_yaml(data_yaml: Path) -> dict:
    with data_yaml.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def dataset_root(data_yaml: Path, data: dict) -> Path:
    raw = data.get("path", data_yaml.parent)
    root = Path(raw)
    return root if root.is_absolute() else data_yaml.parent / root


def split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def count_pairs(img_dir: Path, lbl_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    n = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if (lbl_dir / f"{img.stem}.txt").exists():
            n += 1
    return n


def find_broken_symlinks(root: Path) -> list[Path]:
    broken: list[Path] = []
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = root / sub
        if not d.exists():
            continue
        for entry in d.iterdir():
            if entry.is_symlink() and not entry.exists():
                broken.append(entry)
    return broken


def main() -> None:
    args = parse_args()
    check = Check()

    print("=" * 60)
    print("PHASE 2 PREFLIGHT")
    print("=" * 60)

    # 1. Load config
    config = load_experiment_config(args.config)

    checkpoint = Path(config["model"])
    data_yaml = resolve_from_root(config["data"])
    project = resolve_from_root(config["project"])
    name = config["name"]

    print("\nResolved config:")
    print(f"  model (checkpoint) : {checkpoint}")
    print(f"  data (dataset yaml): {data_yaml}")
    print(f"  project             : {project}")
    print(f"  name                : {name}")
    print(f"  epochs              : {config.get('epochs')}")
    print(f"  batch               : {config.get('batch')}")
    print(f"  imgsz               : {config.get('imgsz')}")
    print(f"  optimizer           : {config.get('optimizer')}")
    print(f"  lr0                 : {config.get('lr0')}")
    print(f"  seed                : {config.get('seed')}")

    print("\nChecks:")

    # 3. checkpoint exists
    if checkpoint.exists():
        check.ok(f"checkpoint exists: {checkpoint}")
    else:
        check.fail(f"checkpoint missing: {checkpoint}")

    # 4. dataset.yaml exists
    if data_yaml.exists():
        check.ok(f"dataset.yaml exists: {data_yaml}")
    else:
        check.fail(f"dataset.yaml missing: {data_yaml}")
        # Nothing further can be checked without it.
        _finish(check)
        return

    data = load_dataset_yaml(data_yaml)
    root = dataset_root(data_yaml, data)

    # 5. class names EXACT
    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names)]
    else:
        ordered = list(names) if names else []
    if ordered == EXPECTED_CLASSES:
        check.ok(f"class names match expected order: {ordered}")
    else:
        check.fail(f"class names mismatch: expected {EXPECTED_CLASSES}, got {ordered}")

    # 6. merged validation image/label pair count
    val_img, val_lbl = split_dirs(root, "val")
    val_count = count_pairs(val_img, val_lbl)
    if val_count == EXPECTED_VAL_COUNT:
        check.ok(f"val image/label pairs = {val_count}")
    else:
        check.fail(f"val image/label pairs = {val_count}, expected {EXPECTED_VAL_COUNT}")

    # 7. stats.json
    stats_path = root / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        images_per_source = stats.get("images_per_source", {})
        for key, expected in EXPECTED_STATS.items():
            actual = images_per_source.get(key)
            if actual == expected:
                check.ok(f"stats.json {key} = {actual}")
            else:
                check.fail(f"stats.json {key} = {actual}, expected {expected}")
    else:
        check.warn(f"stats.json not found at {stats_path} — skipping source-count checks")

    # 8. broken symlinks
    broken = find_broken_symlinks(root)
    if not broken:
        check.ok("no broken symlinks")
    else:
        check.fail(f"{len(broken)} broken symlink(s), e.g. {broken[:3]}")

    # 9. train/val overlap via manifest.csv
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        train_sources: set[str] = set()
        val_sources: set[str] = set()
        with manifest_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                src = row.get("image_path", "")
                if row.get("split") == "train":
                    train_sources.add(src)
                elif row.get("split") == "val":
                    val_sources.add(src)
        overlap = train_sources & val_sources
        if not overlap:
            check.ok("no train/val source overlap in manifest.csv")
        else:
            check.fail(f"{len(overlap)} source image(s) appear in both train and val: "
                       f"{sorted(overlap)[:3]}")
    else:
        check.warn(f"manifest.csv not found at {manifest_path} — skipping overlap check")

    _finish(check)


def _finish(check: Check) -> None:
    print()
    if check.failed:
        print("PHASE 2 PREFLIGHT: FAIL")
        sys.exit(1)
    print("PHASE 2 PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
