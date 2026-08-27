#!/usr/bin/env python3
"""Preflight checks for the Phase 2 RT-DETR run — no GPU, no training.

Enforces the OFFICIAL Phase 2 protocol before any GPU time is spent:
  - single model: RT-DETR-L, continued from stage2_xwod_rtdetr_from_bdd30k/weights/best.pt
  - dataset: phase2_merged = XWOD train (6006) + ACDC train (1182) + FULL BDD train (30000)
    => base_train_images = 37188, before rare-class oversampling
  - batch = 16, lr0 = 0.00005, seed = 42 (training config) AND stats.json seed = 42,
    bdd_mode = "full" (dataset-build provenance — a legacy replay-mode build must
    never pass, even if it happens to contain 30000 BDD samples)
  - validation = XWOD val (1001) only
  - rare-class oversampling (bicycle x2, motorcycle x3, bus x3) enabled, train-only
  - no XWOD test / ACDC val / ACDC test / BDD val / BDD test / DAWN data anywhere in
    the merged train or val split

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
EXPECTED_DATASET_KEY = "phase2_merged"
EXPECTED_FROM_RUN = "stage2_xwod_rtdetr_from_bdd30k"
EXPECTED_BATCH = 16
EXPECTED_LR0 = 0.00005
EXPECTED_SEED = 42

EXPECTED_VAL_COUNT = 1001
EXPECTED_SOURCE_COUNTS = {
    "xwod_train": 6006,
    "acdc_train": 1182,
    "bdd_train": 30000,
    "xwod_val": 1001,
}
EXPECTED_BASE_TRAIN_IMAGES = 37188
EXPECTED_RARE_MULTIPLIERS = {"bicycle": 2, "motorcycle": 3, "bus": 3}

# source_dataset values that are legitimately allowed to appear in manifest.csv.
ALLOWED_MANIFEST_SOURCES = {"xwod", "acdc", "bdd", "oversample"}
# Any of these substrings appearing in an original source path is a leakage red flag.
FORBIDDEN_PATH_MARKERS = ("dawn",)

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

    def equal(self, label: str, actual, expected) -> None:
        if actual == expected:
            self.ok(f"{label} = {actual}")
        else:
            self.fail(f"{label} = {actual}, expected {expected}")


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


def source_split_segment(path_str: str) -> str | None:
    """Extract the split directory (train/val/test/...) from an original 'images/<split>/...' path."""
    parts = Path(path_str).parts
    for i, part in enumerate(parts):
        if part == "images" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def check_manifest_leakage(manifest_path: Path, check: Check) -> None:
    """Verify every manifest row's ORIGINAL source split matches its MERGED split, and
    that no DAWN data or unknown source ever appears."""
    with manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    bad_sources = {r["source_dataset"] for r in rows} - ALLOWED_MANIFEST_SOURCES
    if bad_sources:
        check.fail(f"manifest.csv contains unexpected source_dataset value(s): {sorted(bad_sources)}")
    else:
        check.ok(f"manifest.csv source_dataset values are all expected: {sorted(ALLOWED_MANIFEST_SOURCES)}")

    dawn_rows = [r for r in rows if any(m in r["image_path"].lower() for m in FORBIDDEN_PATH_MARKERS)]
    if dawn_rows:
        check.fail(f"{len(dawn_rows)} manifest row(s) reference DAWN data: {dawn_rows[0]['image_path']}")
    else:
        check.ok("no DAWN data referenced in manifest.csv")

    mismatched: list[str] = []
    for row in rows:
        if row["source_dataset"] == "oversample":
            continue  # duplicates reference the already-merged image, not an original source split
        merged_split = row.get("split")
        original_split = source_split_segment(row.get("image_path", ""))
        if original_split != merged_split:
            mismatched.append(
                f"{row['image_path']} -> merged split={merged_split!r} but original split={original_split!r}"
            )
    if mismatched:
        check.fail(f"{len(mismatched)} manifest row(s) whose original split doesn't match the merged "
                   f"split (possible val/test leakage), e.g. {mismatched[0]}")
    else:
        check.ok("every manifest row's original split matches its merged split (train<-train, val<-val)")

    train_sources = {r["image_path"] for r in rows if r["split"] == "train"}
    val_sources = {r["image_path"] for r in rows if r["split"] == "val"}
    overlap = train_sources & val_sources
    if overlap:
        check.fail(f"{len(overlap)} source image(s) appear in both train and val: {sorted(overlap)[:3]}")
    else:
        check.ok("no train/val source overlap in manifest.csv")


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

    # 2. Protocol invariants declared directly in the resolved config.
    check.equal("dataset", config.get("dataset"), EXPECTED_DATASET_KEY)
    check.equal("from_run", config.get("from_run"), EXPECTED_FROM_RUN)
    check.equal("batch", config.get("batch"), EXPECTED_BATCH)
    check.equal("lr0", config.get("lr0"), EXPECTED_LR0)
    check.equal("seed", config.get("seed"), EXPECTED_SEED)

    # Checkpoint must resolve via from_run — never a hard-coded absolute path.
    expected_checkpoint = project / EXPECTED_FROM_RUN / "weights" / "best.pt"
    if checkpoint == expected_checkpoint:
        check.ok(f"checkpoint resolves via from_run: {checkpoint}")
    else:
        check.fail(f"checkpoint does not resolve to <project>/{EXPECTED_FROM_RUN}/weights/best.pt: "
                   f"got {checkpoint}, expected {expected_checkpoint}")

    # 3. checkpoint exists
    if checkpoint.exists():
        check.ok(f"checkpoint exists: {checkpoint}")
    else:
        check.fail(f"checkpoint missing: {checkpoint}")

    # 4. dataset.yaml exists
    if not data_yaml.exists():
        check.fail(f"dataset.yaml missing: {data_yaml}")
        _finish(check)
        return
    check.ok(f"dataset.yaml exists: {data_yaml}")

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
    check.equal("val image/label pairs", val_count, EXPECTED_VAL_COUNT)

    # 7. stats.json — REQUIRED (not optional) for the official protocol.
    stats_path = root / "stats.json"
    if not stats_path.exists():
        check.fail(f"stats.json missing: {stats_path}")
        stats = None
    else:
        check.ok(f"stats.json exists: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if stats is not None:
        # Dataset-build provenance: must have been built with --bdd-use-all (official
        # mode), and with the official seed. A legacy replay-mode dataset must never
        # pass, even if it happens to contain 30000 BDD samples by coincidence.
        check.equal("stats.json bdd_mode", stats.get("bdd_mode"), "full")
        check.equal("stats.json seed", stats.get("seed"), EXPECTED_SEED)

        images_per_source = stats.get("images_per_source", {})
        for key, expected in EXPECTED_SOURCE_COUNTS.items():
            check.equal(f"stats.json images_per_source.{key}", images_per_source.get(key), expected)

        oversample = stats.get("oversample_rare", {})
        check.equal("stats.json oversample_rare.enabled", oversample.get("enabled"), True)
        check.equal("stats.json oversample_rare.multipliers", oversample.get("multipliers"),
                    EXPECTED_RARE_MULTIPLIERS)
        base_train_images = oversample.get("base_train_images")
        check.equal("stats.json oversample_rare.base_train_images", base_train_images,
                    EXPECTED_BASE_TRAIN_IMAGES)

        duplicate_images = oversample.get("duplicate_images")
        train_total = stats.get("train_total")
        if isinstance(duplicate_images, int) and isinstance(base_train_images, int) \
                and isinstance(train_total, int):
            expected_train_total = base_train_images + duplicate_images
            check.equal("stats.json train_total (base_train_images + duplicate_images)",
                        train_total, expected_train_total)
            if train_total < EXPECTED_BASE_TRAIN_IMAGES:
                check.fail(f"stats.json train_total ({train_total}) is below the required base of "
                           f"{EXPECTED_BASE_TRAIN_IMAGES} — oversampling must only ADD images")
            else:
                check.ok(f"stats.json train_total ({train_total}) >= base "
                         f"{EXPECTED_BASE_TRAIN_IMAGES}")
        else:
            check.fail("stats.json missing duplicate_images/base_train_images/train_total — "
                       "cannot verify train_total consistency")

        check.equal("stats.json val_total", stats.get("val_total"), EXPECTED_VAL_COUNT)

    # 8. broken symlinks
    broken = find_broken_symlinks(root)
    if not broken:
        check.ok("no broken symlinks")
    else:
        check.fail(f"{len(broken)} broken symlink(s), e.g. {broken[:3]}")

    # 9. manifest.csv — REQUIRED; verifies train/val overlap AND source-split leakage.
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        check.fail(f"manifest.csv missing: {manifest_path}")
    else:
        check.ok(f"manifest.csv exists: {manifest_path}")
        check_manifest_leakage(manifest_path, check)

    _finish(check)


def _finish(check: Check) -> None:
    print()
    if check.failed:
        print("PHASE 2 PREFLIGHT: FAIL")
        sys.exit(1)
    print("PHASE 2 PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
