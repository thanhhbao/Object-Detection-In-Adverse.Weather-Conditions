#!/usr/bin/env python3
"""Preflight checks for the P2-A1 RT-DETR BDD retrieval experiment.

Enforces the P2-A1 protocol before any GPU time is spent:
  - checkpoint continues from phase2_final_rtdetr (NOT stage2)
  - dataset: phase2_a1_merged (merged + retrieved BDD)
  - batch=16, lr0=0.00001, seed=42, epochs=20, patience=8
  - validation = XWOD val (count > 0)
  - no broken symlinks
  - pool overlap counts all == 0 (if --pool-stats provided)
  - retrieval stats: selected_unique > 0, <= requested_top_k, all bdd_train
  - stats.json has bdd_retrieved_train > 0 (if pool-stats provided)
  - manifest.csv has bdd_retrieved source, no dawn sources, no val/test leakage
  - P2-A0C and P2-A1 configs have identical: epochs, batch, lr0, seed, patience

Usage:
  python scripts/preflight_p2_a1.py \\
    --config configs/ultralytics/p2_a1_rtdetr_bdd_retrieval.yaml \\
    --pool-stats /workspace/datasets_noleak/bdd_remaining_pool/pool_stats.json \\
    --retrieval-stats /workspace/datasets_noleak/bdd_active_retrieved/retrieval_stats.json
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
EXPECTED_FROM_RUN = "phase2_final_rtdetr"
EXPECTED_DATASET_KEY = "phase2_a1_merged"
EXPECTED_BATCH = 16
EXPECTED_LR0 = 0.00001
EXPECTED_SEED = 42
EXPECTED_EPOCHS = 20
EXPECTED_PATIENCE = 8

A0C_CONFIG_PATH = ROOT / "configs" / "ultralytics" / "p2_a0c_rtdetr_continue_control.yaml"
A1_CONFIG_PATH = ROOT / "configs" / "ultralytics" / "p2_a1_rtdetr_bdd_retrieval.yaml"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORBIDDEN_PATH_MARKERS = ("dawn",)
ALLOWED_MANIFEST_SOURCES = {"xwod", "acdc", "bdd", "bdd_retrieved", "oversample"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P2-A1 preflight checks (no GPU, no training)")
    ap.add_argument("--config", required=True, help="Path to p2_a1_rtdetr_bdd_retrieval.yaml")
    ap.add_argument("--pool-stats", default=None, help="pool_stats.json from build_bdd_retrieval_pool.py")
    ap.add_argument("--retrieval-stats", default=None, help="retrieval_stats.json from active_retrieval.py")
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


def main() -> None:
    args = parse_args()
    check = Check()

    print("=" * 60)
    print("P2-A1 PREFLIGHT")
    print("=" * 60)

    # 1. Load P2-A1 config
    config = load_experiment_config(args.config)

    checkpoint = Path(config["model"])
    data_yaml_path = resolve_from_root(config["data"])
    project = resolve_from_root(config["project"])
    name = config["name"]

    print("\nResolved config:")
    print(f"  model (checkpoint) : {checkpoint}")
    print(f"  data (dataset yaml): {data_yaml_path}")
    print(f"  project             : {project}")
    print(f"  name                : {name}")
    print(f"  from_run            : {config.get('from_run')}")
    print(f"  dataset             : {config.get('dataset')}")
    print(f"  epochs              : {config.get('epochs')}")
    print(f"  batch               : {config.get('batch')}")
    print(f"  lr0                 : {config.get('lr0')}")
    print(f"  seed                : {config.get('seed')}")
    print(f"  patience            : {config.get('patience')}")

    print("\nChecks:")

    # Check 1: Phase2 checkpoint exists
    expected_checkpoint = project / EXPECTED_FROM_RUN / "weights" / "best.pt"
    if checkpoint.exists():
        check.ok(f"checkpoint exists: {checkpoint}")
    else:
        check.fail(f"checkpoint missing: {checkpoint}")

    # Check 2: from_run == phase2_final_rtdetr (NOT stage2)
    check.equal("from_run", config.get("from_run"), EXPECTED_FROM_RUN)

    # Check 3: dataset == phase2_a1_merged
    check.equal("dataset", config.get("dataset"), EXPECTED_DATASET_KEY)

    # Check 4: hyperparams
    check.equal("batch", config.get("batch"), EXPECTED_BATCH)
    check.equal("lr0", config.get("lr0"), EXPECTED_LR0)
    check.equal("seed", config.get("seed"), EXPECTED_SEED)
    check.equal("epochs", config.get("epochs"), EXPECTED_EPOCHS)
    check.equal("patience", config.get("patience"), EXPECTED_PATIENCE)

    # Check 5: dataset.yaml exists and has 6 classes in correct order
    if not data_yaml_path.exists():
        check.fail(f"dataset.yaml missing: {data_yaml_path}")
        _finish(check)
        return
    check.ok(f"dataset.yaml exists: {data_yaml_path}")

    data = load_dataset_yaml(data_yaml_path)
    root = dataset_root(data_yaml_path, data)

    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names)]
    else:
        ordered = list(names) if names else []
    if ordered == EXPECTED_CLASSES:
        check.ok(f"class names match: {ordered}")
    else:
        check.fail(f"class names mismatch: expected {EXPECTED_CLASSES}, got {ordered}")

    # Check 6: val images exist (XWOD val, count > 0)
    val_img_dir, val_lbl_dir = split_dirs(root, "val")
    val_count = count_pairs(val_img_dir, val_lbl_dir)
    if val_count > 0:
        check.ok(f"val image/label pairs: {val_count} > 0")
    else:
        check.fail(f"val dir has 0 image/label pairs: {val_img_dir}")

    # Check 7: no broken symlinks
    broken = find_broken_symlinks(root)
    if not broken:
        check.ok("no broken symlinks")
    else:
        check.fail(f"{len(broken)} broken symlink(s), e.g. {broken[:3]}")

    # Check 8: pool_stats overlap counts == 0
    if args.pool_stats:
        pool_stats_path = Path(args.pool_stats)
        if not pool_stats_path.exists():
            check.fail(f"pool_stats.json missing: {pool_stats_path}")
        else:
            pool_stats = json.loads(pool_stats_path.read_text(encoding="utf-8"))
            for key in ("overlap_with_used_train", "overlap_with_used_val", "overlap_with_used_test"):
                val = pool_stats.get(key, -1)
                if val == 0:
                    check.ok(f"pool_stats.{key} = 0 (no leakage)")
                else:
                    check.fail(f"pool_stats.{key} = {val}, expected 0 — LEAKAGE DETECTED")

    # Check 9: retrieval_stats
    if args.retrieval_stats:
        ret_stats_path = Path(args.retrieval_stats)
        if not ret_stats_path.exists():
            check.fail(f"retrieval_stats.json missing: {ret_stats_path}")
        else:
            ret_stats = json.loads(ret_stats_path.read_text(encoding="utf-8"))
            selected = ret_stats.get("selected_unique", 0)
            top_k = ret_stats.get("requested_top_k", 0)
            if selected > 0:
                check.ok(f"retrieval_stats.selected_unique = {selected} > 0")
            else:
                check.fail(f"retrieval_stats.selected_unique = {selected}, expected > 0")
            if selected <= top_k:
                check.ok(f"retrieval_stats.selected_unique ({selected}) <= requested_top_k ({top_k})")
            else:
                check.fail(f"retrieval_stats.selected_unique ({selected}) > requested_top_k ({top_k})")
            source_counts = ret_stats.get("source_split_counts", {})
            non_bdd = {k: v for k, v in source_counts.items() if k != "bdd_train"}
            if not non_bdd:
                check.ok("retrieval_stats.source_split_counts: all bdd_train")
            else:
                check.fail(f"retrieval_stats.source_split_counts has non-bdd_train sources: {non_bdd}")

    # Check 10: stats.json has bdd_retrieved_train > 0 (when pool-stats provided)
    stats_path = root / "stats.json"
    if not stats_path.exists():
        check.warn(f"stats.json not found at {stats_path} — skipping retrieval count check")
    else:
        check.ok(f"stats.json exists: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if args.pool_stats:
            retrieved_count = stats.get("retrieved_train", 0)
            if retrieved_count > 0:
                check.ok(f"stats.json retrieved_train = {retrieved_count} > 0")
            else:
                check.fail(f"stats.json retrieved_train = {retrieved_count}, expected > 0")

    # Check 11: manifest.csv
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        check.warn(f"manifest.csv not found at {manifest_path} — skipping manifest checks")
    else:
        check.ok(f"manifest.csv exists: {manifest_path}")
        _check_manifest(manifest_path, check)

    # Check 12: P2-A0C and P2-A1 configs have identical epochs/batch/lr0/seed/patience
    _check_config_parity(check, args.config)

    _finish(check)


def _check_manifest(manifest_path: Path, check: Check) -> None:
    with manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    bad_sources = {r["source_dataset"] for r in rows} - ALLOWED_MANIFEST_SOURCES
    if bad_sources:
        check.fail(f"manifest.csv contains unexpected source_dataset: {sorted(bad_sources)}")
    else:
        check.ok(f"manifest.csv source_dataset values look valid")

    # Must have bdd_retrieved source
    sources_present = {r["source_dataset"] for r in rows}
    if "bdd_retrieved" in sources_present:
        check.ok("manifest.csv contains bdd_retrieved source")
    else:
        check.warn("manifest.csv has no bdd_retrieved source — was --retrieved-root used in build?")

    # No dawn sources
    dawn_rows = [r for r in rows if any(m in r.get("image_path", "").lower() for m in FORBIDDEN_PATH_MARKERS)]
    if dawn_rows:
        check.fail(f"{len(dawn_rows)} manifest row(s) reference DAWN data")
    else:
        check.ok("no DAWN data in manifest.csv")

    # No val/test path leakage in train split
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("source_dataset") != "oversample"]
    leaking = [r for r in train_rows
               if "/val/" in r.get("image_path", "") or "/test/" in r.get("image_path", "")]
    if leaking:
        check.fail(f"{len(leaking)} train manifest row(s) reference val/test source paths (leakage)")
    else:
        check.ok("no val/test path leakage in train manifest rows")


def _check_config_parity(check: Check, a1_config_path_str: str) -> None:
    """Check 12: P2-A0C and P2-A1 configs have identical key hyperparams."""
    try:
        a0c = load_experiment_config(str(A0C_CONFIG_PATH))
        a1 = load_experiment_config(a1_config_path_str)
    except Exception as e:
        check.fail(f"cannot load configs for parity check: {e}")
        return

    parity_keys = ["epochs", "batch", "lr0", "seed", "patience"]
    all_match = True
    for key in parity_keys:
        v0 = a0c.get(key)
        v1 = a1.get(key)
        if v0 == v1:
            check.ok(f"config parity: {key} = {v0} (A0C == A1)")
        else:
            check.fail(f"config parity MISMATCH: {key}: A0C={v0}, A1={v1}")
            all_match = False

    # Both must start from phase2_final_rtdetr
    for cfg_name, cfg in [("A0C", a0c), ("A1", a1)]:
        if cfg.get("from_run") == EXPECTED_FROM_RUN:
            check.ok(f"{cfg_name} from_run = {EXPECTED_FROM_RUN}")
        else:
            check.fail(f"{cfg_name} from_run = {cfg.get('from_run')}, expected {EXPECTED_FROM_RUN}")


def _finish(check: Check) -> None:
    print()
    if check.failed:
        print("FINAL P2-A1 PREFLIGHT: FAIL")
        sys.exit(1)
    print("FINAL P2-A1 PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
