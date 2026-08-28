"""Tests for build_phase2_dataset.py with --retrieved-root option.

Verifies:
- Without --retrieved-root: output identical to baseline
- With --retrieved-root: retrieved samples added with bdd_retrieved prefix/source
- Val remains XWOD val only
- stats.json has new retrieved_train field when --retrieved-root provided
- Existing stats fields still present when --retrieved-root used
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_phase2_dataset.py"


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str, classes: list[int]) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"fake-image-bytes")
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def _make_source(root: Path, splits: dict[str, list[list[int]]]) -> None:
    for split, images in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        for i, classes in enumerate(images):
            _write_pair(img_dir, lbl_dir, f"{split}_{i:03d}", classes)


def _make_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    xwod = tmp_path / "xwod"
    acdc = tmp_path / "acdc"
    bdd = tmp_path / "bdd"

    _make_source(xwod, {
        "train": [[0], [2], [1]],
        "val": [[0], [2]],
        "test": [[5]],
    })
    _make_source(acdc, {
        "train": [[2]],
        "val": [[9999]],
    })
    _make_source(bdd, {
        "train": [[2]] * 4,
        "val": [[5]],
    })
    return xwod, acdc, bdd


def _make_retrieved(tmp_path: Path, n: int = 3) -> Path:
    retrieved = tmp_path / "retrieved"
    for i in range(n):
        _write_pair(
            retrieved / "images" / "train",
            retrieved / "labels" / "train",
            f"ret_{i:03d}", [1]  # bicycle
        )
    return retrieved


def _run_build(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), *args],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )


# ── Test G: without --retrieved-root, output identical to baseline ─────────────

def test_without_retrieved_root_baseline_unchanged(tmp_path):
    """Without --retrieved-root, stats.json must be identical to the no-retrieval baseline."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())
    # Core baseline fields must be present
    assert stats["bdd_mode"] == "full"
    assert stats["images_per_source"]["xwod_train"] == 3
    assert stats["images_per_source"]["acdc_train"] == 1
    assert stats["images_per_source"]["bdd_train"] == 4
    # retrieved fields should be zero when not used
    assert stats.get("retrieved_train", 0) == 0
    assert stats["base_train_total_before_retrieval"] == 3 + 1 + 4


def test_without_retrieved_root_no_retrieved_bdd_prefix(tmp_path):
    """Without --retrieved-root, no retrieved_bdd_ prefixed files should exist."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    train_files = [p.name for p in (out / "images" / "train").iterdir()]
    assert not any(n.startswith("retrieved_bdd_") for n in train_files), (
        f"Found retrieved_bdd_ files without --retrieved-root: {[n for n in train_files if n.startswith('retrieved_bdd_')]}"
    )


# ── Test H: with --retrieved-root, retrieved samples are added ────────────────

def test_with_retrieved_root_adds_samples(tmp_path):
    """With --retrieved-root, bdd_retrieved images are added to the merged train."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=3)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())
    assert stats["retrieved_train"] == 3, f"Expected 3 retrieved, got {stats['retrieved_train']}"
    assert stats["images_per_source"]["bdd_retrieved_train"] == 3


# ── Test: retrieved samples get prefix retrieved_bdd_ ─────────────────────────

def test_retrieved_prefix(tmp_path):
    """Retrieved images must be prefixed with retrieved_bdd_ in the output."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=2)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    train_files = [p.name for p in (out / "images" / "train").iterdir()]
    retrieved_files = [n for n in train_files if n.startswith("retrieved_bdd_")]
    assert len(retrieved_files) == 2, f"Expected 2 retrieved_bdd_ files, got {retrieved_files}"


# ── Test I: val remains XWOD val only ─────────────────────────────────────────

def test_val_remains_xwod_only_with_retrieved(tmp_path):
    """Adding --retrieved-root must not change the val split."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=2)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    val_files = sorted(p.name for p in (out / "images" / "val").iterdir())
    assert all(n.startswith("xwod_") for n in val_files), (
        f"Non-XWOD files in val: {[n for n in val_files if not n.startswith('xwod_')]}"
    )
    assert not any(n.startswith("retrieved_bdd_") for n in val_files)


# ── Test: stats.json has retrieved_train field when --retrieved-root provided ─

def test_stats_has_retrieved_train_field(tmp_path):
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=4)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())
    assert "retrieved_train" in stats
    assert "base_train_total_before_retrieval" in stats
    assert "train_total_with_retrieval_before_oversampling" in stats
    assert "retrieved_class_counts" in stats
    assert stats["retrieved_train"] == 4
    base = stats["base_train_total_before_retrieval"]  # xwod+acdc+bdd = 3+1+4
    assert stats["train_total_with_retrieval_before_oversampling"] == base + 4


# ── Test: existing stats fields still present with --retrieved-root ────────────

def test_existing_stats_fields_preserved_with_retrieved(tmp_path):
    """--retrieved-root must not remove any existing stats fields."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=2)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())
    expected_keys = [
        "seed", "mode", "bdd_mode",
        "bdd_replay_ratio", "bdd_replay_images_requested", "bdd_replay_images_actual",
        "images_per_source", "oversample_rare",
        "train_total", "val_total",
        "train_boxes_per_class_before", "train_boxes_per_class_after", "val_boxes_per_class",
    ]
    for key in expected_keys:
        assert key in stats, f"Existing stats key missing: {key}"


# ── Test: manifest.csv has bdd_retrieved source ───────────────────────────────

def test_manifest_has_bdd_retrieved_source(tmp_path):
    xwod, acdc, bdd = _make_sources(tmp_path)
    retrieved = _make_retrieved(tmp_path, n=2)
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy", "--clean",
    ])

    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    sources = {r["source_dataset"] for r in rows}
    assert "bdd_retrieved" in sources, f"bdd_retrieved not in manifest sources: {sources}"
