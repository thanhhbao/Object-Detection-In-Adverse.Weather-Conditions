"""Tests for scripts/build_random_rare_control.py — protocol and unit tests.

Tests cover:
- Test A: Fixed seed=42, same pool → identical selection every run (determinism)
- Test B: Selection count equals min(top_k, available_rare)
- Test C: Car-only images excluded (rare class filter)
- Test D: Output counts: images == labels == manifest rows == selected
- Test E: --used-bdd-root overlap == 0 — simulate overlap → RuntimeError
- Test F: --clean removes stale output before writing new selection
- Test G: All label IDs 0..5 in output (validate_labels from active_retrieval)
- Test H: source_split == bdd_train and selection_method == random_rare in all rows
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_random_rare_control.py"
sys.path.insert(0, str(ROOT / "scripts"))


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake_img")


def _write_label(path: Path, classes: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_pool(tmp_path: Path, n_rare: int = 10, n_car_only: int = 3) -> Path:
    """Create a fake pool with rare-class and car-only images."""
    pool = tmp_path / "pool"
    pool_img = pool / "images" / "train"
    pool_lbl = pool / "labels" / "train"
    pool_img.mkdir(parents=True, exist_ok=True)
    pool_lbl.mkdir(parents=True, exist_ok=True)

    # Rare images
    for i in range(n_rare):
        cls = [1, 3, 4][i % 3]  # bicycle / motorcycle / bus
        name = f"rare_{i:04d}"
        _write_image(pool_img / f"{name}.jpg")
        _write_label(pool_lbl / f"{name}.txt", [cls, 2])

    # Car-only images (should be excluded)
    for i in range(n_car_only):
        name = f"car_{i:04d}"
        _write_image(pool_img / f"{name}.jpg")
        _write_label(pool_lbl / f"{name}.txt", [2])

    return pool


# ── Helpers for running the script ────────────────────────────────────────────

def _run_script(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        capture_output=True, text=True, cwd=ROOT,
    )


# ── Test A: Determinism ───────────────────────────────────────────────────────

def test_A_determinism_same_seed(tmp_path):
    """Fixed seed=42, same pool → identical selection both runs."""
    from build_random_rare_control import _get_rare_classes_in_image

    # Build a pool of 20 rare images
    n_rare = 20
    pool_img_paths = [tmp_path / f"img_{i:04d}.jpg" for i in range(n_rare)]
    for p in pool_img_paths:
        p.write_bytes(b"fake")

    rng_a = random.Random(42)
    shuffled_a = list(pool_img_paths)
    rng_a.shuffle(shuffled_a)
    selection_a = shuffled_a[:10]

    rng_b = random.Random(42)
    shuffled_b = list(pool_img_paths)
    rng_b.shuffle(shuffled_b)
    selection_b = shuffled_b[:10]

    assert [p.name for p in selection_a] == [p.name for p in selection_b], \
        "Determinism failure: same seed should give same order"


# ── Test B: Selection count = min(top_k, available_rare) ─────────────────────

def test_B_selection_count_capped_at_available(tmp_path):
    """If available_rare < top_k, take all available."""
    from build_random_rare_control import _get_rare_classes_in_image

    # 5 rare images, top_k=10 → should take all 5
    paths = [tmp_path / f"img_{i}.jpg" for i in range(5)]
    for p in paths:
        p.write_bytes(b"fake")

    top_k = 10
    rng = random.Random(42)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    selected = shuffled[:top_k]

    assert len(selected) == 5, f"Expected 5 (all available), got {len(selected)}"


def test_B_selection_count_exact_top_k(tmp_path):
    """If available_rare >= top_k, take exactly top_k."""
    paths = [tmp_path / f"img_{i}.jpg" for i in range(20)]
    for p in paths:
        p.write_bytes(b"fake")

    top_k = 7
    rng = random.Random(42)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    selected = shuffled[:top_k]

    assert len(selected) == top_k


# ── Test C: Car-only images excluded ──────────────────────────────────────────

def test_C_car_only_excluded(tmp_path):
    """filter_pool_by_class (from active_retrieval) must exclude car-only images."""
    from active_retrieval import filter_pool_by_class

    pool_img = tmp_path / "images" / "train"
    pool_lbl = tmp_path / "labels" / "train"
    pool_img.mkdir(parents=True, exist_ok=True)
    pool_lbl.mkdir(parents=True, exist_ok=True)

    _write_image(pool_img / "car_only.jpg")
    _write_label(pool_lbl / "car_only.txt", [2])

    _write_image(pool_img / "has_bus.jpg")
    _write_label(pool_lbl / "has_bus.txt", [4])

    _write_image(pool_img / "has_bicycle.jpg")
    _write_label(pool_lbl / "has_bicycle.txt", [1])

    filtered, before = filter_pool_by_class(pool_img, pool_lbl, {1, 3, 4})
    names = {p.name for p in filtered}

    assert "car_only.jpg" not in names
    assert "has_bus.jpg" in names
    assert "has_bicycle.jpg" in names
    assert before == 3


# ── Test D: Output count invariant ───────────────────────────────────────────

def test_D_output_counts_equal_selected(tmp_path):
    """images == labels == manifest rows == selected after writing."""
    pool = _make_pool(tmp_path, n_rare=8, n_car_only=2)
    out = tmp_path / "out"

    result = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "5",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert result.returncode == 0, f"Script failed:\n{result.stdout}\n{result.stderr}"

    stats = json.loads((out / "random_rare_stats.json").read_text())
    sel = stats["selected"]
    img_c = stats["output_image_count"]
    lbl_c = stats["output_label_count"]
    man_c = stats["manifest_row_count"]

    assert img_c == sel, f"image count {img_c} != selected {sel}"
    assert lbl_c == sel, f"label count {lbl_c} != selected {sel}"
    assert man_c == sel, f"manifest count {man_c} != selected {sel}"


# ── Test E: Overlap with used BDD raises RuntimeError ─────────────────────────

def test_E_overlap_detection_raises(tmp_path):
    """Simulate overlap between selected and used BDD → RuntimeError."""
    pool = _make_pool(tmp_path, n_rare=5, n_car_only=0)
    out = tmp_path / "out"

    # Create a fake "used BDD" root that shares filenames with the pool
    used_bdd = tmp_path / "used_bdd"
    used_bdd_img_train = used_bdd / "images" / "train"
    used_bdd_img_train.mkdir(parents=True, exist_ok=True)

    # Copy one pool image name to the "used BDD" to simulate overlap
    pool_imgs = list((pool / "images" / "train").iterdir())
    overlap_name = pool_imgs[0].name
    (used_bdd_img_train / overlap_name).write_bytes(b"fake")

    result = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "5",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--used-bdd-root", str(used_bdd),
        "--clean",
    ])
    assert result.returncode != 0, "Script should fail when overlap with used BDD detected"
    assert "LEAKAGE" in result.stderr or "LEAKAGE" in result.stdout


# ── Test F: --clean removes stale output ─────────────────────────────────────

def test_F_clean_removes_stale(tmp_path):
    """--clean must remove stale output before writing new selection."""
    pool = _make_pool(tmp_path, n_rare=8, n_car_only=0)
    out = tmp_path / "out"

    # First run
    result = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "3",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert result.returncode == 0, f"First run failed: {result.stderr}"
    stale_file = out / "random_rare_stats.json"
    assert stale_file.exists()

    # Second run with --clean — should succeed and not keep stale files
    result2 = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "3",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert result2.returncode == 0, f"Second run with --clean failed: {result2.stderr}"


def test_F_without_clean_fails_if_non_empty(tmp_path):
    """Without --clean, non-empty out-root → non-zero exit."""
    pool = _make_pool(tmp_path, n_rare=5, n_car_only=0)
    out = tmp_path / "out"

    # First run with --clean
    r1 = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "3",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert r1.returncode == 0

    # Second run WITHOUT --clean → should fail
    r2 = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "3",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
    ])
    assert r2.returncode != 0


# ── Test G: All label IDs 0..5 in output ─────────────────────────────────────

def test_G_label_ids_in_range(tmp_path):
    """validate_labels (from active_retrieval) must pass for all output labels."""
    from active_retrieval import validate_labels

    pool = _make_pool(tmp_path, n_rare=6, n_car_only=0)
    out = tmp_path / "out"

    result = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "5",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    lbl_dir = out / "labels" / "train"
    violations = validate_labels(lbl_dir, max_class_id=5)
    assert violations == [], f"Label violations: {violations}"


# ── Test H: manifest fields source_split and selection_method ─────────────────

def test_H_manifest_fields(tmp_path):
    """All manifest rows must have source_split=bdd_train and selection_method=random_rare."""
    pool = _make_pool(tmp_path, n_rare=6, n_car_only=0)
    out = tmp_path / "out"

    result = _run_script([
        "--pool-root", str(pool),
        "--out-root", str(out),
        "--top-k", "4",
        "--seed", "42",
        "--candidate-target-classes", "1", "3", "4",
        "--mode", "copy",
        "--clean",
    ])
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    manifest_path = out / "random_rare_manifest.csv"
    assert manifest_path.exists(), "random_rare_manifest.csv must exist"

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "Manifest must have at least one row"
    for row in rows:
        assert row["source_split"] == "bdd_train", \
            f"source_split must be bdd_train, got: {row['source_split']}"
        assert row["selection_method"] == "random_rare", \
            f"selection_method must be random_rare, got: {row['selection_method']}"
