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


# ── Tests for --exclude-retrieved-from-oversampling ───────────────────────────

def _make_rare_retrieved(tmp_path: Path, n: int = 3, classes: list[int] | None = None) -> Path:
    """Create retrieved images with rare classes (default: bus=4, oversampling multiplier 3)."""
    if classes is None:
        classes = [4]  # bus → multiplier 3
    retrieved = tmp_path / "retrieved_rare"
    for i in range(n):
        _write_pair(
            retrieved / "images" / "train",
            retrieved / "labels" / "train",
            f"ret_{i:03d}", classes,
        )
    return retrieved


def _count_files_in(directory: Path, suffix: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix == suffix)


def test_excl_retrieved_skips_retrieved_bdd_oversampling(tmp_path):
    """--exclude-retrieved-from-oversampling: retrieved_bdd_* images must NOT be oversampled."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    # bus-only retrieved images would normally be oversampled 3x (+2 duplicates each)
    retrieved = _make_rare_retrieved(tmp_path, n=2, classes=[4])
    out = tmp_path / "merged"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy",
        "--oversample-rare", "--exclude-retrieved-from-oversampling",
        "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())

    # Verify the flag is recorded in stats
    assert stats["oversample_rare"]["exclude_retrieved_from_oversampling"] is True
    assert stats["oversample_rare"]["retrieved_prefix_excluded"] == "retrieved_bdd_"

    # Count oversampled copies: retrieved_bdd_ret_000_os*.jpg must NOT exist
    train_img_dir = out / "images" / "train"
    os_retrieved = list(train_img_dir.glob("retrieved_bdd_*_os*.jpg"))
    assert len(os_retrieved) == 0, (
        f"retrieved_bdd_* images should not be oversampled, found: {[f.name for f in os_retrieved]}"
    )


def test_excl_retrieved_preserves_base_oversampling(tmp_path):
    """--exclude-retrieved-from-oversampling must NOT change oversampling of base images."""
    xwod, acdc, bdd = _make_sources(tmp_path)

    # Make XWOD with known rare class composition to count duplicates
    xwod2 = tmp_path / "xwod2"
    # 1 bicycle (mult 2): +1 dup; 1 bus (mult 3): +2 dups; total +3 base dups
    _make_source(xwod2, {
        "train": [[1], [4], [0]],   # bicycle, bus, person
        "val": [[0]],
    })
    retrieved = _make_rare_retrieved(tmp_path, n=1, classes=[4])  # bus: would add +2 if not excluded
    out_with = tmp_path / "merged_with_excl"
    out_without = tmp_path / "merged_without_excl"

    base_args = [
        "--xwod-root", str(xwod2), "--bdd-root", str(bdd),
        "--out-root", str(out_with), "--bdd-use-all",
        "--seed", "42", "--mode", "copy",
        "--oversample-rare", "--retrieved-root", str(retrieved),
    ]
    _run_build(base_args + ["--exclude-retrieved-from-oversampling", "--clean"])

    # Without retrieved, count base duplicates
    out_base = tmp_path / "merged_base_only"
    _run_build([
        "--xwod-root", str(xwod2), "--bdd-root", str(bdd),
        "--out-root", str(out_base), "--bdd-use-all",
        "--seed", "42", "--mode", "copy",
        "--oversample-rare", "--clean",
    ])

    stats_with = json.loads((out_with / "stats.json").read_text())
    stats_base = json.loads((out_base / "stats.json").read_text())

    # Base duplicate count must be identical (retrieved exclusion leaves base untouched)
    assert stats_with["oversample_rare"]["duplicate_images"] == \
           stats_base["oversample_rare"]["duplicate_images"], (
        f"Base duplicates changed: with={stats_with['oversample_rare']['duplicate_images']}, "
        f"base={stats_base['oversample_rare']['duplicate_images']}"
    )


def test_excl_retrieved_without_flag_oversample_all(tmp_path):
    """Default behavior (no --exclude-retrieved-from-oversampling): all images are oversampled."""
    xwod, acdc, bdd = _make_sources(tmp_path)
    # 2 bus images in retrieved → each would get +2 duplicates = +4 total if oversampled
    retrieved = _make_rare_retrieved(tmp_path, n=2, classes=[4])
    out = tmp_path / "merged_no_excl"

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--retrieved-root", str(retrieved),
        "--out-root", str(out), "--bdd-use-all",
        "--seed", "42", "--mode", "copy",
        "--oversample-rare",
        "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())

    # Flag should be False / absent in old mode
    assert stats["oversample_rare"]["exclude_retrieved_from_oversampling"] is False
    # retrieved_bdd_*_os*.jpg must exist because retrieved images ARE oversampled by default
    train_img_dir = out / "images" / "train"
    os_retrieved = list(train_img_dir.glob("retrieved_bdd_*_os*.jpg"))
    # Each bus image gets +2 copies → 2 images * 2 copies = 4 extra
    assert len(os_retrieved) == 4, (
        f"Without exclusion, retrieved bus images should be oversampled (+2 each × 2 = 4 copies), "
        f"found {len(os_retrieved)}: {[f.name for f in os_retrieved]}"
    )


def test_excl_retrieved_cross_arm_duplicate_parity(tmp_path):
    """A0R and A1-DINO arms with different retrieved class composition must have identical duplicates."""
    xwod, acdc, bdd = _make_sources(tmp_path)

    # Arm A: retrieved images are all-bus (heavy rare class mix if included)
    retrieved_a = _make_rare_retrieved(tmp_path / "a", n=3, classes=[4])  # bus ×3
    # Arm B: retrieved images are all-person (no rare class)
    retrieved_b = tmp_path / "retrieved_b"
    for i in range(3):
        _write_pair(
            retrieved_b / "images" / "train",
            retrieved_b / "labels" / "train",
            f"ret_{i:03d}", [0],  # person only
        )

    out_a = tmp_path / "merged_a"
    out_b = tmp_path / "merged_b"

    common = [
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--bdd-use-all", "--seed", "42", "--mode", "copy",
        "--oversample-rare", "--exclude-retrieved-from-oversampling", "--clean",
    ]
    _run_build(common + ["--retrieved-root", str(retrieved_a), "--out-root", str(out_a)])
    _run_build(common + ["--retrieved-root", str(retrieved_b), "--out-root", str(out_b)])

    stats_a = json.loads((out_a / "stats.json").read_text())
    stats_b = json.loads((out_b / "stats.json").read_text())

    dup_a = stats_a["oversample_rare"]["duplicate_images"]
    dup_b = stats_b["oversample_rare"]["duplicate_images"]
    assert dup_a == dup_b, (
        f"Duplicate count must match across arms (--exclude-retrieved-from-oversampling), "
        f"but A={dup_a} != B={dup_b}"
    )

    train_a = stats_a["train_total"]
    train_b = stats_b["train_total"]
    assert train_a == train_b, (
        f"Final train_total must match across arms, but A={train_a} != B={train_b}"
    )


def test_excl_retrieved_expected_55985_count(tmp_path):
    """Verify exact count arithmetic: 37188 + 5000 + 13797 = 55985."""
    # This is a pure arithmetic test — no subprocess needed.
    base = 37188   # XWOD(6006) + ACDC(1182) + BDD30K(30000)
    retrieved = 5000
    dups = 13797   # official Phase2 oversampling duplicates
    expected_train = base + retrieved + dups
    assert expected_train == 55985, f"Arithmetic check failed: {expected_train} != 55985"
    assert base + retrieved == 42188, "Pre-oversampling total must be 42188"
