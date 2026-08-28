"""Tests for scripts/build_bdd_retrieval_pool.py.

Uses tiny fake datasets (b"fake" images, YOLO .txt labels) — no GPU, no real BDD data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_bdd_retrieval_pool.py"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str, classes: list[int] | None = None) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"fake")
    if classes is None:
        classes = [2]
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def _make_full_root(root: Path, names: list[str]) -> Path:
    """full_root with images/train only (BDD full train split)."""
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    for name in names:
        stem = Path(name).stem if "." in name else name
        _write_pair(img_dir, lbl_dir, stem)
    return root


def _make_used_root(root: Path,
                    train: list[str] | None = None,
                    val: list[str] | None = None,
                    test: list[str] | None = None) -> Path:
    """used_root with some of train/val/test splits."""
    for split, names in [("train", train or []), ("val", val or []), ("test", test or [])]:
        if names:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            for name in names:
                stem = Path(name).stem if "." in name else name
                _write_pair(img_dir, lbl_dir, stem)
    return root


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=check, cwd=ROOT, capture_output=True, text=True,
    )


# ── Test A: only novel images are candidates ───────────────────────────────────

def test_only_novel_images_are_candidates(tmp_path):
    """a.jpg in used_train, b.jpg in used_val, only c.jpg should be in pool."""
    full = _make_full_root(tmp_path / "full", ["a", "b", "c"])
    used = _make_used_root(tmp_path / "used", train=["a"], val=["b"])
    out = tmp_path / "pool"

    _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ])

    pool_images = sorted((out / "images" / "train").iterdir())
    names = [p.name for p in pool_images if p.suffix == ".jpg"]
    assert names == ["c.jpg"], f"Expected only c.jpg in pool, got {names}"

    stats = json.loads((out / "pool_stats.json").read_text())
    assert stats["candidate_pool"] == 1
    assert stats["excluded_existing_names"] == 2


# ── Test B: pool never takes from full_root val or test ───────────────────────

def test_pool_only_from_full_train(tmp_path):
    """The pool script only reads full_root/images/train, not val/test."""
    full = tmp_path / "full"
    # Write extra splits in full_root — they must be ignored
    for split in ("val", "test"):
        _write_pair(full / "images" / split, full / "labels" / split, f"should_not_appear_{split}")
    _make_full_root(full, ["train_img"])
    used = _make_used_root(tmp_path / "used")
    out = tmp_path / "pool"

    _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ])

    pool_images = [p.name for p in (out / "images" / "train").iterdir() if p.suffix == ".jpg"]
    assert all("should_not_appear" not in n for n in pool_images), (
        f"val/test images leaked into pool: {pool_images}"
    )
    assert "train_img.jpg" in pool_images


# ── Test D: missing full_root/train causes RuntimeError ──────────────────────

def test_missing_full_train_dir_causes_failure(tmp_path):
    """When full_root has no images/train directory, the script must fail."""
    # Create full_root with NO train dir (only labels)
    full = tmp_path / "full"
    full.mkdir()
    # Deliberately no images/train

    used = _make_used_root(tmp_path / "used", train=["some_img"])
    out = tmp_path / "pool"

    result = _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ], check=False)

    assert result.returncode != 0, (
        f"Expected non-zero exit when full_root has no train dir.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ── Test D2: full=used overlap is filtered, pool is empty, invariant holds ────

def test_full_same_as_used_gives_empty_pool(tmp_path):
    """When full_root and used_root share all images, pool is empty, overlap=0."""
    # Same images in both full_root/train and used_root/train
    full = _make_full_root(tmp_path / "full", ["shared_a", "shared_b"])
    used = _make_used_root(tmp_path / "used", train=["shared_a", "shared_b"])

    out = tmp_path / "pool"
    result = _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ], check=False)

    # Script should succeed (overlap after filtering is 0)
    assert result.returncode == 0, (
        f"Expected success when all are excluded.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    stats = json.loads((out / "pool_stats.json").read_text())
    assert stats["candidate_pool"] == 0
    assert stats["overlap_with_used_train"] == 0  # candidates have 0 overlap


# ── Test: pool_stats.json has correct keys ───────────────────────────────────

def test_pool_stats_has_correct_keys(tmp_path):
    full = _make_full_root(tmp_path / "full", ["a", "b", "c"])
    used = _make_used_root(tmp_path / "used", train=["a"])
    out = tmp_path / "pool"

    _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "pool_stats.json").read_text())
    required_keys = [
        "full_train_pairs", "used_bdd_train", "used_bdd_val", "used_bdd_test",
        "excluded_existing_names", "candidate_pool",
        "overlap_with_used_train", "overlap_with_used_val", "overlap_with_used_test",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key in pool_stats.json: {key}"


# ── Test: overlap_with_used_train == 0 invariant ─────────────────────────────

def test_overlap_with_used_train_zero(tmp_path):
    """When there is no overlap, the script succeeds and reports 0 overlap."""
    full = _make_full_root(tmp_path / "full", ["x", "y", "z"])
    used = _make_used_root(tmp_path / "used", train=[], val=["other"])
    out = tmp_path / "pool"

    _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ])

    stats = json.loads((out / "pool_stats.json").read_text())
    assert stats["overlap_with_used_train"] == 0
    assert stats["overlap_with_used_val"] == 0
    assert stats["overlap_with_used_test"] == 0
    assert stats["candidate_pool"] == 3


# ── Test: candidate_manifest.csv fields ──────────────────────────────────────

def test_candidate_manifest_fields(tmp_path):
    full = _make_full_root(tmp_path / "full", ["img1", "img2"])
    used = _make_used_root(tmp_path / "used")
    out = tmp_path / "pool"

    _run([
        "--full-root", str(full),
        "--used-root", str(used),
        "--out-root", str(out),
        "--mode", "copy", "--clean",
    ])

    import csv
    rows = list(csv.DictReader((out / "candidate_manifest.csv").open()))
    assert len(rows) == 2
    for row in rows:
        assert row["source_split"] == "bdd_train"
        assert row["novel_vs_bdd30k"] == "1"
        assert "image_name" in row
        assert "image" in row
        assert "label" in row
