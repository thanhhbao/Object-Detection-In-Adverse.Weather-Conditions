"""Synthetic-dataset tests for scripts/build_phase2_dataset.py.

Uses tiny fake YOLO-format dataset roots (a few images/labels each) so the merge
logic can be verified without any real dataset or GPU:
  - XWOD train (all) + ACDC train (all) + BDD train replay only
  - validation = XWOD val only
  - test splits of every source are never read
  - rare-class oversampling (bicycle x2, motorcycle x3, bus x3) touches train only
  - class IDs are copied verbatim (no remapping)
  - BDD replay selection is deterministic for a fixed seed
"""

from __future__ import annotations

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


def _run_build(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), *args],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )


def _build(tmp_path: Path, out_name: str = "merged", *, replay: int = 3) -> tuple[Path, dict]:
    xwod, acdc, bdd = tmp_path / "xwod", tmp_path / "acdc", tmp_path / "bdd"
    out = tmp_path / out_name

    _make_source(xwod, {
        "train": [[0], [2], [1], [3], [4]],  # person, car, bicycle, motorcycle, bus
        "val": [[0], [2]],
        "test": [[5], [5]],  # must never be pulled into the merge
    })
    _make_source(acdc, {
        "train": [[2]],  # car only — no rare classes here
        "val": [[9999]],  # deliberately invalid marker; must never appear in output
        "test": [[5]],
    })
    _make_source(bdd, {
        "train": [[2]] * 10,
        "val": [[5]],
        "test": [[5]],
    })

    _run_build([
        "--xwod-root", str(xwod), "--acdc-root", str(acdc), "--bdd-root", str(bdd),
        "--out-root", str(out), "--bdd-replay-images", str(replay),
        "--seed", "42", "--mode", "copy", "--oversample-rare", "--clean",
    ])

    stats = json.loads((out / "stats.json").read_text())
    return out, stats


def test_sources_merged_correctly(tmp_path):
    out, stats = _build(tmp_path)

    assert stats["images_per_source"]["xwod_train"] == 5
    assert stats["images_per_source"]["acdc_train"] == 1
    assert stats["images_per_source"]["bdd_replay"] == 3
    assert stats["bdd_replay_images_requested"] == 3
    assert stats["bdd_replay_images_actual"] == 3


def test_validation_is_xwod_only(tmp_path):
    out, stats = _build(tmp_path)

    assert stats["val_total"] == 2
    val_images = sorted(p.name for p in (out / "images" / "val").iterdir())
    assert val_images == ["xwod_val_000.jpg", "xwod_val_001.jpg"]
    # ACDC/BDD val must never leak in, even though those splits exist on disk.
    assert not any(name.startswith(("acdc_", "bdd_")) for name in val_images)


def test_test_splits_never_leak_into_merge(tmp_path):
    out, _ = _build(tmp_path)

    all_names = [p.name for p in (out / "images" / "train").rglob("*")]
    all_names += [p.name for p in (out / "images" / "val").rglob("*")]
    # class 5 (truck) only appears in *_test splits in this fixture — it must be absent.
    for lbl in (out / "labels" / "train").glob("*.txt"):
        assert "5" not in lbl.read_text().split()
    for lbl in (out / "labels" / "val").glob("*.txt"):
        content = lbl.read_text().split()
        assert "9999" not in content  # the ACDC val marker must never appear


def test_rare_class_oversampling_train_only(tmp_path):
    out, stats = _build(tmp_path)

    # bicycle(x2) + motorcycle(x3) + bus(x3) images => (2-1)+(3-1)+(3-1) = 5 duplicates
    assert stats["oversample_rare"]["duplicate_images"] == 5
    assert stats["train_total"] == stats["oversample_rare"]["base_train_images"] + 5

    train_dup_files = list((out / "images" / "train").glob("*_os*"))
    val_dup_files = list((out / "images" / "val").glob("*_os*"))
    assert len(train_dup_files) == 5
    assert val_dup_files == []


def test_class_ids_copied_verbatim(tmp_path):
    out, _ = _build(tmp_path)

    copied = (out / "labels" / "train" / "xwod_train_002.txt").read_text()
    assert copied.strip() == "1 0.5 0.5 0.1 0.1"  # bicycle, unchanged id


def test_bdd_replay_is_deterministic(tmp_path):
    out_a, stats_a = _build(tmp_path, out_name="merged_a")
    out_b, stats_b = _build(tmp_path, out_name="merged_b")

    def bdd_sources(out: Path) -> set[str]:
        import csv
        rows = csv.DictReader((out / "manifest.csv").open(encoding="utf-8"))
        return {r["image_path"] for r in rows if r["source_dataset"] == "bdd"}

    assert bdd_sources(out_a) == bdd_sources(out_b)
    assert stats_a["images_per_source"]["bdd_replay"] == stats_b["images_per_source"]["bdd_replay"]
