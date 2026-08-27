"""Unit tests for scripts/preflight_phase2.py — no GPU, no real dataset needed.

Builds a tiny synthetic run/dataset/config/stats/manifest quintuple per test and
asserts the script exits non-zero (and prints the expected failure line) for each
required failure mode of the OFFICIAL Phase 2 protocol, plus one happy-path pass.

Uses small synthetic counts (not the real 6006/1182/30000/1001/37188 numbers) by
monkeypatching the module's expected-value constants, so tests stay fast and do
not require the real merged dataset.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import preflight_phase2 as pf

DATASET_KEY = "phase2_merged"
FROM_RUN = "stage2_xwod_rtdetr_from_bdd30k"
SOURCE_COUNTS = {"xwod_train": 2, "acdc_train": 1, "bdd_train": 1, "xwod_val": 2}
BASE_TRAIN_IMAGES = sum(SOURCE_COUNTS[k] for k in ("xwod_train", "acdc_train", "bdd_train"))
VAL_COUNT = SOURCE_COUNTS["xwod_val"]
MULTIPLIERS = {"bicycle": 2, "motorcycle": 3, "bus": 3}


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"fake")
    (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")


def _manifest_rows(source_counts: dict, *, overlap: bool = False,
                    bad_source: bool = False, leaked_split: bool = False) -> list[dict]:
    rows: list[dict] = []

    def add(source_dataset: str, split: str, n: int, original_split: str | None = None) -> None:
        original_split = original_split or split
        for i in range(n):
            rows.append({
                "source_dataset": source_dataset, "split": split,
                "image_path": f"/fake/{source_dataset}/images/{original_split}/img{i}.jpg",
                "label_path": f"/fake/{source_dataset}/labels/{original_split}/img{i}.txt",
                "new_image": f"/merged/images/{split}/{source_dataset}_img{i}.jpg",
                "new_label": f"/merged/labels/{split}/{source_dataset}_img{i}.txt",
            })

    add("xwod", "train", source_counts["xwod_train"])
    add("acdc", "train", source_counts["acdc_train"])
    # leaked_split simulates a bug that pulled BDD *val* images into the merged train split.
    add("bdd", "train", source_counts["bdd_train"], original_split="val" if leaked_split else "train")
    add("xwod", "val", source_counts["xwod_val"])

    if overlap:
        dup = dict(rows[0])
        dup["split"] = "val"
        rows.append(dup)
    if bad_source:
        rows.append({
            "source_dataset": "dawn", "split": "train",
            "image_path": "/fake/dawn/images/train/img0.jpg", "label_path": "x",
            "new_image": "/merged/images/train/dawn_img0.jpg", "new_label": "x",
        })
    return rows


def _write_manifest(root: Path, rows: list[dict]) -> None:
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "split", "image_path", "label_path", "new_image", "new_label"])
        writer.writeheader()
        writer.writerows(rows)


def _write_stats(
    root: Path,
    *,
    source_counts: dict | None = None,
    base_train_images: int | None = None,
    duplicate_images: int = 0,
    train_total: int | None = None,
    val_total: int = VAL_COUNT,
    oversample_enabled: bool = True,
    multipliers: dict | None = MULTIPLIERS,
) -> None:
    source_counts = source_counts if source_counts is not None else SOURCE_COUNTS
    if base_train_images is None:
        base_train_images = BASE_TRAIN_IMAGES
    if train_total is None:
        train_total = base_train_images + duplicate_images

    stats = {
        "seed": 42, "mode": "symlink", "bdd_mode": "full",
        "bdd_replay_ratio": None, "bdd_replay_images_requested": None,
        "bdd_replay_images_actual": None,
        "images_per_source": source_counts,
        "oversample_rare": {
            "enabled": oversample_enabled,
            "multipliers": multipliers if oversample_enabled else {},
            "base_train_images": base_train_images,
            "duplicate_images": duplicate_images,
            "train_images_after": train_total,
            "multiplier_distribution": {},
        },
        "train_total": train_total,
        "val_total": val_total,
        "train_boxes_per_class_before": {},
        "train_boxes_per_class_after": {},
        "val_boxes_per_class": {},
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _build_env(
    tmp_path: Path,
    *,
    class_names: list[str] | None = None,
    val_count: int = VAL_COUNT,
    checkpoint: bool = True,
    dataset_key: str = DATASET_KEY,
    from_run: str = FROM_RUN,
    batch: int = 16,
    lr0: float = 0.00005,
    seed: int = 42,
    write_stats: bool = True,
    write_manifest: bool = True,
    stats_kwargs: dict | None = None,
    manifest_rows: list[dict] | None = None,
) -> tuple[Path, Path]:
    class_names = class_names or pf.EXPECTED_CLASSES
    root = tmp_path / "datasets" / "merged"
    for i in range(val_count):
        _write_pair(root / "images" / "val", root / "labels" / "val", f"val_{i}")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (root / "dataset.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/val\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / from_run / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        (run_dir / "best.pt").write_bytes(b"fake")

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        f"project: {tmp_path / 'runs'}\ndatasets:\n  {dataset_key}: {root / 'dataset.yaml'}\n",
        encoding="utf-8",
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"paths: {paths_yaml}\nfrom_run: {from_run}\ndataset: {dataset_key}\nname: phase2_final_rtdetr\n"
        # Fixed-point (not scientific) notation: PyYAML's safe_load only recognizes
        # "1.0e-05"-style exponents as float; "1e-05" (no dot) parses as a plain string.
        f"epochs: 1\nbatch: {batch}\nimgsz: 640\noptimizer: AdamW\nlr0: {lr0:.10f}\nseed: {seed}\n",
        encoding="utf-8",
    )

    if write_stats:
        _write_stats(root, **(stats_kwargs or {}))
    if write_manifest:
        _write_manifest(root, manifest_rows if manifest_rows is not None else _manifest_rows(SOURCE_COUNTS))

    return config_yaml, root


@pytest.fixture(autouse=True)
def _no_od_paths_env(monkeypatch):
    # Never let a real OD_PATHS env var override our synthetic paths.yaml.
    monkeypatch.delenv("OD_PATHS", raising=False)


@pytest.fixture(autouse=True)
def _relaxed_expectations(monkeypatch):
    # These fixtures use small synthetic counts, not the real dataset's numbers.
    monkeypatch.setattr(pf, "EXPECTED_VAL_COUNT", VAL_COUNT)
    monkeypatch.setattr(pf, "EXPECTED_SOURCE_COUNTS", SOURCE_COUNTS)
    monkeypatch.setattr(pf, "EXPECTED_BASE_TRAIN_IMAGES", BASE_TRAIN_IMAGES)


def _run(config_yaml: Path, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["preflight_phase2.py", "--config", str(config_yaml)])
    with pytest.raises(SystemExit) as exc:
        pf.main()
    return exc.value.code


def test_preflight_passes_happy_path(tmp_path, monkeypatch):
    config_yaml, _ = _build_env(tmp_path)
    monkeypatch.setattr(sys, "argv", ["preflight_phase2.py", "--config", str(config_yaml)])
    pf.main()  # must return normally (no SystemExit) when everything checks out


def test_preflight_fails_on_missing_checkpoint(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, checkpoint=False)
    assert _run(config_yaml, monkeypatch) == 1
    assert "checkpoint missing" in capsys.readouterr().out


def test_preflight_fails_on_missing_dataset(tmp_path, monkeypatch, capsys):
    config_yaml, root = _build_env(tmp_path)
    (root / "dataset.yaml").unlink()
    assert _run(config_yaml, monkeypatch) == 1
    assert "dataset.yaml missing" in capsys.readouterr().out


def test_preflight_fails_on_wrong_class_order(tmp_path, monkeypatch, capsys):
    bad_order = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    config_yaml, _ = _build_env(tmp_path, class_names=bad_order)
    assert _run(config_yaml, monkeypatch) == 1
    assert "class names mismatch" in capsys.readouterr().out


def test_preflight_fails_on_broken_symlink(tmp_path, monkeypatch, capsys):
    config_yaml, root = _build_env(tmp_path)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train" / "ghost.jpg").symlink_to(tmp_path / "does_not_exist.jpg")
    assert _run(config_yaml, monkeypatch) == 1
    assert "broken symlink" in capsys.readouterr().out


def test_preflight_fails_on_wrong_batch(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, batch=4)  # obsolete Phase 2 batch size
    assert _run(config_yaml, monkeypatch) == 1
    assert "batch = 4, expected 16" in capsys.readouterr().out


def test_preflight_fails_on_wrong_dataset(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, dataset_key="xwod")
    assert _run(config_yaml, monkeypatch) == 1
    assert "dataset = xwod, expected phase2_merged" in capsys.readouterr().out


def test_preflight_fails_on_wrong_from_run(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, from_run="stage1_bdd30k_rtdetr")
    assert _run(config_yaml, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "from_run = stage1_bdd30k_rtdetr, expected stage2_xwod_rtdetr_from_bdd30k" in out


def test_preflight_fails_on_wrong_seed(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, seed=0)
    assert _run(config_yaml, monkeypatch) == 1
    assert "seed = 0, expected 42" in capsys.readouterr().out


def test_preflight_fails_on_wrong_lr0(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, lr0=0.0005)
    assert _run(config_yaml, monkeypatch) == 1
    assert "lr0 = 0.0005, expected 5e-05" in capsys.readouterr().out


def test_preflight_fails_on_wrong_checkpoint_source(tmp_path, monkeypatch, capsys):
    # Checkpoint physically exists, but under the WRONG from_run directory relative
    # to what the config declares — simulates a hard-coded / mismatched checkpoint.
    config_yaml, root = _build_env(tmp_path, from_run=FROM_RUN)
    # Move the checkpoint to a different run folder while the config still says FROM_RUN,
    # by pointing dataset paths.yaml's project elsewhere is complex; instead corrupt by
    # deleting the correct one and creating it under a decoy folder to prove resolution
    # is strict (the decoy must NOT be picked up).
    correct = root.parents[1] / "runs" / FROM_RUN / "weights" / "best.pt"
    correct.unlink()
    decoy = root.parents[1] / "runs" / "some_other_run" / "weights" / "best.pt"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(b"fake")
    assert _run(config_yaml, monkeypatch) == 1
    assert "checkpoint missing" in capsys.readouterr().out


def test_preflight_fails_on_missing_stats_json(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, write_stats=False)
    assert _run(config_yaml, monkeypatch) == 1
    assert "stats.json missing" in capsys.readouterr().out


def test_preflight_fails_on_missing_manifest_csv(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, write_manifest=False)
    assert _run(config_yaml, monkeypatch) == 1
    assert "manifest.csv missing" in capsys.readouterr().out


def test_preflight_fails_on_wrong_xwod_train_count(tmp_path, monkeypatch, capsys):
    counts = {**SOURCE_COUNTS, "xwod_train": 999}
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"source_counts": counts})
    assert _run(config_yaml, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "images_per_source.xwod_train = 999" in out


def test_preflight_fails_on_wrong_acdc_train_count(tmp_path, monkeypatch, capsys):
    counts = {**SOURCE_COUNTS, "acdc_train": 999}
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"source_counts": counts})
    assert _run(config_yaml, monkeypatch) == 1
    assert "images_per_source.acdc_train = 999" in capsys.readouterr().out


def test_preflight_fails_on_wrong_bdd_train_count(tmp_path, monkeypatch, capsys):
    counts = {**SOURCE_COUNTS, "bdd_train": 999}
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"source_counts": counts})
    assert _run(config_yaml, monkeypatch) == 1
    assert "images_per_source.bdd_train = 999" in capsys.readouterr().out


def test_preflight_fails_on_wrong_xwod_val_count(tmp_path, monkeypatch, capsys):
    counts = {**SOURCE_COUNTS, "xwod_val": 999}
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"source_counts": counts})
    assert _run(config_yaml, monkeypatch) == 1
    assert "images_per_source.xwod_val = 999" in capsys.readouterr().out


def test_preflight_fails_on_wrong_base_train_images(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"base_train_images": 999, "train_total": 999})
    assert _run(config_yaml, monkeypatch) == 1
    assert "base_train_images = 999" in capsys.readouterr().out


def test_preflight_fails_on_oversampling_disabled(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"oversample_enabled": False})
    assert _run(config_yaml, monkeypatch) == 1
    assert "oversample_rare.enabled = False" in capsys.readouterr().out


def test_preflight_fails_on_wrong_multiplier(tmp_path, monkeypatch, capsys):
    bad_multipliers = {"bicycle": 1, "motorcycle": 3, "bus": 3}
    config_yaml, _ = _build_env(tmp_path, stats_kwargs={"multipliers": bad_multipliers})
    assert _run(config_yaml, monkeypatch) == 1
    assert "oversample_rare.multipliers" in capsys.readouterr().out


def test_preflight_fails_on_inconsistent_train_total(tmp_path, monkeypatch, capsys):
    config_yaml, _ = _build_env(
        tmp_path, stats_kwargs={"duplicate_images": 2, "train_total": BASE_TRAIN_IMAGES + 5})
    assert _run(config_yaml, monkeypatch) == 1
    assert "train_total (base_train_images + duplicate_images)" in capsys.readouterr().out


def test_preflight_fails_on_train_val_overlap(tmp_path, monkeypatch, capsys):
    rows = _manifest_rows(SOURCE_COUNTS, overlap=True)
    config_yaml, _ = _build_env(tmp_path, manifest_rows=rows)
    assert _run(config_yaml, monkeypatch) == 1
    assert "appear in both train and val" in capsys.readouterr().out


def test_preflight_fails_on_unexpected_manifest_source(tmp_path, monkeypatch, capsys):
    rows = _manifest_rows(SOURCE_COUNTS, bad_source=True)
    config_yaml, _ = _build_env(tmp_path, manifest_rows=rows)
    assert _run(config_yaml, monkeypatch) == 1
    assert "unexpected source_dataset" in capsys.readouterr().out


def test_preflight_fails_on_leaked_source_split(tmp_path, monkeypatch, capsys):
    # BDD *val* images sourced into the merged *train* split — must be caught even
    # though the row is correctly labeled split="train" in the merged manifest.
    rows = _manifest_rows(SOURCE_COUNTS, leaked_split=True)
    config_yaml, _ = _build_env(tmp_path, manifest_rows=rows)
    assert _run(config_yaml, monkeypatch) == 1
    assert "possible val/test leakage" in capsys.readouterr().out
