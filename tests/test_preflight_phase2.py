"""Unit tests for scripts/preflight_phase2.py — no GPU, no real dataset needed.

Builds a tiny synthetic run/dataset/config triple per test and asserts the
script exits non-zero (and prints the expected failure line) for each of the
required failure modes, plus one happy-path pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import preflight_phase2 as pf


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"fake")
    (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")


def _build_env(
    tmp_path: Path,
    *,
    class_names: list[str] | None = None,
    val_count: int = 2,
    checkpoint: bool = True,
) -> tuple[Path, Path]:
    class_names = class_names or pf.EXPECTED_CLASSES
    root = tmp_path / "datasets" / "merged"
    for i in range(val_count):
        _write_pair(root / "images" / "val", root / "labels" / "val", f"val_{i}")
    _write_pair(root / "images" / "train", root / "labels" / "train", "train_0")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (root / "dataset.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/val\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "run1" / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        (run_dir / "best.pt").write_bytes(b"fake")

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        f"project: {tmp_path / 'runs'}\ndatasets:\n  merged_ds: {root / 'dataset.yaml'}\n",
        encoding="utf-8",
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"paths: {paths_yaml}\nfrom_run: run1\ndataset: merged_ds\nname: run1\n"
        "epochs: 1\nbatch: 1\nimgsz: 640\noptimizer: AdamW\nlr0: 0.1\nseed: 42\n",
        encoding="utf-8",
    )
    return config_yaml, root


@pytest.fixture(autouse=True)
def _no_od_paths_env(monkeypatch):
    # Never let a real OD_PATHS env var override our synthetic paths.yaml.
    monkeypatch.delenv("OD_PATHS", raising=False)


@pytest.fixture(autouse=True)
def _relaxed_val_count(monkeypatch):
    # These fixtures build a 2-image val split, not the real 1001-image one.
    monkeypatch.setattr(pf, "EXPECTED_VAL_COUNT", 2)


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
    (root / "images" / "train" / "ghost.jpg").symlink_to(tmp_path / "does_not_exist.jpg")
    assert _run(config_yaml, monkeypatch) == 1
    assert "broken symlink" in capsys.readouterr().out


def test_preflight_fails_on_wrong_expected_source_counts(tmp_path, monkeypatch, capsys):
    config_yaml, root = _build_env(tmp_path)
    (root / "stats.json").write_text(
        json.dumps({"images_per_source": {"xwod_train": 1, "acdc_train": 1, "bdd_replay": 1}}),
        encoding="utf-8",
    )
    assert _run(config_yaml, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "stats.json xwod_train" in out
    assert "stats.json acdc_train" in out
    assert "stats.json bdd_replay" in out
