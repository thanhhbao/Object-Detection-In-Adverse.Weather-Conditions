"""Tests for scripts/preflight_p2_a1.py.

Covers:
- Test F: broken symlink causes failure
- Test J: P2-A0C and P2-A1 configs have identical epochs/batch/lr0/seed/patience
- Test K: both init from phase2_final_rtdetr (not stage2)
- overlap > 0 in pool_stats causes failure
- missing checkpoint causes failure
- happy path passes (with mocked dataset, no real checkpoint)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_p2_a1.py"
A0C_CONFIG = ROOT / "configs" / "ultralytics" / "p2_a0c_rtdetr_continue_control.yaml"
A1_CONFIG = ROOT / "configs" / "ultralytics" / "p2_a1_rtdetr_bdd_retrieval.yaml"

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def _find_python_with_yaml() -> str:
    """Find a Python executable that has PyYAML (needed by preflight scripts)."""
    # Try candidates in order: python3, python, then current executable
    for candidate in ("python3", "python", sys.executable):
        exe = shutil.which(candidate) or candidate
        try:
            result = subprocess.run(
                [exe, "-c", "import yaml; print('ok')"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return exe
        except (OSError, subprocess.TimeoutExpired):
            continue
    return sys.executable  # fallback — may fail if yaml missing


_PYTHON = _find_python_with_yaml()


def _run(args: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_PYTHON, str(SCRIPT), *args],
        check=check, cwd=ROOT, capture_output=True, text=True,
    )


def _dump_yaml(data: dict) -> str:
    """Minimal YAML serializer that avoids yaml dependency in test module.

    Note: floats are serialized as decimal (not scientific notation) so that
    PyYAML 6.0+ parses them as float, not string.  PyYAML on Python 3.13 treats
    '1e-05' as a string, but '0.00001' as a float.
    """
    def _fmt_float(v: float) -> str:
        # Use decimal representation to avoid YAML string-misparse of sci-notation
        return f"{v:.15f}".rstrip("0").rstrip(".")

    lines = []
    for k, v in data.items():
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, float):
            lines.append(f"{k}: {_fmt_float(v)}")
        elif isinstance(v, int):
            lines.append(f"{k}: {v}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for dk, dv in v.items():
                lines.append(f"  {dk}: {dv}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def _build_fake_dataset(root: Path, val_count: int = 2) -> Path:
    """Create a minimal fake dataset that satisfies structure checks."""
    val_img = root / "images" / "val"
    val_lbl = root / "labels" / "val"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)
    for i in range(val_count):
        (val_img / f"xwod_val_{i:03d}.jpg").write_bytes(b"fake")
        (val_lbl / f"xwod_val_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    train_img = root / "images" / "train"
    train_lbl = root / "labels" / "train"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    (train_img / "bdd_retrieved_001.jpg").write_bytes(b"fake")
    (train_lbl / "bdd_retrieved_001.txt").write_text("1 0.5 0.5 0.1 0.1", encoding="utf-8")

    # dataset.yaml — write manually without yaml library
    ds_yaml_text = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES))
    )
    (root / "dataset.yaml").write_text(ds_yaml_text, encoding="utf-8")

    stats = {
        "seed": 42,
        "mode": "copy",
        "bdd_mode": "full",
        "bdd_replay_ratio": None,
        "bdd_replay_images_requested": None,
        "bdd_replay_images_actual": None,
        "images_per_source": {
            "xwod_train": 5, "acdc_train": 1, "bdd_train": 30000,
            "bdd_retrieved_train": 500, "xwod_val": val_count,
        },
        "oversample_rare": {
            "enabled": True, "multipliers": {"bicycle": 2, "motorcycle": 3, "bus": 3},
            "base_train_images": 30006, "duplicate_images": 100,
            "train_images_after": 30106, "multiplier_distribution": {},
        },
        "train_total": 30106,
        "val_total": val_count,
        "train_boxes_per_class_before": {c: 100 for c in TARGET_CLASSES},
        "train_boxes_per_class_after": {c: 120 for c in TARGET_CLASSES},
        "val_boxes_per_class": {c: 10 for c in TARGET_CLASSES},
        "retrieved_train": 500,
        "base_train_total_before_retrieval": 30006,
        "train_total_with_retrieval_before_oversampling": 30506,
        "retrieved_class_counts": {c: 50 for c in TARGET_CLASSES},
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    import csv
    rows = [
        {"source_dataset": "xwod", "split": "train", "image_path": str(root / "images/train/xwod_t.jpg"),
         "label_path": "", "new_image": "", "new_label": ""},
        {"source_dataset": "bdd_retrieved", "split": "train",
         "image_path": str(root / "images/train/bdd_retrieved_001.jpg"),
         "label_path": "", "new_image": "", "new_label": ""},
        {"source_dataset": "xwod", "split": "val",
         "image_path": str(root / "images/val/xwod_val_000.jpg"),
         "label_path": "", "new_image": "", "new_label": ""},
    ]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return root


def _build_fake_config(tmp_path: Path, dataset_yaml: Path,
                       checkpoint: Path, project: Path,
                       **overrides) -> Path:
    """Build a minimal p2-a1-style config YAML for the preflight script."""
    cfg_data = {
        "name": "p2_a1_rtdetr_bdd_retrieval",
        "data": str(dataset_yaml),
        "project": str(project),
        "model": str(checkpoint),
        "from_run": "phase2_final_rtdetr",
        "dataset": "phase2_a1_merged",
        "epochs": 20,
        "patience": 8,
        "lr0": 0.00001,
        "batch": 16,
        "seed": 42,
    }
    cfg_data.update(overrides)
    cfg_path = tmp_path / "test_p2_a1.yaml"
    cfg_path.write_text(_dump_yaml(cfg_data), encoding="utf-8")
    return cfg_path


# ── Test: happy path passes ───────────────────────────────────────────────────

def test_happy_path_passes(tmp_path):
    """When checkpoint exists and dataset is valid, preflight must exit 0."""
    dataset = _build_fake_dataset(tmp_path / "dataset")
    project = tmp_path / "runs"
    ckpt = project / "phase2_final_rtdetr" / "weights" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake-checkpoint")

    cfg = _build_fake_config(tmp_path, dataset / "dataset.yaml", ckpt, project)

    result = _run(["--config", str(cfg)])
    assert result.returncode == 0, (
        f"Expected preflight to pass.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FINAL P2-A1 PREFLIGHT: PASS" in result.stdout


# ── Test: missing checkpoint causes failure ────────────────────────────────────

def test_missing_checkpoint_fails(tmp_path):
    dataset = _build_fake_dataset(tmp_path / "dataset")
    project = tmp_path / "runs"
    ckpt = project / "phase2_final_rtdetr" / "weights" / "best.pt"
    # Do NOT create checkpoint

    cfg = _build_fake_config(tmp_path, dataset / "dataset.yaml", ckpt, project)

    result = _run(["--config", str(cfg)])
    assert result.returncode != 0, "Expected failure when checkpoint is missing"
    assert "FAIL" in result.stdout


# ── Test F: broken symlink causes failure ─────────────────────────────────────

def test_broken_symlink_in_dataset_fails(tmp_path):
    dataset = _build_fake_dataset(tmp_path / "dataset")
    project = tmp_path / "runs"
    ckpt = project / "phase2_final_rtdetr" / "weights" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake-checkpoint")

    # Create a broken symlink in train
    broken = dataset / "images" / "train" / "broken_link.jpg"
    broken.symlink_to(tmp_path / "nonexistent" / "file.jpg")  # points to nothing

    cfg = _build_fake_config(tmp_path, dataset / "dataset.yaml", ckpt, project)

    result = _run(["--config", str(cfg)])
    assert result.returncode != 0, "Expected failure due to broken symlink"
    assert "broken" in result.stdout.lower() or "symlink" in result.stdout.lower()


# ── Test: overlap > 0 in pool_stats causes failure ────────────────────────────

def test_pool_stats_overlap_nonzero_fails(tmp_path):
    dataset = _build_fake_dataset(tmp_path / "dataset")
    project = tmp_path / "runs"
    ckpt = project / "phase2_final_rtdetr" / "weights" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake-checkpoint")

    cfg = _build_fake_config(tmp_path, dataset / "dataset.yaml", ckpt, project)

    pool_stats = {
        "full_train_pairs": 100, "used_bdd_train": 30000,
        "used_bdd_val": 1000, "used_bdd_test": 1000,
        "excluded_existing_names": 30000, "candidate_pool": 70000,
        "overlap_with_used_train": 5,  # NON-ZERO — should fail
        "overlap_with_used_val": 0,
        "overlap_with_used_test": 0,
    }
    pool_stats_path = tmp_path / "pool_stats.json"
    pool_stats_path.write_text(json.dumps(pool_stats))

    result = _run(["--config", str(cfg), "--pool-stats", str(pool_stats_path)])
    assert result.returncode != 0, "Expected failure when pool_stats has overlap > 0"
    assert "FAIL" in result.stdout


# ── Test J: A0C and A1 configs have identical hyperparams ─────────────────────

def test_a0c_a1_configs_have_identical_hyperparams():
    """P2-A0C and P2-A1 configs must share identical epochs/batch/lr0/seed/patience."""
    import re

    def parse_yaml_simple(text: str) -> dict:
        """Parse simple flat YAML (no nesting needed for these configs)."""
        result = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                if v:
                    try:
                        if "." in v:
                            result[k] = float(v)
                        else:
                            result[k] = int(v)
                    except ValueError:
                        result[k] = v
        return result

    a0c = parse_yaml_simple(A0C_CONFIG.read_text())
    a1 = parse_yaml_simple(A1_CONFIG.read_text())

    for key in ("epochs", "batch", "lr0", "seed", "patience"):
        assert key in a0c and key in a1, f"Key {key} missing from one config"
        assert a0c[key] == a1[key], (
            f"Config parity mismatch for {key}: A0C={a0c[key]}, A1={a1[key]}"
        )


# ── Test K: both configs init from phase2_final_rtdetr ────────────────────────

def test_both_configs_init_from_phase2_final_rtdetr():
    """Both A0C and A1 must use from_run=phase2_final_rtdetr (not stage2)."""
    def get_from_run(text: str) -> str:
        for line in text.splitlines():
            if line.strip().startswith("from_run:"):
                return line.split(":", 1)[1].strip()
        return ""

    a0c_from_run = get_from_run(A0C_CONFIG.read_text())
    a1_from_run = get_from_run(A1_CONFIG.read_text())

    assert a0c_from_run == "phase2_final_rtdetr", (
        f"A0C from_run = {a0c_from_run!r}, expected phase2_final_rtdetr"
    )
    assert a1_from_run == "phase2_final_rtdetr", (
        f"A1 from_run = {a1_from_run!r}, expected phase2_final_rtdetr"
    )
    assert "stage2" not in a0c_from_run
    assert "stage2" not in a1_from_run
