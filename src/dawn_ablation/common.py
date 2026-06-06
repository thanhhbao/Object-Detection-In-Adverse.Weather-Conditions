"""Shared experiment helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def ensure_project_imports() -> None:
    """Make local modules importable when scripts are run from any directory."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return config


def resolve_from_root(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def register_custom_modules() -> None:
    """Expose local modules to Ultralytics' YAML parser before model creation."""
    from ultralytics.nn import tasks

    from dawn_ablation.attention import CBAMResearch

    tasks.CBAMResearch = CBAMResearch


def variant_paths(variant: str, config: dict[str, Any]) -> tuple[Path, Path]:
    if variant not in {"baseline", "cbam"}:
        raise ValueError(f"Unknown variant: {variant}")
    model_name = (
        "yolov8n_baseline.yaml" if variant == "baseline" else "yolov8n_cbam_neck.yaml"
    )
    model_yaml = ROOT / "models" / model_name
    run_dir = resolve_from_root(config["project"]) / variant
    return model_yaml, run_dir
