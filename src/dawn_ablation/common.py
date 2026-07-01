"""Shared experiment helpers."""

from __future__ import annotations

import json
import os
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


def resolve_from_root(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return config


def merge_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge shared defaults and paths into one runnable experiment config.

    The paths file is resolved in this order (first match wins):
      1. OD_PATHS environment variable  (e.g. export OD_PATHS=configs/common/paths_vast.yaml)
      2. 'paths' key inside the experiment YAML
    """
    merged: dict[str, Any] = {}

    if "defaults" in config:
        merged.update(load_config(resolve_from_root(config["defaults"])))

    merged.update(config)

    env_paths = os.environ.get("OD_PATHS")
    if env_paths:
        merged["paths"] = env_paths

    if "paths" in merged:
        paths_config = load_config(resolve_from_root(merged["paths"]))
        merged.setdefault("project", paths_config.get("project"))

        dataset_key = merged.get("dataset")
        if dataset_key and "data" not in merged:
            try:
                merged["data"] = paths_config["datasets"][dataset_key]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset key in config: {dataset_key}") from exc

    if "from_run" in merged and "model" not in merged:
        checkpoint_file = merged.get("checkpoint_file", "best.pt")
        merged["model"] = (
            Path(merged["project"]) / merged["from_run"] / "weights" / checkpoint_file
        )

    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML and apply shared defaults/paths."""
    config = load_config(resolve_from_root(path))
    if "base_config" not in config:
        return merge_experiment_config(config)

    base = load_experiment_config(config["base_config"])
    merged = {**base, **config}
    merged["project"] = base["project"]
    merged["data"] = base["data"]
    merged.setdefault("model", base.get("model"))
    return merged


def experiment_run_dir(config: dict[str, Any]) -> Path:
    """Return the run folder used by both training and evaluation scripts."""
    return resolve_from_root(config["project"]) / config["name"]


def experiment_checkpoint(config: dict[str, Any], filename: str = "best.pt") -> Path:
    """Return a checkpoint path inside the run folder."""
    return experiment_run_dir(config) / "weights" / filename


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def register_custom_modules() -> None:
    """Expose local modules to Ultralytics' YAML parser before model creation."""
    from ultralytics.nn import tasks

    from dawn_ablation.attention import CBAMResearch

    tasks.CBAMResearch = CBAMResearch
