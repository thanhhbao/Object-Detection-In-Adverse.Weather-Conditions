#!/usr/bin/env python3
"""Evaluate multiple Ultralytics runs using the shared evaluation script."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=sorted(str(path) for path in (ROOT / "configs" / "ultralytics").glob("stage2_*.yaml")),
        help="One or more Ultralytics experiment configs.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    return parser.parse_args()


def evaluator_for(config: str) -> str:
    """Faster R-CNN configs live under configs/torchvision and use a separate
    evaluator; everything else is an Ultralytics run."""
    name = "evaluate_torchvision.py" if "torchvision" in Path(config).parts else "evaluate.py"
    return str(ROOT / "scripts" / name)


def main() -> None:
    args = parse_args()
    for config in args.configs:
        command = [
            sys.executable,
            evaluator_for(config),
            "--config",
            config,
            "--split",
            args.split,
        ]
        print(f"\nRunning: {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
