#!/usr/bin/env python3
"""Collect evaluation JSON files into one CSV table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dawn_ablation.common import experiment_run_dir, load_experiment_config, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=sorted(str(path) for path in (ROOT / "configs" / "ultralytics").glob("stage2_*.yaml")),
        help="Configs whose run folders should be collected.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    first_config = None

    for config_path in args.configs:
        config = load_experiment_config(config_path)
        first_config = first_config or config
        metrics_path = experiment_run_dir(config) / f"{args.split}_metrics.json"
        if not metrics_path.exists():
            print(f"Skip missing metrics: {metrics_path}")
            continue
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    if not rows:
        raise RuntimeError("No metrics files found. Run scripts/evaluate.py or scripts/eval_all.py first.")

    table = pd.DataFrame(rows).sort_values("map50_95", ascending=False)
    output = (
        Path(args.output)
        if args.output
        else resolve_from_root(first_config["project"]) / f"{args.split}_summary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
