#!/usr/bin/env python3
"""Compare evaluated runs against a chosen baseline run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "precision",
    "recall",
    "map50",
    "map50_95",
    "ultralytics_inference_ms",
    "inference_ms_batch1",
    "fps_batch1",
    "parameters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from scripts/collect_results.py.")
    parser.add_argument("--baseline", required=True, help="Run name used as comparison baseline.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input)
    if args.baseline not in set(table["run"]):
        raise ValueError(f"Baseline run not found in table: {args.baseline}")

    baseline = table.loc[table["run"] == args.baseline].iloc[0]
    rows = []
    for _, row in table.iterrows():
        result = {"run": row["run"], "baseline": args.baseline}
        for metric in METRIC_COLUMNS:
            if metric not in table.columns:
                continue
            result[metric] = row[metric]
            result[f"{metric}_delta"] = row[metric] - baseline[metric]
            if baseline[metric] != 0:
                result[f"{metric}_relative_pct"] = (
                    (row[metric] - baseline[metric]) / baseline[metric] * 100
                )
        rows.append(result)

    comparison = pd.DataFrame(rows)
    output = Path(args.output) if args.output else Path(args.input).with_name("comparison.csv")
    comparison.to_csv(output, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
