#!/usr/bin/env python3
"""Create a fair baseline-vs-CBAM comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from dawn_ablation.common import load_config, resolve_from_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation/dawn_cbam_local.yaml")
    args = parser.parse_args()
    config = load_config(resolve_from_root(args.config))
    project = resolve_from_root(config["project"])

    rows = []
    for variant in ("baseline", "cbam"):
        path = project / variant / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Run evaluation first; missing {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    baseline, cbam = rows
    delta = {"variant": "cbam_minus_baseline"}
    relative = {"variant": "cbam_relative_change_pct"}
    for key in baseline:
        if key != "variant":
            delta[key] = cbam[key] - baseline[key]
            relative[key] = (cbam[key] - baseline[key]) / baseline[key] * 100
    rows.append(delta)
    rows.append(relative)
    table = pd.DataFrame(rows)
    output = project / "ablation_comparison.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
