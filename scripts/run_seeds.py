#!/usr/bin/env python3
"""Run one experiment across several seeds and report mean ± std.

DAWN nhỏ nên kết luận từ một seed dễ bất ổn (xem docs/RESEARCH_PROTOCOL.md).
Script này lặp cùng một config qua nhiều seed:

1. Train mỗi seed vào thư mục run riêng `<name>_seed<k>`.
2. Evaluate mỗi seed trên split được chọn (ghi `<split>_metrics.json`).
3. Gom Precision / Recall / mAP50 / mAP50-95 và in mean ± std.

Tự chọn trainer/evaluator theo loại config:
- configs/ablation + variant cbam -> train.py
- configs/torchvision           -> train_torchvision.py + evaluate_torchvision.py
- còn lại (configs/ultralytics)  -> train_ultralytics.py + evaluate.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.common import experiment_run_dir, load_experiment_config

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--skip-train", action="store_true", help="Only re-evaluate and aggregate.")
    return parser.parse_args()


def dispatch_scripts(config_path: str, config: dict) -> tuple[str, str]:
    parts = Path(config_path).parts
    if "torchvision" in parts:
        return "train_torchvision.py", "evaluate_torchvision.py"
    if "ablation" in parts and config.get("variant") == "cbam":
        return "train.py", "evaluate.py"
    return "train_ultralytics.py", "evaluate.py"


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    base_name = config["name"]
    trainer, evaluator = dispatch_scripts(args.config, config)
    print(f"config: {args.config}\ntrainer: {trainer} | evaluator: {evaluator}")

    per_seed = []
    for seed in args.seeds:
        run_name = f"{base_name}_seed{seed}"

        if not args.skip_train:
            run([
                sys.executable, str(ROOT / "scripts" / trainer),
                "--config", args.config, "--name", run_name, "--seed", str(seed),
            ])

        run([
            sys.executable, str(ROOT / "scripts" / evaluator),
            "--config", args.config, "--split", args.split, "--run-name", run_name,
        ])

        config_for_run = {**config, "name": run_name}
        metrics_path = experiment_run_dir(config_for_run) / f"{args.split}_metrics.json"
        per_seed.append({"seed": seed, **json.loads(metrics_path.read_text(encoding="utf-8"))})

    summary = {key: aggregate([row[key] for row in per_seed]) for key in METRIC_KEYS}

    output = experiment_run_dir(config) / f"{args.split}_seeds_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"run": base_name, "seeds": args.seeds, "summary": summary, "per_seed": per_seed}, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== {base_name} over {len(args.seeds)} seeds ({args.split}) ===")
    for key in METRIC_KEYS:
        stat = summary[key]
        print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}  (min {stat['min']:.4f}, max {stat['max']:.4f})")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
