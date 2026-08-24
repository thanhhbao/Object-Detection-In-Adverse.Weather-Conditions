#!/usr/bin/env python3
"""Run one experiment across several seeds and report mean ± std per evaluation dataset.

Script này lặp cùng một config qua nhiều seed:
1. Train mỗi seed vào thư mục run riêng `<name>_seed<k>`.
2. Evaluate mỗi seed trên từng dataset được chỉ định.
3. Gom Precision / Recall / mAP50 / mAP50-95 và in mean ± std cho từng dataset.

Tự chọn trainer/evaluator theo loại config:
- configs/ablation + variant cbam -> train.py
- configs/torchvision           -> train_torchvision.py + evaluate_torchvision.py
- còn lại (configs/ultralytics)  -> train_ultralytics.py + evaluate.py

Ví dụ đánh giá ba miền (E2/E3 tuần 3):
  export OD_PATHS=configs/common/paths_vast.yaml
  python scripts/run_seeds.py \\
      --config configs/ablation/phase2_yolov8n_p2.yaml \\
      --seeds 42 43 44 \\
      --init-weights /workspace/runs/stage2_xwod_yolov8n_from_bdd30k/weights/best.pt \\
      --max-transfer-layer 15 \\
      --data /workspace/datasets/phase2_v3/dataset.yaml \\
      --eval-data xwod_test:test=/workspace/datasets/xwod_6cls_yolo/dataset.yaml \\
      --eval-data dawn_val:val=/workspace/datasets/dawn_6cls_yolo/dataset.yaml \\
      --eval-data acdc_val:val=/workspace/datasets/acdc_6cls_yolo/dataset.yaml

Kết quả ghi riêng cho từng dataset:
  runs/<name>/xwod_test_seeds_summary.json
  runs/<name>/dawn_val_seeds_summary.json
  runs/<name>/acdc_val_seeds_summary.json
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
    parser = argparse.ArgumentParser(
        description="Multi-seed training and evaluation with per-dataset summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split", default="val", choices=("train", "val", "test"),
                        help="Eval split used when --eval-data is NOT specified (legacy mode).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only re-evaluate and aggregate existing checkpoints.")

    # ── Training pass-through ──────────────────────────────────────────────
    parser.add_argument("--data", default=None,
                        help="Training dataset YAML forwarded to trainer. "
                             "Does NOT control evaluation datasets; use --eval-data for that.")
    parser.add_argument("--init-weights", default=None,
                        help="Checkpoint for partial weight transfer, forwarded to trainer.")
    parser.add_argument("--max-transfer-layer", type=int, default=None,
                        help="Layer cap for --init-weights (e.g. 15 for P2), forwarded to trainer.")

    # ── Evaluation datasets ────────────────────────────────────────────────
    parser.add_argument(
        "--eval-data",
        action="append",
        dest="eval_data",
        metavar="TAG:SPLIT=/path/dataset.yaml",
        help="Evaluation dataset spec: TAG (output file prefix), SPLIT (val|test), "
             "and path to dataset.yaml.  Can be repeated for multiple datasets.  "
             "Example: --eval-data xwod_test:test=/workspace/datasets/xwod_6cls_yolo/dataset.yaml  "
             "When provided, --split is ignored.  "
             "Each dataset writes its own {TAG}_metrics.json and {TAG}_seeds_summary.json.",
    )
    return parser.parse_args()


def parse_eval_specs(spec_list: list[str]) -> list[tuple[str, str, str]]:
    """Parse 'TAG:SPLIT=/path' → list of (tag, split, data_path).

    SPLIT defaults to 'val' if omitted (i.e. 'TAG=/path' form is also accepted).
    """
    result = []
    for spec in spec_list:
        tag_split, _, data_path = spec.partition("=")
        if not data_path:
            raise ValueError(
                f"--eval-data spec must contain '=': got '{spec}'\n"
                "Expected format: TAG:SPLIT=/path/to/dataset.yaml"
            )
        if ":" in tag_split:
            tag, _, split = tag_split.partition(":")
        else:
            tag, split = tag_split, "val"
        result.append((tag.strip(), split.strip(), data_path.strip()))
    return result


def dispatch_scripts(config_path: str, config: dict) -> tuple[str, str]:
    parts = Path(config_path).parts
    if "torchvision" in parts:
        return "train_torchvision.py", "evaluate_torchvision.py"
    if "ablation" in parts and config.get("variant") == "cbam":
        return "train.py", "evaluate.py"
    return "train_ultralytics.py", "evaluate.py"


def supports_eval_data_flag(evaluator: str) -> bool:
    """Only evaluate.py (Ultralytics) supports --data and --metrics-tag."""
    return evaluator == "evaluate.py"


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


def _eval_one(
    evaluator: str,
    config_path: str,
    run_name: str,
    split: str,
    data_path: str | None,
    metrics_tag: str,
) -> None:
    """Call evaluator for a single (run, dataset) combination."""
    eval_cmd = [
        sys.executable, str(ROOT / "scripts" / evaluator),
        "--config", config_path,
        "--split", split,
        "--run-name", run_name,
        "--metrics-tag", metrics_tag,
    ]
    if data_path and supports_eval_data_flag(evaluator):
        eval_cmd += ["--data", data_path]
    elif data_path:
        print(f"[run_seeds] WARNING: {evaluator} does not support --data; "
              "using dataset from config.")
    run(eval_cmd)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    base_name = config["name"]
    trainer, evaluator = dispatch_scripts(args.config, config)
    print(f"config: {args.config}\ntrainer: {trainer} | evaluator: {evaluator}")

    # Resolve evaluation specs
    if args.eval_data:
        eval_specs = parse_eval_specs(args.eval_data)
        print(f"Evaluation datasets ({len(eval_specs)}):")
        for tag, split, path in eval_specs:
            print(f"  {tag}:{split} → {path}")
    else:
        # Legacy single-dataset mode
        eval_specs = [("", args.split, None)]  # empty tag → legacy filename

    # Per-dataset accumulator: tag → list of per-seed metric dicts
    per_seed_by_tag: dict[str, list[dict]] = {tag: [] for tag, _, _ in eval_specs}

    for seed in args.seeds:
        run_name = f"{base_name}_seed{seed}"

        # ── Training ──────────────────────────────────────────────────────
        if not args.skip_train:
            train_cmd = [
                sys.executable, str(ROOT / "scripts" / trainer),
                "--config", args.config, "--name", run_name, "--seed", str(seed),
            ]
            if args.data:
                train_cmd += ["--data", args.data]
            if args.init_weights:
                train_cmd += ["--init-weights", args.init_weights]
            if args.max_transfer_layer is not None:
                train_cmd += ["--max-transfer-layer", str(args.max_transfer_layer)]
            run(train_cmd)

        # ── Evaluation (one call per dataset) ─────────────────────────────
        config_for_run = {**config, "name": run_name}
        for tag, split, data_path in eval_specs:
            metrics_tag = tag if tag else split  # legacy: tag="" → fall back to split
            _eval_one(evaluator, args.config, run_name, split, data_path, metrics_tag)

            metrics_path = experiment_run_dir(config_for_run) / f"{metrics_tag}_metrics.json"
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
            per_seed_by_tag[tag].append({"seed": seed, **row})

    # ── Aggregate and save per dataset ────────────────────────────────────
    run_dir = experiment_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    for tag, split, _ in eval_specs:
        per_seed = per_seed_by_tag[tag]
        summary = {key: aggregate([row[key] for row in per_seed]) for key in METRIC_KEYS}
        metrics_tag = tag if tag else split

        output = run_dir / f"{metrics_tag}_seeds_summary.json"
        output.write_text(
            json.dumps({
                "run": base_name,
                "dataset_tag": metrics_tag,
                "seeds": args.seeds,
                "summary": summary,
                "per_seed": per_seed,
            }, indent=2),
            encoding="utf-8",
        )

        label = f"{base_name} / {metrics_tag} / {len(args.seeds)} seeds"
        print(f"=== {label} ===")
        for key in METRIC_KEYS:
            stat = summary[key]
            print(
                f"  {key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}"
                f"  (min {stat['min']:.4f}, max {stat['max']:.4f})"
            )
        print(f"  Saved: {output}")
        print()


if __name__ == "__main__":
    main()
