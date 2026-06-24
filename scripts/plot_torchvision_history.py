#!/usr/bin/env python3
"""Vẽ biểu đồ huấn luyện cho run Faster R-CNN từ train_history.json.

TorchVision không tự vẽ như Ultralytics, nên script này đọc train_history.json
(do train_torchvision.py lưu) và xuất các đường cong theo epoch: train loss,
mAP50, mAP50-95, Precision, Recall — để có hình minh họa cho luận văn.

Chạy:
  python scripts/plot_torchvision_history.py --config configs/torchvision/stage2_dawn_faster_rcnn_from_bdd.yaml
hoặc trỏ thẳng thư mục run:
  python scripts/plot_torchvision_history.py --run-dir <project>/stage2_dawn_faster_rcnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # không cần màn hình
import matplotlib.pyplot as plt

from dawn_ablation.common import experiment_run_dir, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.config:
        run_dir = experiment_run_dir(load_experiment_config(args.config))
    else:
        raise SystemExit("Cần --config hoặc --run-dir")

    history_path = run_dir / "train_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Không thấy {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
    epochs = [row["epoch"] + 1 for row in history]

    def series(key):
        return [row.get(key) for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, series("train_loss"), color="tab:red", marker="o", ms=3)
    axes[0].set_title("Train loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, series("map50"), label="mAP@50", marker="o", ms=3)
    axes[1].plot(epochs, series("map50_95"), label="mAP@50-95", marker="s", ms=3)
    axes[1].set_title("mAP (val)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, series("precision"), label="Precision", marker="o", ms=3)
    axes[2].plot(epochs, series("recall"), label="Recall", marker="s", ms=3)
    axes[2].set_title("Precision / Recall (val)"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Giá trị")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Quá trình huấn luyện: {run_dir.name}")
    fig.tight_layout()

    output = run_dir / "training_curves.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
