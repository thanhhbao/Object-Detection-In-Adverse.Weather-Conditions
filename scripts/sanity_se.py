#!/usr/bin/env python3
"""Sanity-check the +SE ablation before launching a full training run.

Verifies:
  1. The architecture builds and contains EXACTLY ONE SEResearch block.
  2. That SE block sits immediately AFTER the SPPF layer (single-variable change).
  3. SE parameters are eager (exist before optimizer) and require grad.
  4. The data YAML has nc=6 and the correct class order.
  5. Pretrained weights transfer into the SE model (backbone + head), skipping SE.
  6. A forward pass runs and produces detection outputs.

Example:
  python scripts/sanity_se.py --cfg configs/yolov8n_se.yaml --data configs/xwod.yaml \\
      --weights /content/workspace/runs/stage1_bdd30k_yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dawn_ablation.common import register_custom_modules  # noqa: E402

EXPECTED_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()

    register_custom_modules()
    from ultralytics import YOLO
    from ultralytics.nn.modules import SPPF
    from dawn_ablation.attention import SEResearch

    results: list[bool] = []

    # 1–3. Architecture + SE placement + eager params
    model = YOLO(args.cfg)
    layers = list(model.model.model)
    se_idx = [i for i, l in enumerate(layers) if isinstance(l, SEResearch)]
    sppf_idx = [i for i, l in enumerate(layers) if isinstance(l, SPPF)]

    results.append(check("Exactly one SEResearch block", len(se_idx) == 1,
                         f"found {len(se_idx)}"))
    placed = bool(se_idx) and bool(sppf_idx) and se_idx[0] == sppf_idx[0] + 1
    results.append(check("SE is immediately after SPPF", placed,
                         f"SPPF@{sppf_idx} SE@{se_idx}"))
    if se_idx:
        se = layers[se_idx[0]]
        nparam = sum(p.numel() for p in se.parameters())
        eager = nparam > 0 and all(p.requires_grad for p in se.parameters())
        results.append(check("SE params eager & require grad", eager, f"{nparam} params"))

    # 4. Data YAML
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = data.get("names", {})
    names_list = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    results.append(check("Data nc == 6", data.get("nc") == 6, str(data.get("nc"))))
    results.append(check("Class order correct", names_list == EXPECTED_CLASSES, str(names_list)))

    # 5. Pretrained transfer
    from train import transfer_pretrained
    stats = transfer_pretrained(model.model, args.weights)
    ok_transfer = stats["loaded_tensors"] > 0 and stats["skipped_attention_layers"] == len(se_idx)
    results.append(check("Pretrained weights transferred", ok_transfer, str(stats)))

    # 6. Forward pass
    try:
        model.model.eval()
        with torch.no_grad():
            out = model.model(torch.zeros(1, 3, 640, 640))
        fwd_ok = out is not None
    except Exception as exc:  # noqa: BLE001
        fwd_ok = False
        print("forward error:", exc)
    results.append(check("Forward pass runs", fwd_ok))

    print("\n" + ("ALL CHECKS PASSED ✅" if all(results) else "SOME CHECKS FAILED ❌"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
