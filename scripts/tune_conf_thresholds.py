#!/usr/bin/env python3
"""Tune a per-class detection confidence threshold on a validation split.

Ultralytics reports per-class precision/recall at a single shared confidence:
`ap_per_class` picks the index maximising the *mean* F1 across classes, then
reads every class off that one index. Classes whose score distribution sits
lower than average — the rare ones here — are therefore evaluated at a
threshold tuned for someone else, and lose recall for it.

This script runs inference once at a very low threshold, matches detections to
ground truth, and then sweeps thresholds offline. That gives, per class, the
threshold maximising that class's own F1, and quantifies what the shared
threshold was costing.

Tune on val, never on test: the chosen thresholds become part of the model
configuration, so fitting them on test would invalidate it as held-out data.

Example:
  python scripts/tune_conf_thresholds.py \
    --weights /workspace/runs/p2_a1_dino_rtdetr_retrieval/weights/best.pt \
    --data /workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
    --split val --out /workspace/runs/evals/thresholds_acdc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
LOW_SUPPORT = 100


def resolve_split(data_yaml: Path, split: str) -> tuple[Path, Path]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    rel = cfg.get(split)
    if not rel:
        raise SystemExit(f"{data_yaml} has no '{split}' key")
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    img_dir = base / rel
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))
    if not img_dir.is_dir():
        raise SystemExit(f"Image dir not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise SystemExit(f"Label dir not found: {lbl_dir}")
    return img_dir, lbl_dir


def load_gt(lbl_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO normalised cx cy w h -> pixel xyxy, plus class ids."""
    if not lbl_path.is_file():
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
    boxes, cls = [], []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(c)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.array(boxes, dtype=np.float32), np.array(cls, dtype=np.int32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def collect(weights: str, img_dir: Path, lbl_dir: Path, imgsz: int, device: str,
            min_conf: float, match_iou: float) -> tuple[dict, dict]:
    """Run inference once, return per-class (conf, is_tp) records and GT counts."""
    from dawn_ablation.common import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise SystemExit(f"No images under {img_dir}")
    print(f"  {len(imgs)} images, inference at conf>={min_conf}")

    model = YOLO(weights)
    records: dict[int, list] = {i: [] for i in range(len(CLASS_NAMES))}
    n_gt: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}

    done = 0
    for res in model.predict(source=[str(p) for p in imgs], stream=True,
                             conf=min_conf, imgsz=imgsz, device=device,
                             verbose=False):
        done += 1
        if done % 500 == 0:
            print(f"    {done}/{len(imgs)}", flush=True)

        h, w = res.orig_shape
        stem = Path(res.path).stem
        gt_boxes, gt_cls = load_gt(lbl_dir / f"{stem}.txt", w, h)
        for c in gt_cls:
            if 0 <= c < len(CLASS_NAMES):
                n_gt[int(c)] += 1

        b = res.boxes
        if b is None or len(b) == 0:
            continue
        p_xyxy = b.xyxy.cpu().numpy()
        p_cls = b.cls.cpu().numpy().astype(int)
        p_conf = b.conf.cpu().numpy()

        # Match within each class: greedy, highest confidence first, one GT each.
        for c in np.unique(p_cls):
            if not (0 <= c < len(CLASS_NAMES)):
                continue
            pm = p_cls == c
            pb, pc = p_xyxy[pm], p_conf[pm]
            order = np.argsort(-pc)
            pb, pc = pb[order], pc[order]

            gm = gt_cls == c
            gb = gt_boxes[gm]
            taken = np.zeros(len(gb), dtype=bool)
            ious = iou_matrix(pb, gb)

            for i in range(len(pb)):
                tp = False
                if len(gb):
                    row = ious[i].copy()
                    row[taken] = -1.0
                    j = int(np.argmax(row)) if row.size else -1
                    if j >= 0 and row[j] >= match_iou:
                        taken[j] = True
                        tp = True
                records[int(c)].append((float(pc[i]), tp))

    return records, n_gt


def sweep(records: dict, n_gt: dict, grid: np.ndarray) -> dict:
    """F1 curve per class over the threshold grid."""
    out = {}
    for c in range(len(CLASS_NAMES)):
        recs = records[c]
        g = n_gt[c]
        if g == 0:
            out[c] = None
            continue
        if recs:
            conf = np.array([r[0] for r in recs], dtype=np.float32)
            istp = np.array([r[1] for r in recs], dtype=bool)
        else:
            conf = np.zeros(0, dtype=np.float32)
            istp = np.zeros(0, dtype=bool)

        f1s, ps, rs = [], [], []
        for t in grid:
            keep = conf >= t
            tp = int(np.count_nonzero(istp & keep))
            fp = int(np.count_nonzero(~istp & keep))
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / g
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
            ps.append(p)
            rs.append(r)
        out[c] = {"f1": np.array(f1s), "p": np.array(ps), "r": np.array(rs)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--min-conf", type=float, default=0.001)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--grid-step", type=float, default=0.01)
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "Refusing to tune on the frozen test set without --allow-test.\n"
            "Thresholds fitted on test would make it no longer held out.")

    img_dir, lbl_dir = resolve_split(args.data, args.split)
    print(f"Images: {img_dir}\nLabels: {lbl_dir}")

    records, n_gt = collect(args.weights, img_dir, lbl_dir, args.imgsz,
                            args.device, args.min_conf, args.match_iou)

    grid = np.arange(args.grid_step, 0.95 + 1e-9, args.grid_step)
    curves = sweep(records, n_gt, grid)

    present = [c for c in range(len(CLASS_NAMES)) if curves[c] is not None]
    if not present:
        raise SystemExit("No ground truth found for any class.")

    # Shared threshold, chosen the way Ultralytics does it: maximise mean F1.
    mean_f1 = np.mean([curves[c]["f1"] for c in present], axis=0)
    gi = int(np.argmax(mean_f1))
    global_t = float(grid[gi])

    rows, thresholds = [], {}
    for c in present:
        cur = curves[c]
        bi = int(np.argmax(cur["f1"]))
        rows.append({
            "class_id": c, "class": CLASS_NAMES[c], "support": n_gt[c],
            "shared_threshold": round(global_t, 4),
            "shared_f1": round(float(cur["f1"][gi]), 4),
            "shared_p": round(float(cur["p"][gi]), 4),
            "shared_r": round(float(cur["r"][gi]), 4),
            "best_threshold": round(float(grid[bi]), 4),
            "best_f1": round(float(cur["f1"][bi]), 4),
            "best_p": round(float(cur["p"][bi]), 4),
            "best_r": round(float(cur["r"][bi]), 4),
            "delta_f1": round(float(cur["f1"][bi] - cur["f1"][gi]), 4),
            "low_support": n_gt[c] < LOW_SUPPORT,
        })
        thresholds[CLASS_NAMES[c]] = round(float(grid[bi]), 4)

    mean_shared = float(np.mean([r["shared_f1"] for r in rows]))
    mean_best = float(np.mean([r["best_f1"] for r in rows]))

    print(f"\nShared threshold (Ultralytics-style): {global_t:.2f}\n")
    hdr = (f"{'class':12s} {'sup':>6s} {'shared F1':>10s} {'P':>7s} {'R':>7s}"
           f" | {'best t':>7s} {'F1':>8s} {'P':>7s} {'R':>7s} {'ΔF1':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = "!" if r["low_support"] else " "
        print(f"{r['class']:11s}{mark} {r['support']:6d} {r['shared_f1']:10.4f} "
              f"{r['shared_p']:7.4f} {r['shared_r']:7.4f} | "
              f"{r['best_threshold']:7.2f} {r['best_f1']:8.4f} {r['best_p']:7.4f} "
              f"{r['best_r']:7.4f} {r['delta_f1']:+8.4f}")
    print("-" * len(hdr))
    print(f"{'mean F1':12s} {'':6s} {mean_shared:10.4f} {'':7s} {'':7s} | "
          f"{'':7s} {mean_best:8.4f} {'':7s} {'':7s} {mean_best-mean_shared:+8.4f}")
    print("\n! = fewer than "
          f"{LOW_SUPPORT} ground-truth boxes; the tuned threshold is fitted to "
          "few examples and may not carry over.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thresholds.json").write_text(json.dumps({
        "weights": args.weights,
        "data": str(args.data),
        "split": args.split,
        "match_iou": args.match_iou,
        "shared_threshold": round(global_t, 4),
        "mean_f1_shared": round(mean_shared, 4),
        "mean_f1_per_class_tuned": round(mean_best, 4),
        "mean_f1_gain": round(mean_best - mean_shared, 4),
        "per_class_thresholds": thresholds,
        "detail": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved → {args.out / 'thresholds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
