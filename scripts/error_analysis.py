#!/usr/bin/env python3
"""Error analysis for a detection model: categorize errors to find the real bottleneck.

Runs the model on a split, matches predictions to ground truth (greedy, IoU-based),
and labels every prediction / GT box as one of:
  true_positive     — correct class, IoU >= 0.75
  loose_box         — correct class, 0.50 <= IoU < 0.75  (explains high mAP50 / low mAP50-95)
  poor_localization — correct class, 0.10 <= IoU < 0.50  (pred exists but box off)
  wrong_class       — a GT is overlapped (IoU>=0.5) by a prediction of a DIFFERENT class
  missed_object     — GT with no correct-class prediction at IoU>=0.5 (false negative)
  false_positive    — prediction not overlapping any GT (IoU<0.5, not poor/ wrong)

Outputs (under --out):
  error_summary.csv       one row per error/pred/GT
  per_class_summary.csv   per class: n_gt, TP, loose, poor, missed, wrong, FP, recall
  per_size_summary.csv    per COCO size bucket: recall small/medium/large
  per_dataset_summary.csv one-row summary for this run
  visualizations/         ~40 representative error images (GT green, pred red)

Example:
  python scripts/error_analysis.py \\
    --weights /workspace/runs/final_yolov8n_phase2_v3_rare_960/weights/best.pt \\
    --data configs/xwod.yaml --split test --imgsz 960 \\
    --out /workspace/runs/error_analysis/v3_960/xwod_test
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
RARE = {"bicycle", "motorcycle", "bus"}
IOU_TP, IOU_TIGHT, IOU_POOR = 0.50, 0.75, 0.10
ERROR_TYPES = ["true_positive", "loose_box", "poor_localization",
               "wrong_class", "missed_object", "false_positive"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes. a:(N,4) b:(M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def box_area(box) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _fmt(box) -> str:
    return "" if box is None else " ".join(str(int(round(v))) for v in box)


# ---------------------------------------------------------------------------
# Per-image error labelling
# ---------------------------------------------------------------------------


def analyze_image(gt: list[tuple], preds: list[tuple]) -> list[dict]:
    """gt: [(cls, x1,y1,x2,y2)]; preds: [(cls, conf, x1,y1,x2,y2)] sorted by conf desc."""
    rows: list[dict] = []
    gt_boxes = np.array([g[1:] for g in gt], float).reshape(-1, 4)
    gt_cls = np.array([g[0] for g in gt], int)
    pr_boxes = np.array([p[2:] for p in preds], float).reshape(-1, 4)
    pr_cls = np.array([p[0] for p in preds], int)
    pr_conf = np.array([p[1] for p in preds], float)

    M = iou_matrix(pr_boxes, gt_boxes)  # (P, G)
    gt_matched = np.zeros(len(gt), bool)
    pr_used = np.zeros(len(preds), bool)

    def add(cls, et, gtb, prb, iou, conf, area, note=""):
        rows.append({
            "class_id": int(cls), "class_name": TARGET_CLASSES[int(cls)],
            "error_type": et, "gt_box": _fmt(gtb), "pred_box": _fmt(prb),
            "iou": round(float(iou), 4), "confidence": ("" if conf is None else round(float(conf), 4)),
            "box_area": int(area), "size_bucket": size_bucket(area), "note": note,
        })

    # 1) Correct-class matches (TP / loose_box), greedy by confidence order
    for pi in range(len(preds)):
        cand = [gi for gi in range(len(gt))
                if not gt_matched[gi] and gt_cls[gi] == pr_cls[pi] and M[pi, gi] >= IOU_TP]
        if cand:
            gi = max(cand, key=lambda g: M[pi, g])
            gt_matched[gi] = True
            pr_used[pi] = True
            iou = M[pi, gi]
            et = "true_positive" if iou >= IOU_TIGHT else "loose_box"
            add(pr_cls[pi], et, gt_boxes[gi], pr_boxes[pi], iou, pr_conf[pi], box_area(gt_boxes[gi]))

    # 2) Remaining predictions -> wrong_class / poor_localization / false_positive
    for pi in range(len(preds)):
        if pr_used[pi]:
            continue
        if len(gt):
            gi = int(np.argmax(M[pi]))
            best = M[pi, gi]
        else:
            gi, best = -1, 0.0
        if gi >= 0 and best >= IOU_TP and gt_cls[gi] != pr_cls[pi]:
            add(pr_cls[pi], "wrong_class", gt_boxes[gi], pr_boxes[pi], best, pr_conf[pi],
                box_area(pr_boxes[pi]), note=f"gt_class={TARGET_CLASSES[gt_cls[gi]]}")
        elif gi >= 0 and IOU_POOR <= best < IOU_TP and gt_cls[gi] == pr_cls[pi]:
            add(pr_cls[pi], "poor_localization", gt_boxes[gi], pr_boxes[pi], best, pr_conf[pi],
                box_area(pr_boxes[pi]))
        else:
            add(pr_cls[pi], "false_positive", None, pr_boxes[pi], best, pr_conf[pi],
                box_area(pr_boxes[pi]))

    # 3) Unmatched GTs -> missed_object
    for gi in range(len(gt)):
        if not gt_matched[gi]:
            add(gt_cls[gi], "missed_object", gt_boxes[gi], None, 0.0, None, box_area(gt_boxes[gi]))

    return rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gt(label_path: Path, w: int, h: int) -> list[tuple]:
    gt: list[tuple] = []
    if not label_path.exists():
        return gt
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
        gt.append((cls, x1, y1, x2, y2))
    return gt


def resolve_images(data_yaml: str, split: str) -> list[Path]:
    from ultralytics.data.utils import check_det_dataset
    d = check_det_dataset(data_yaml)
    p = d[split]
    paths = p if isinstance(p, list) else [p]
    imgs: list[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for item in paths:
        item = Path(item)
        if item.is_dir():
            imgs += [q for q in sorted(item.rglob("*")) if q.suffix.lower() in exts]
        elif item.suffix.lower() == ".txt":
            imgs += [Path(l.strip()) for l in item.read_text().splitlines() if l.strip()]
    return imgs


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw(img_path: Path, gt: list[tuple], preds: list[tuple], out_path: Path) -> None:
    im = cv2.imread(str(img_path))
    if im is None:
        return
    for cls, x1, y1, x2, y2 in gt:  # GT green
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(im, TARGET_CLASSES[cls], (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    for cls, conf, x1, y1, x2, y2 in preds:  # pred red
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 230), 2)
        cv2.putText(im, f"{TARGET_CLASSES[cls]} {conf:.2f}", (int(x1), int(y2) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence (operating point)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-vis", type=int, default=40, help="Number of error images to save")
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset_name = out.name

    try:
        from dawn_ablation.common import register_custom_modules
        register_custom_modules()
    except Exception:
        pass
    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = resolve_images(args.data, args.split)
    print(f"[{dataset_name}] {len(images)} images | conf={args.conf} imgsz={args.imgsz}")

    all_rows: list[dict] = []
    per_image: dict[str, dict] = {}   # img -> {gt, preds, n_err, has_rare_err}

    for k, img in enumerate(images):
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = load_gt(label_for(img), w, h)
        r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          device=args.device, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append((int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))
        preds.sort(key=lambda p: -p[1])

        rows = analyze_image(gt, preds)
        for row in rows:
            row["dataset"] = dataset_name
            row["image_path"] = str(img)
        all_rows.extend(rows)

        err = [x for x in rows if x["error_type"] != "true_positive"]
        has_rare_err = any(x["class_name"] in RARE and x["error_type"] in
                           ("missed_object", "wrong_class", "poor_localization") for x in err)
        per_image[str(img)] = {"gt": gt, "preds": preds, "n_err": len(err), "has_rare_err": has_rare_err}

        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(images)}")

    # ── error_summary.csv ──
    cols = ["dataset", "image_path", "class_id", "class_name", "error_type",
            "gt_box", "pred_box", "iou", "confidence", "box_area", "size_bucket", "note"]
    with (out / "error_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for row in all_rows:
            wr.writerow({c: row.get(c, "") for c in cols})

    # ── aggregates ──
    et_by_class = defaultdict(Counter)      # class -> error_type -> n
    gt_by_class = Counter()
    gt_by_size = Counter()
    tp_by_size = Counter()                  # correct-class match (TP+loose)
    for row in all_rows:
        et = row["error_type"]; cn = row["class_name"]
        et_by_class[cn][et] += 1
        if et in ("true_positive", "loose_box", "missed_object", "poor_localization"):
            # GT-side rows (poor_localization is pred-side but its GT stays missed → counted there)
            pass
    # GT-side counts: TP+loose+missed+wrong are the GT outcomes; recompute from GT perspective
    # A GT is counted once: TP/loose (matched) OR missed (includes wrong-class overlaps).
    for row in all_rows:
        et = row["error_type"]; cn = row["class_name"]
        if et in ("true_positive", "loose_box", "missed_object"):
            gt_by_class[cn] += 1
            gt_by_size[row["size_bucket"]] += 1
            if et in ("true_positive", "loose_box"):
                tp_by_size[row["size_bucket"]] += 1

    totals = Counter(row["error_type"] for row in all_rows)
    TP = totals["true_positive"] + totals["loose_box"]
    FP = totals["false_positive"]
    FN = totals["missed_object"]

    # ── per_class_summary.csv ──
    with (out / "per_class_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "n_gt", "true_positive", "loose_box", "poor_localization",
                     "missed_object", "wrong_class", "false_positive", "recall"])
        for c in TARGET_CLASSES:
            e = et_by_class[c]
            n_gt = e["true_positive"] + e["loose_box"] + e["missed_object"]
            recall = (e["true_positive"] + e["loose_box"]) / n_gt if n_gt else 0.0
            wr.writerow([c, n_gt, e["true_positive"], e["loose_box"], e["poor_localization"],
                         e["missed_object"], e["wrong_class"], e["false_positive"], round(recall, 4)])

    # ── per_size_summary.csv ──
    with (out / "per_size_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["size_bucket", "n_gt", "matched", "recall"])
        for s in ("small", "medium", "large"):
            n = gt_by_size[s]; m = tp_by_size[s]
            wr.writerow([s, n, m, round(m / n, 4) if n else 0.0])

    # ── per_dataset_summary.csv ──
    with (out / "per_dataset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "TP", "loose_box", "poor_localization", "missed_object",
                     "wrong_class", "false_positive", "recall", "loose_ratio"])
        n_gt_total = TP + FN
        loose_ratio = totals["loose_box"] / TP if TP else 0.0
        wr.writerow([dataset_name, totals["true_positive"], totals["loose_box"],
                     totals["poor_localization"], FN, totals["wrong_class"], FP,
                     round(TP / n_gt_total, 4) if n_gt_total else 0.0, round(loose_ratio, 4)])

    # ── visualizations ──
    ranked = sorted(per_image.items(),
                    key=lambda kv: (kv[1]["has_rare_err"], kv[1]["n_err"]), reverse=True)
    vis_dir = out / "visualizations"
    for i, (img, d) in enumerate(ranked[:args.max_vis]):
        if d["n_err"] == 0:
            break
        draw(Path(img), d["gt"], d["preds"], vis_dir / f"{i:02d}_{Path(img).stem}.jpg")

    # ── console summary ──
    print("\n" + "=" * 56)
    print(f"ERROR ANALYSIS — {dataset_name}")
    print("=" * 56)
    print(f"TP={TP}  loose_box={totals['loose_box']}  poor_loc={totals['poor_localization']}  "
          f"missed(FN)={FN}  wrong_class={totals['wrong_class']}  FP={FP}")
    print(f"loose_box / TP = {loose_ratio:.1%}  (cao = box không khít → giải thích mAP50-95 thấp)")
    print("\nRecall theo kích thước:")
    for s in ("small", "medium", "large"):
        n = gt_by_size[s]
        print(f"  {s:<7} recall={ (tp_by_size[s]/n if n else 0):.3f}  (n_gt={n})")
    print("\nPer-class (n_gt | missed | wrong | poor | loose | recall):")
    for c in TARGET_CLASSES:
        e = et_by_class[c]; n_gt = e["true_positive"] + e["loose_box"] + e["missed_object"]
        rec = (e["true_positive"] + e["loose_box"]) / n_gt if n_gt else 0
        star = " *" if c in RARE else ""
        print(f"  {c:<11}{n_gt:>5} |{e['missed_object']:>5} |{e['wrong_class']:>4} |"
              f"{e['poor_localization']:>4} |{e['loose_box']:>5} | {rec:.3f}{star}")
    print("\nTop 20 ảnh nhiều lỗi nhất:")
    for img, d in ranked[:20]:
        if d["n_err"] == 0:
            break
        print(f"  {d['n_err']:>3} lỗi {'[rare]' if d['has_rare_err'] else '      '} {Path(img).name}")
    print(f"\nOutput → {out}  (+ {min(args.max_vis, sum(1 for _,d in ranked if d['n_err']>0))} ảnh visualizations)")


if __name__ == "__main__":
    main()
