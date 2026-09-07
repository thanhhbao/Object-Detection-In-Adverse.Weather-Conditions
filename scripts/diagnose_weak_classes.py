#!/usr/bin/env python3
"""Find out why a class fails, before spending effort on a fix.

Four questions, one inference pass:

  1. Condition — does the class fail everywhere, or only at night / in snow?
     Aggregate mAP hides this: on ACDC the model matches XWOD on fog and rain
     and collapses at night, so "add weather data" would be the wrong fix.
  2. Object size — are the missed objects simply smaller here than in the
     domain where the class works? Scale is not fixed by more weather.
  3. Blind vs confused — for each missed ground truth, is there any detection
     of another class on top of it? Confusion needs separation, not more data.
  4. Training support — how many instances of the class does the train split
     actually contain?

Example:
  python scripts/diagnose_weak_classes.py \
    --weights /workspace/runs/p2_a1_dino_rtdetr_retrieval/weights/best.pt \
    --data /workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
    --split val --classes bicycle motorcycle \
    --compare-data /workspace/datasets_noleak/xwod_6cls_yolo/dataset.yaml \
    --out /workspace/runs/evals/diag_acdc
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from tune_conf_thresholds import (  # noqa: E402
    CLASS_NAMES, iou_matrix, load_gt, resolve_split,
)

# COCO size bands, on box area in original pixels.
BANDS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2),
         ("large", 96 ** 2, float("inf"))]
CONDITION_HINTS = ["fog", "night", "rain", "snow", "sand", "dust",
                   "flooding", "wildfire", "tornado", "clear"]


def conditions_from_manifest(root: Path, split: str) -> dict[str, str]:
    """image stem -> condition, read from the dataset manifest when present."""
    for name in ("manifest.csv", "manifest_split.csv"):
        mf = root / name
        if not mf.is_file():
            continue
        out: dict[str, str] = {}
        with mf.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            img_col = next((c for c in ("image", "output_image", "file", "path")
                            if c in fields), None)
            if img_col is None or "weather" not in fields:
                continue
            for row in reader:
                if "split" in fields and row.get("split") not in (split, None, ""):
                    continue
                out[Path(row[img_col]).stem] = row["weather"]
        if out:
            print(f"  conditions: {mf.name} ({len(out)} entries)")
            return out
    return {}


# Split names, which must never be read as a condition: "train" contains "rain".
SPLIT_WORDS = {"train", "val", "valid", "validation", "test"}


def condition_from_name(stem: str) -> str:
    """Condition from the filename, matched on whole tokens.

    Substring matching is unsafe here — "flooding_train_0007" would report
    "rain" because the split name contains it.
    """
    for tok in re.split(r"[^a-z0-9]+", stem.lower()):
        if not tok or tok in SPLIT_WORDS:
            continue
        for hint in CONDITION_HINTS:
            if tok == hint or tok.startswith(hint):
                return hint
    return "unknown"


def count_train_support(data_yaml: Path) -> Counter:
    try:
        img_dir, lbl_dir = resolve_split(data_yaml, "train")
    except SystemExit:
        return Counter()
    ct: Counter = Counter()
    for p in lbl_dir.rglob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ct[int(line.split()[0])] += 1
    return ct


def gt_box_areas(data_yaml: Path, split: str) -> dict[int, list]:
    """Ground-truth box areas per class, in pixels, without running the model."""
    from PIL import Image

    img_dir, lbl_dir = resolve_split(data_yaml, split)
    sizes: dict[str, tuple] = {}
    out: dict[int, list] = defaultdict(list)
    for lbl in lbl_dir.rglob("*.txt"):
        stem = lbl.stem
        img = next((p for ext in (".jpg", ".jpeg", ".png")
                    for p in [img_dir / f"{stem}{ext}"] if p.is_file()), None)
        if img is None:
            hits = list(img_dir.rglob(f"{stem}.*"))
            img = hits[0] if hits else None
        if img is None:
            continue
        key = str(img)
        if key not in sizes:
            with Image.open(img) as im:
                sizes[key] = im.size
        w, h = sizes[key]
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            out[int(parts[0])].append(float(parts[3]) * w * float(parts[4]) * h)
    return out


def band_of(area: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= area < hi:
            return name
    return "large"


def run(weights, data_yaml, split, imgsz, device, conf, match_iou, batch):
    """One inference pass; returns everything the four analyses need."""
    from dawn_ablation.common import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    img_dir, lbl_dir = resolve_split(data_yaml, split)
    root = Path(data_yaml).parent
    cond_map = conditions_from_manifest(root, split)

    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"  {len(imgs)} images, conf>={conf}, IoU>={match_iou}")

    model = YOLO(weights)
    # per (class, condition): tp / fp / fn
    cell: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    # per (class, size band): matched / total
    size_hit: dict = defaultdict(lambda: {"hit": 0, "n": 0})
    # missed ground truth: blind, or covered by another class?
    miss_kind: dict = defaultdict(Counter)

    def stream():
        for i in range(0, len(imgs), batch):
            yield from model.predict(source=[str(p) for p in imgs[i:i + batch]],
                                     stream=True, conf=conf, imgsz=imgsz,
                                     device=device, verbose=False)

    done = 0
    for res in stream():
        done += 1
        if done % 500 == 0:
            print(f"    {done}/{len(imgs)}", flush=True)

        h, w = res.orig_shape
        stem = Path(res.path).stem
        cond = cond_map.get(stem) or condition_from_name(stem)

        gt_boxes, gt_cls = load_gt(lbl_dir / f"{stem}.txt", w, h)
        b = res.boxes
        if b is not None and len(b):
            p_xyxy = b.xyxy.cpu().numpy()
            p_cls = b.cls.cpu().numpy().astype(int)
            p_conf = b.conf.cpu().numpy()
        else:
            p_xyxy = np.zeros((0, 4), dtype=np.float32)
            p_cls = np.zeros(0, dtype=int)
            p_conf = np.zeros(0, dtype=np.float32)

        for c in range(len(CLASS_NAMES)):
            gm = gt_cls == c
            gb = gt_boxes[gm]
            pm = p_cls == c
            pb, pc = p_xyxy[pm], p_conf[pm]
            order = np.argsort(-pc)
            pb = pb[order]

            taken = np.zeros(len(gb), dtype=bool)
            ious = iou_matrix(pb, gb)
            for i in range(len(pb)):
                if len(gb):
                    row = ious[i].copy()
                    row[taken] = -1.0
                    j = int(np.argmax(row)) if row.size else -1
                    if j >= 0 and row[j] >= match_iou:
                        taken[j] = True
                        cell[(c, cond)]["tp"] += 1
                        continue
                cell[(c, cond)]["fp"] += 1

            for j in range(len(gb)):
                area = float((gb[j][2] - gb[j][0]) * (gb[j][3] - gb[j][1]))
                band = band_of(area)
                size_hit[(c, band)]["n"] += 1
                if taken[j]:
                    size_hit[(c, band)]["hit"] += 1
                    continue
                cell[(c, cond)]["fn"] += 1
                # Missed: is anything else sitting on it?
                if len(p_xyxy):
                    o = iou_matrix(gb[j:j + 1], p_xyxy)[0]
                    k = int(np.argmax(o))
                    if o[k] >= match_iou:
                        miss_kind[c][f"as_{CLASS_NAMES[int(p_cls[k])]}"] += 1
                        continue
                miss_kind[c]["blind"] += 1

    return cell, size_hit, miss_kind


def prf(d: dict) -> tuple:
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--compare-data", type=Path, default=None,
                    help="Second dataset, for the box-size comparison only")
    ap.add_argument("--classes", nargs="+", default=["bicycle", "motorcycle"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to diagnose on the frozen test set without "
                         "--allow-test.")

    focus = [CLASS_NAMES.index(c) for c in args.classes if c in CLASS_NAMES]
    if not focus:
        raise SystemExit(f"--classes must be from {CLASS_NAMES}")

    print(f"Diagnosing {args.classes} on {args.data} / {args.split}")
    cell, size_hit, miss_kind = run(args.weights, args.data, args.split,
                                    args.imgsz, args.device, args.conf,
                                    args.match_iou, args.batch)

    report: dict = {"data": str(args.data), "split": args.split,
                    "conf": args.conf, "match_iou": args.match_iou}

    # ── 1. class x condition ────────────────────────────────────────────────
    conds = sorted({c for _, c in cell})
    print("\n" + "=" * 72)
    print("1. F1 THEO LOP x DIEU KIEN")
    print("=" * 72)
    per_cond = {}
    for c in range(len(CLASS_NAMES)):
        rows = []
        for cond in conds:
            d = cell.get((c, cond))
            if not d or (d["tp"] + d["fn"]) == 0:
                continue
            p, r, f = prf(d)
            rows.append((cond, d["tp"] + d["fn"], p, r, f))
        if not rows:
            continue
        star = " <<<" if c in focus else ""
        print(f"\n{CLASS_NAMES[c]}{star}")
        print(f"  {'dieu kien':12s} {'GT':>6s} {'P':>7s} {'R':>7s} {'F1':>7s}")
        for cond, n, p, r, f in sorted(rows, key=lambda x: x[4]):
            print(f"  {cond:12s} {n:6d} {p:7.3f} {r:7.3f} {f:7.3f}")
        per_cond[CLASS_NAMES[c]] = [
            {"condition": cond, "gt": n, "precision": round(p, 4),
             "recall": round(r, 4), "f1": round(f, 4)}
            for cond, n, p, r, f in rows]
    report["per_condition"] = per_cond

    # ── 2. recall by object size ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("2. RECALL THEO KICH THUOC VAT THE")
    print("=" * 72)
    print(f"\n{'lop':12s} " + " ".join(f"{b:>18s}" for b, _, _ in BANDS))
    size_tbl = {}
    for c in range(len(CLASS_NAMES)):
        cells, entry = [], {}
        for band, _, _ in BANDS:
            d = size_hit.get((c, band), {"hit": 0, "n": 0})
            if d["n"]:
                rec = d["hit"] / d["n"]
                cells.append(f"{rec:.3f} (n={d['n']})".rjust(18))
                entry[band] = {"recall": round(rec, 4), "n": d["n"]}
            else:
                cells.append("—".rjust(18))
        if entry:
            star = " <" if c in focus else "  "
            print(f"{CLASS_NAMES[c]:12s}{star}" + " ".join(cells))
            size_tbl[CLASS_NAMES[c]] = entry
    report["recall_by_size"] = size_tbl

    # ── 3. blind vs confused ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("3. BO SOT: KHONG THAY HAY NHAM LOP?")
    print("=" * 72)
    miss_tbl = {}
    for c in focus:
        ctr = miss_kind.get(c)
        if not ctr:
            continue
        total = sum(ctr.values())
        print(f"\n{CLASS_NAMES[c]} — {total} GT bi bo sot")
        for k, v in ctr.most_common():
            label = "khong phat hien gi" if k == "blind" else f"co hop lop {k[3:]}"
            print(f"  {label:28s} {v:5d}  {v/total*100:5.1f}%")
        miss_tbl[CLASS_NAMES[c]] = {"total": total, **dict(ctr)}
    report["missed_breakdown"] = miss_tbl

    # ── 4. box size + train support, this dataset vs a reference ────────────
    print("\n" + "=" * 72)
    print("4. KICH THUOC HOP GT VA SO MAU HUAN LUYEN")
    print("=" * 72)
    ds_list = [(args.data, "này")]
    if args.compare_data:
        ds_list.append((args.compare_data, "so sánh"))
    size_cmp, sup_cmp = {}, {}
    for dy, tag in ds_list:
        areas = gt_box_areas(dy, args.split)
        sup = count_train_support(dy)
        name = Path(dy).parent.name
        print(f"\n{name} ({tag})")
        print(f"  {'lop':12s} {'GT val':>8s} {'canh trung vi':>14s} "
              f"{'% nho':>8s} {'instance train':>15s}")
        for c in range(len(CLASS_NAMES)):
            a = areas.get(c, [])
            if not a:
                continue
            arr = np.array(a)
            med_side = float(np.sqrt(np.median(arr)))
            frac_small = float(np.mean(arr < 32 ** 2))
            star = " <" if c in focus else "  "
            print(f"  {CLASS_NAMES[c]:12s}{star}{len(a):6d} {med_side:14.1f} "
                  f"{frac_small*100:7.1f}% {sup.get(c,0):15d}")
            size_cmp.setdefault(name, {})[CLASS_NAMES[c]] = {
                "n_val": len(a), "median_side_px": round(med_side, 1),
                "frac_small": round(frac_small, 4)}
            sup_cmp.setdefault(name, {})[CLASS_NAMES[c]] = sup.get(c, 0)
    report["box_size"] = size_cmp
    report["train_support"] = sup_cmp

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2,
                                                        ensure_ascii=False),
                                             encoding="utf-8")
    print(f"\nSaved → {args.out / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
