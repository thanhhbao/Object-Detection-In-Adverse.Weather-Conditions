#!/usr/bin/env python3
"""Compare whole-image inference against sliced (SAHI-style) inference.

The resolution sweep showed the model collapses when the *input canvas* is
resized away from the 640 it was trained on, while mild object magnification
(imgsz 960) helped. Slicing separates those two effects: every tile is fed at
exactly 640, so the canvas never changes, yet a 14 px object stays 14 px
instead of shrinking to 4.6 px.

Both modes run through identical matching and threshold-selection code, so the
two F1 numbers are directly comparable. Results are also broken down by object
size band, because slicing is expected to help small objects and may *hurt*
large ones that straddle a tile boundary.

Example:
  python scripts/eval_tiled_inference.py \
    --weights /workspace/runs/p2_a1_dino_rtdetr_retrieval/weights/best.pt \
    --data /workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
    --split val --tile 640 --overlap 0.2 \
    --out /workspace/runs/evals/tiled_acdc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from tune_conf_thresholds import (  # noqa: E402
    CLASS_NAMES, iou_matrix, load_gt, resolve_split,
)

BANDS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2),
         ("large", 96 ** 2, float("inf"))]
LOW_SUPPORT = 100


def band_of(area: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= area < hi:
            return name
    return "large"


def tile_origins(w: int, h: int, tile: int, overlap: float) -> list:
    """Top-left corners of a covering grid of exactly tile x tile windows.

    Edge windows are shifted back inside the image rather than truncated, so
    every tile is the full size the model expects and nothing needs resizing.
    """
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list:
    """Greedy NMS, indices kept, highest score first."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = rest[ious < thr]
    return keep


def predict_full(model, img, imgsz, device, conf):
    r = next(iter(model.predict(source=[img], stream=True, conf=conf,
                                imgsz=imgsz, device=device, verbose=False)))
    b = r.boxes
    if b is None or len(b) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, int),
                np.zeros(0, np.float32))
    return (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
            b.conf.cpu().numpy())


def predict_tiled(model, img, tile, overlap, device, conf, batch,
                  nms_iou, add_full):
    """Detections in full-image coordinates, merged across tiles with NMS."""
    h, w = img.shape[:2]
    if w <= tile and h <= tile:
        return predict_full(model, img, tile, device, conf)

    origins = tile_origins(w, h, tile, overlap)
    all_b, all_c, all_s = [], [], []

    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        crops = [img[y:y + tile, x:x + tile] for x, y in chunk]
        for (ox, oy), r in zip(chunk, model.predict(
                source=crops, stream=True, conf=conf, imgsz=tile,
                device=device, verbose=False)):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().copy()
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            all_b.append(xyxy)
            all_c.append(b.cls.cpu().numpy().astype(int))
            all_s.append(b.conf.cpu().numpy())

    if add_full:
        fb, fc, fs = predict_full(model, img, tile, device, conf)
        if len(fb):
            all_b.append(fb)
            all_c.append(fc)
            all_s.append(fs)

    if not all_b:
        return (np.zeros((0, 4), np.float32), np.zeros(0, int),
                np.zeros(0, np.float32))

    boxes = np.concatenate(all_b).astype(np.float32)
    cls = np.concatenate(all_c)
    scores = np.concatenate(all_s).astype(np.float32)

    keep_all = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        keep_all.extend(idx[k] for k in nms(boxes[idx], scores[idx], nms_iou))
    keep = np.array(sorted(keep_all), dtype=int)
    return boxes[keep], cls[keep], scores[keep]


def collect(model, imgs, lbl_dir, mode, args):
    """Records of (conf, is_tp) per class, plus GT counts overall and by band."""
    import cv2

    records: dict = defaultdict(list)
    n_gt: dict = defaultdict(int)
    band_gt: dict = defaultdict(int)
    band_hit: dict = defaultdict(list)   # (class, band) -> [conf of matched GT]

    for k, path in enumerate(imgs, 1):
        if k % 100 == 0:
            print(f"    {mode}: {k}/{len(imgs)}", flush=True)
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_gt(lbl_dir / f"{path.stem}.txt", w, h)

        if mode == "full":
            pb, pc, ps = predict_full(model, img, args.imgsz, args.device,
                                      args.min_conf)
        else:
            pb, pc, ps = predict_tiled(model, img, args.tile, args.overlap,
                                       args.device, args.min_conf, args.batch,
                                       args.nms_iou, args.add_full_image)

        for c in range(len(CLASS_NAMES)):
            gm = gt_cls == c
            gb = gt_boxes[gm]
            n_gt[c] += len(gb)
            areas = [(gb[j][2] - gb[j][0]) * (gb[j][3] - gb[j][1])
                     for j in range(len(gb))]
            for a in areas:
                band_gt[(c, band_of(float(a)))] += 1

            m = pc == c
            b_, s_ = pb[m], ps[m]
            order = np.argsort(-s_)
            b_, s_ = b_[order], s_[order]

            taken = np.full(len(gb), -1.0)   # matched GT -> conf of its match
            ious = iou_matrix(b_, gb)
            for i in range(len(b_)):
                tp = False
                if len(gb):
                    row = ious[i].copy()
                    row[taken >= 0] = -1.0
                    j = int(np.argmax(row)) if row.size else -1
                    if j >= 0 and row[j] >= args.match_iou:
                        taken[j] = float(s_[i])
                        tp = True
                records[c].append((float(s_[i]), tp))
            for j in range(len(gb)):
                if taken[j] >= 0:
                    band_hit[(c, band_of(float(areas[j])))].append(taken[j])

    return records, n_gt, band_gt, band_hit


def best_f1(records: dict, n_gt: dict, grid: np.ndarray) -> dict:
    out = {}
    for c in range(len(CLASS_NAMES)):
        g = n_gt.get(c, 0)
        if g == 0:
            continue
        recs = records.get(c, [])
        conf = np.array([r[0] for r in recs], dtype=np.float64)
        istp = np.array([r[1] for r in recs], dtype=bool)
        best = (-1.0, 0.0, 0.0, 0.0)
        for t in grid:
            keep = conf >= t
            tp = int(np.count_nonzero(istp & keep))
            fp = int(np.count_nonzero(~istp & keep))
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / g
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            if f > best[0]:
                best = (f, p, r, float(t))
        out[c] = {"f1": best[0], "precision": best[1], "recall": best[2],
                  "threshold": best[3], "support": g}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--add-full-image", action="store_true",
                    help="Also run whole-image inference and merge it in")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Input size for the whole-image baseline")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8, help="Tiles per call")
    ap.add_argument("--min-conf", type=float, default=0.001)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--grid-step", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0,
                    help="Use only the first N images (smoke test)")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to touch the frozen test set without "
                         "--allow-test.")

    img_dir, lbl_dir = resolve_split(args.data, args.split)
    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        raise SystemExit(f"No images under {img_dir}")

    import cv2
    probe = cv2.imread(str(imgs[0]))
    ph, pw = probe.shape[:2]
    n_tiles = len(tile_origins(pw, ph, args.tile, args.overlap))
    print(f"{len(imgs)} images, first is {pw}x{ph}")
    print(f"tile={args.tile} overlap={args.overlap} -> {n_tiles} tiles/image")
    print(f"whole-image baseline at imgsz={args.imgsz}\n")

    from dawn_ablation.common import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO
    model = YOLO(args.weights)

    grid = np.arange(args.grid_step, 0.95 + 1e-9, args.grid_step)
    results, bands = {}, {}
    for mode in ("full", "tiled"):
        print(f"[{mode}]")
        rec, ngt, bgt, bhit = collect(model, imgs, lbl_dir, mode, args)
        results[mode] = best_f1(rec, ngt, grid)
        bands[mode] = (bgt, bhit)

    # ── per class ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("F1 THEO LOP — toan anh vs cat o")
    print("=" * 78)
    hdr = (f"{'lop':12s} {'sup':>6s} {'toan anh':>10s} {'cat o':>9s} "
           f"{'delta':>9s} {'R toan anh':>11s} {'R cat o':>9s}")
    print(hdr)
    print("-" * len(hdr))
    fa, ta = [], []
    for c in range(len(CLASS_NAMES)):
        a, b = results["full"].get(c), results["tiled"].get(c)
        if not a or not b:
            continue
        fa.append(a["f1"])
        ta.append(b["f1"])
        mark = "!" if a["support"] < LOW_SUPPORT else " "
        print(f"{CLASS_NAMES[c]:11s}{mark} {a['support']:6d} {a['f1']:10.4f} "
              f"{b['f1']:9.4f} {b['f1']-a['f1']:+9.4f} {a['recall']:11.3f} "
              f"{b['recall']:9.3f}")
    print("-" * len(hdr))
    mf, mt = float(np.mean(fa)), float(np.mean(ta))
    print(f"{'F1 trung binh':12s} {'':6s} {mf:10.4f} {mt:9.4f} {mt-mf:+9.4f}")

    # ── per size band: does slicing hurt large objects? ─────────────────────
    print("\n" + "=" * 78)
    print("RECALL THEO DAI KICH THUOC (o nguong tot nhat cua tung che do)")
    print("=" * 78)
    band_rows = []
    print(f"\n{'lop':12s} {'dai':8s} {'GT':>6s} {'toan anh':>10s} "
          f"{'cat o':>9s} {'delta':>9s}")
    for c in range(len(CLASS_NAMES)):
        if c not in results["full"]:
            continue
        for band, _, _ in BANDS:
            bgt = bands["full"][0].get((c, band), 0)
            if bgt == 0:
                continue
            rf = sum(1 for s in bands["full"][1].get((c, band), [])
                     if s >= results["full"][c]["threshold"]) / bgt
            rt = sum(1 for s in bands["tiled"][1].get((c, band), [])
                     if s >= results["tiled"][c]["threshold"]) / bgt
            print(f"{CLASS_NAMES[c]:12s} {band:8s} {bgt:6d} {rf:10.3f} "
                  f"{rt:9.3f} {rt-rf:+9.3f}")
            band_rows.append({"class": CLASS_NAMES[c], "band": band, "gt": bgt,
                              "recall_full": round(rf, 4),
                              "recall_tiled": round(rt, 4),
                              "delta": round(rt - rf, 4)})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tiled_results.json").write_text(json.dumps({
        "data": str(args.data), "split": args.split, "images": len(imgs),
        "tile": args.tile, "overlap": args.overlap, "nms_iou": args.nms_iou,
        "add_full_image": args.add_full_image, "baseline_imgsz": args.imgsz,
        "tiles_per_image": n_tiles,
        "mean_f1_full": round(mf, 4), "mean_f1_tiled": round(mt, 4),
        "mean_f1_delta": round(mt - mf, 4),
        "per_class": {CLASS_NAMES[c]: {
            "full": {k: round(v, 4) for k, v in results["full"][c].items()},
            "tiled": {k: round(v, 4) for k, v in results["tiled"][c].items()},
        } for c in results["full"] if c in results["tiled"]},
        "by_size_band": band_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {args.out / 'tiled_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
