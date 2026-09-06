#!/usr/bin/env python3
"""Per-class F1 matrix across arms x datasets x splits, with val/test agreement check.

Answers three questions the ablation depends on:
  1. Do val and test rank the arms the same way? If they disagree, the split is
     unsound and no downstream conclusion holds.
  2. Per class and per dataset, where is the model weak?
  3. Which of those weaknesses are measured on enough ground truth to trust?

Held-out test sets are frozen. --split test requires --allow-test so it cannot
be touched by accident during config search.

Example:
  python scripts/eval_f1_matrix.py \
    --arm phase2=/workspace/runs/phase2_final_rtdetr/weights/best.pt \
    --arm a0r=/workspace/runs/p2_a0r_rtdetr/weights/best.pt \
    --arm a1_dino=/workspace/runs/p2_a1_dino_rtdetr/weights/best.pt \
    --dataset xwod=/workspace/datasets_noleak/xwod_6cls_yolo/dataset.yaml \
    --dataset acdc=/workspace/datasets_noleak/acdc_6cls_yolo/dataset.yaml \
    --split val \
    --out /workspace/runs/evals/f1_matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# Below this many ground-truth boxes a per-class F1 is too noisy to rank arms by.
LOW_SUPPORT = 100


def parse_kv(values: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--{what} expects name=path, got: {v!r}")
        name, path = v.split("=", 1)
        if name in out:
            raise SystemExit(f"Duplicate --{what} name: {name!r}")
        out[name] = path
    return out


def split_exists(data_yaml: Path, split: str) -> bool:
    """True when the yaml declares the split and its image dir is populated."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    rel = cfg.get(split)
    if not rel:
        return False
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    img_dir = base / rel
    if not img_dir.is_dir():
        return False
    return any(
        next(img_dir.rglob(pat), None) is not None
        for pat in ("*.jpg", "*.jpeg", "*.png")
    )


def count_support(data_yaml: Path, split: str) -> Counter:
    """Ground-truth boxes per class id for a split, read straight from labels."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    rel = cfg.get(split)
    if rel is None:
        return Counter()
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    img_dir = base / rel
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))
    if not lbl_dir.is_dir():
        return Counter()

    ct: Counter = Counter()
    for p in lbl_dir.rglob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ct[int(line.split()[0])] += 1
    return ct


def eval_one(weights: str, data_yaml: str, split: str, imgsz: int,
             batch: int, device: str, project: str, name: str) -> dict:
    """Run Ultralytics validation and pull out per-class P/R/F1/AP."""
    from dawn_ablation.common import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(weights)
    m = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch,
                  device=device, project=project, name=name, exist_ok=True,
                  verbose=False)
    box = m.box

    # ap_class_index lists only classes that actually appear; arrays are aligned
    # to it, not to the global class id space.
    idx = list(getattr(box, "ap_class_index", range(len(CLASS_NAMES))))
    per_class: dict[int, dict] = {}
    for i, cls_id in enumerate(idx):
        p = float(box.p[i]) if hasattr(box, "p") and i < len(box.p) else float("nan")
        r = float(box.r[i]) if hasattr(box, "r") and i < len(box.r) else float("nan")
        if hasattr(box, "f1") and i < len(box.f1):
            f1 = float(box.f1[i])
        else:
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[int(cls_id)] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "ap50": float(box.ap50[i]) if hasattr(box, "ap50") and i < len(box.ap50) else float("nan"),
            "ap50_95": float(box.ap[i]) if hasattr(box, "ap") and i < len(box.ap) else float("nan"),
        }

    return {
        "per_class": per_class,
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1": (2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) > 0 else 0.0,
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
    }


def fmt(x: float) -> str:
    return "  -  " if x != x else f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-class F1 matrix across arms and datasets")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=WEIGHTS")
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DATA_YAML")
    ap.add_argument("--split", nargs="+", default=["val"], choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true",
                    help="Required to evaluate on the frozen held-out test sets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if "test" in args.split and not args.allow_test:
        raise SystemExit(
            "Refusing to touch the frozen test sets without --allow-test.\n"
            "Test data is for the final, one-shot comparison after configs are fixed."
        )

    arms = parse_kv(args.arm, "arm")
    datasets = parse_kv(args.dataset, "dataset")

    for name, w in arms.items():
        if not Path(w).is_file():
            raise SystemExit(f"Arm {name!r}: weights not found: {w}")
    for name, d in datasets.items():
        if not Path(d).is_file():
            raise SystemExit(f"Dataset {name!r}: yaml not found: {d}")

    args.out.mkdir(parents=True, exist_ok=True)

    support: dict[tuple[str, str], Counter] = {}
    for ds_name, ds_yaml in datasets.items():
        for split in args.split:
            support[(ds_name, split)] = count_support(Path(ds_yaml), split)

    rows: list[dict] = []
    results: dict = defaultdict(dict)
    failures: list[dict] = []

    total = len(arms) * len(datasets) * len(args.split)
    n = 0
    for arm_name, weights in arms.items():
        for ds_name, ds_yaml in datasets.items():
            for split in args.split:
                n += 1
                tag = f"{arm_name}__{ds_name}__{split}"
                if not split_exists(Path(ds_yaml), split):
                    print(f"[{n}/{total}] {tag} — skipped (no '{split}' split)", flush=True)
                    continue
                print(f"[{n}/{total}] {tag}", flush=True)
                try:
                    res = eval_one(weights, ds_yaml, split, args.imgsz, args.batch,
                                   args.device, str(args.out / "runs"), tag)
                except Exception as exc:  # keep the rest of the matrix alive
                    print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                    failures.append({"arm": arm_name, "dataset": ds_name,
                                     "split": split, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                results[(arm_name, ds_name, split)] = res

                sup = support[(ds_name, split)]
                for cls_id, mm in sorted(res["per_class"].items()):
                    rows.append({
                        "arm": arm_name, "dataset": ds_name, "split": split,
                        "class_id": cls_id,
                        "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                        "support": sup.get(cls_id, 0),
                        **{k: round(v, 6) for k, v in mm.items()},
                    })
                rows.append({
                    "arm": arm_name, "dataset": ds_name, "split": split,
                    "class_id": -1, "class": "ALL",
                    "support": sum(sup.values()),
                    **{k: round(v, 6) for k, v in res["overall"].items()},
                })

    csv_path = args.out / "f1_matrix.csv"
    fields = ["arm", "dataset", "split", "class_id", "class", "support",
              "precision", "recall", "f1", "ap50", "ap50_95", "map50", "map50_95"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ── Per-class F1 table, one block per dataset+split ──────────────────────
    lines: list[str] = ["# Per-class F1 matrix", ""]
    arm_names = list(arms)
    for ds_name in datasets:
        for split in args.split:
            sup = support[(ds_name, split)]
            lines += [f"## {ds_name} / {split}", "",
                      "| class | support | " + " | ".join(arm_names) + " | best |",
                      "|---|---:|" + "---:|" * len(arm_names) + "---|"]
            for cls_id, cname in enumerate(CLASS_NAMES):
                vals = []
                for a in arm_names:
                    pc = results.get((a, ds_name, split), {}).get("per_class", {})
                    vals.append(pc.get(cls_id, {}).get("f1", float("nan")))
                real = [v for v in vals if v == v]
                best = arm_names[vals.index(max(real))] if real else "-"
                s = sup.get(cls_id, 0)
                mark = " ⚠" if 0 < s < LOW_SUPPORT else ("" if s else " (absent)")
                lines.append(f"| {cname}{mark} | {s} | "
                             + " | ".join(fmt(v) for v in vals) + f" | {best} |")
            ov = []
            for a in arm_names:
                ov.append(results.get((a, ds_name, split), {}).get("overall", {}).get("f1", float("nan")))
            real = [v for v in ov if v == v]
            best = arm_names[ov.index(max(real))] if real else "-"
            lines.append(f"| **ALL** | {sum(sup.values())} | "
                         + " | ".join(f"**{fmt(v)}**" for v in ov) + f" | **{best}** |")
            lines.append("")
    lines += [f"⚠ = fewer than {LOW_SUPPORT} ground-truth boxes; F1 too noisy to rank arms by.", ""]

    # ── val/test agreement: does the ranking survive the split? ──────────────
    inversions: list[dict] = []
    if len(arm_names) >= 2 and {"val", "test"} <= set(args.split):
        lines += ["## val ↔ test agreement", "",
                  "Question 1 of the plan: if an arm wins on test it must also win on val.",
                  "A disagreement means the two splits are not drawn from the same",
                  "distribution, and every comparison built on them is unsafe.", ""]
        for ds_name in datasets:
            for cls_id, cname in enumerate(CLASS_NAMES):
                def f1_of(a: str, sp: str) -> float:
                    return results.get((a, ds_name, sp), {}).get(
                        "per_class", {}).get(cls_id, {}).get("f1", float("nan"))

                for i in range(len(arm_names)):
                    for j in range(i + 1, len(arm_names)):
                        a, b = arm_names[i], arm_names[j]
                        dv, dt = f1_of(a, "val") - f1_of(b, "val"), f1_of(a, "test") - f1_of(b, "test")
                        if dv != dv or dt != dt or dv == 0 or dt == 0:
                            continue
                        if (dv > 0) != (dt > 0):
                            s_val = support[(ds_name, "val")].get(cls_id, 0)
                            s_test = support[(ds_name, "test")].get(cls_id, 0)
                            inversions.append({
                                "dataset": ds_name, "class": cname,
                                "arm_a": a, "arm_b": b,
                                "val_delta": round(dv, 4), "test_delta": round(dt, 4),
                                "support_val": s_val, "support_test": s_test,
                                "low_support": bool(s_val < LOW_SUPPORT or s_test < LOW_SUPPORT),
                            })
        if inversions:
            solid = [x for x in inversions if not x["low_support"]]
            lines += ["| dataset | class | pair | Δval | Δtest | sup val | sup test | trust |",
                      "|---|---|---|---:|---:|---:|---:|---|"]
            for x in inversions:
                lines.append(
                    f"| {x['dataset']} | {x['class']} | {x['arm_a']} vs {x['arm_b']} | "
                    f"{x['val_delta']:+.4f} | {x['test_delta']:+.4f} | "
                    f"{x['support_val']} | {x['support_test']} | "
                    + ("low support" if x["low_support"] else "**solid**") + " |")
            lines += ["", f"{len(inversions)} inversion(s), {len(solid)} on adequate support.", ""]
            if solid:
                lines += ["**Verdict: split unsound.** Ranking flips between val and test on "
                          "classes with enough ground truth to trust. Per the plan, re-split "
                          "val/test before drawing conclusions from either.", ""]
            else:
                lines += ["**Verdict: acceptable.** Every inversion sits on a low-support class, "
                          "which is measurement noise rather than distribution shift.", ""]
        else:
            lines += ["No inversions: val and test agree on every arm pair and class. "
                      "The split is safe to build on.", ""]
    elif len(args.split) == 1:
        lines += ["## val ↔ test agreement", "",
                  f"Skipped — only `{args.split[0]}` was evaluated. Re-run with "
                  "`--split val test --allow-test` to run this check.", ""]

    # ── Weakest classes, to aim the next retrieval round at ─────────────────
    lines += ["## Weakest classes", "",
              "Question 3 of the plan: which class to spend the next 5K on.", ""]
    ref = arm_names[-1]
    ref_split = "val" if "val" in args.split else args.split[0]
    weak: list[tuple] = []
    for ds_name in datasets:
        sup = support[(ds_name, ref_split)]
        pc = results.get((ref, ds_name, ref_split), {}).get("per_class", {})
        for cls_id, mm in pc.items():
            s = sup.get(cls_id, 0)
            if s > 0:
                weak.append((mm["f1"], ds_name, CLASS_NAMES[cls_id], s,
                             mm["precision"], mm["recall"]))
    weak.sort()
    lines += [f"Arm `{ref}` on `{ref_split}`, worst F1 first:", "",
              "| dataset | class | F1 | P | R | support |",
              "|---|---|---:|---:|---:|---:|"]
    for f1, ds_name, cname, s, p, r in weak[:15]:
        mark = " ⚠" if s < LOW_SUPPORT else ""
        lines.append(f"| {ds_name} | {cname}{mark} | {f1:.4f} | {p:.4f} | {r:.4f} | {s} |")
    lines += ["", "Read P vs R to tell the failure apart: low recall means the class is "
              "being missed and more data should help; low precision with high recall "
              "means confusion with another class, which more data will not fix.", ""]

    md_path = args.out / "f1_matrix.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (args.out / "inversions.json").write_text(
        json.dumps(inversions, indent=2), encoding="utf-8")

    if failures:
        (args.out / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8")
        print(f"\n{len(failures)} evaluation(s) failed — see {args.out / 'failures.json'}")
        for f in failures:
            print(f"  {f['arm']}/{f['dataset']}/{f['split']}: {f['error']}")

    print(f"\nCSV      → {csv_path}")
    print(f"Report   → {md_path}")
    if inversions:
        solid = [x for x in inversions if not x["low_support"]]
        print(f"\nval/test inversions: {len(inversions)} ({len(solid)} on adequate support)")
        if solid:
            print("SPLIT UNSOUND — re-split val/test before trusting any comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
