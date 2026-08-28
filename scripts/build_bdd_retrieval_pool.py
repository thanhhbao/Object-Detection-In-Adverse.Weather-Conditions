#!/usr/bin/env python3
"""Build the BDD remaining-pool dataset (full BDD train minus BDD30K used).

Only images from the full BDD train split that do NOT appear in any BDD30K used
split (train/val/test) are written to the pool.  This guarantees zero overlap with
the training data already used in Phase 2, so retrieved images cannot cause leakage.

HARD INVARIANTS (RuntimeError if violated):
  overlap_with_used_train == 0
  overlap_with_used_val   == 0
  overlap_with_used_test  == 0

Usage:
  python scripts/build_bdd_retrieval_pool.py \\
    --full-root  /data/bdd100k_6cls_full_yolo \\
    --used-root  /data/bdd100k_6cls_yolo \\
    --out-root   /data/bdd_remaining_pool \\
    --mode symlink --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build BDD remaining retrieval pool")
    ap.add_argument("--full-root", type=Path, required=True,
                    help="BDD full train root (prepare_bdd100k_yolo.py --full output)")
    ap.add_argument("--used-root", type=Path, required=True,
                    help="Official BDD30K already used in Phase 2 project")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output directory for the candidate pool")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--clean", action="store_true", help="Remove --out-root if it exists")
    return ap.parse_args()


def collect_basenames(root: Path, split: str) -> set[str]:
    """Collect image basenames from root/images/<split> that exist."""
    img_dir = root / "images" / split
    if not img_dir.exists():
        return set()
    names: set[str] = set()
    for p in img_dir.iterdir():
        if p.suffix.lower() in IMAGE_SUFFIXES:
            names.add(p.name)
    return names


def place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    args = parse_args()
    full_root = args.full_root.resolve()
    used_root = args.used_root.resolve()
    out_root = args.out_root.resolve()

    if args.clean and out_root.exists():
        print(f"Removing existing {out_root} ...")
        shutil.rmtree(out_root)

    # ── Collect used names ──────────────────────────────────────────────────
    used_train = collect_basenames(used_root, "train")
    used_val = collect_basenames(used_root, "val")
    used_test = collect_basenames(used_root, "test")
    excluded_names = used_train | used_val | used_test

    print(f"Used BDD names — train: {len(used_train)}, val: {len(used_val)}, test: {len(used_test)}")
    print(f"Total excluded: {len(excluded_names)}")

    # ── Candidate pairs from full_root/images/train ─────────────────────────
    full_img_dir = full_root / "images" / "train"
    full_lbl_dir = full_root / "labels" / "train"

    if not full_img_dir.exists():
        raise RuntimeError(f"Full BDD train images not found: {full_img_dir}")

    all_pairs: list[tuple[Path, Path]] = []
    for img in sorted(full_img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = full_lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        all_pairs.append((img, lbl))

    full_train_count = len(all_pairs)
    print(f"Full BDD train pairs: {full_train_count}")

    # ── Filter candidates ────────────────────────────────────────────────────
    candidates: list[tuple[Path, Path]] = []
    for img, lbl in all_pairs:
        if img.name not in excluded_names:
            candidates.append((img, lbl))

    # ── Overlap breakdown for stats ─────────────────────────────────────────
    # Overlap = how many CANDIDATE images appear in each used split.
    # After filtering, this should always be 0. The hard invariants enforce this.
    candidate_names = {img.name for img, _ in candidates}
    overlap_train = len(candidate_names & used_train)
    overlap_val = len(candidate_names & used_val)
    overlap_test = len(candidate_names & used_test)

    # HARD INVARIANTS: candidates must have ZERO overlap with any used split
    if overlap_train != 0:
        raise RuntimeError(
            f"LEAKAGE: {overlap_train} candidate(s) overlap with used_train. "
            "This must be 0 — check your --full-root and --used-root."
        )
    if overlap_val != 0:
        raise RuntimeError(
            f"LEAKAGE: {overlap_val} candidate(s) overlap with used_val. "
            "BDD val images must not appear in the pool."
        )
    if overlap_test != 0:
        raise RuntimeError(
            f"LEAKAGE: {overlap_test} candidate(s) overlap with used_test. "
            "BDD test images must not appear in the pool."
        )

    excluded_existing = len(all_pairs) - len(candidates)
    print(f"Excluded (in used BDD): {excluded_existing}")
    print(f"Candidate pool size:    {len(candidates)}")

    # ── Write output ─────────────────────────────────────────────────────────
    out_img_dir = out_root / "images" / "train"
    out_lbl_dir = out_root / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for img, lbl in candidates:
        place(img, out_img_dir / img.name, args.mode)
        place(lbl, out_lbl_dir / lbl.name, args.mode)
        manifest_rows.append({
            "image_name": img.name,
            "image": str(out_img_dir / img.name),
            "label": str(out_lbl_dir / lbl.name),
            "source_split": "bdd_train",
            "novel_vs_bdd30k": 1,
        })

    # ── dataset.yaml ────────────────────────────────────────────────────────
    yaml_text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES))
    )
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # ── candidate_manifest.csv ───────────────────────────────────────────────
    with (out_root / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "image", "label", "source_split", "novel_vs_bdd30k"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── excluded_existing_bdd_names.txt ─────────────────────────────────────
    excluded_name_set = {img.name for img, _ in all_pairs if img.name in excluded_names}
    (out_root / "excluded_existing_bdd_names.txt").write_text(
        "\n".join(sorted(excluded_name_set)), encoding="utf-8"
    )

    # ── pool_stats.json ──────────────────────────────────────────────────────
    pool_stats = {
        "full_train_pairs": full_train_count,
        "used_bdd_train": len(used_train),
        "used_bdd_val": len(used_val),
        "used_bdd_test": len(used_test),
        "excluded_existing_names": excluded_existing,
        "candidate_pool": len(candidates),
        "overlap_with_used_train": overlap_train,
        "overlap_with_used_val": overlap_val,
        "overlap_with_used_test": overlap_test,
    }
    (out_root / "pool_stats.json").write_text(
        json.dumps(pool_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nPool built → {out_root}")
    print(f"  candidate_pool: {len(candidates)}")
    print(f"  overlap_with_used_train: {overlap_train}  (must be 0)")
    print(f"  overlap_with_used_val:   {overlap_val}  (must be 0)")
    print(f"  overlap_with_used_test:  {overlap_test}  (must be 0)")


if __name__ == "__main__":
    main()
