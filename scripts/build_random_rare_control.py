#!/usr/bin/env python3
"""
A0R control: uniformly random sampling from BDD remaining pool filtered to rare classes.

This is the A0R ablation arm: same pool, same count as A1/A1-DINO, zero intelligence
in selection. Picks exactly --top-k images at random from the rare-class-filtered pool.

Rare classes: 1=bicycle, 3=motorcycle, 4=bus
Class order (project): 0=person 1=bicycle 2=car 3=motorcycle 4=bus 5=truck

Usage:
  python scripts/build_random_rare_control.py \\
    --pool-root /workspace/datasets_noleak/bdd_remaining_pool \\
    --out-root  /workspace/datasets_noleak/bdd_random_rare \\
    --top-k 5000 \\
    --candidate-target-classes 1 3 4 \\
    --seed 42 \\
    --mode symlink \\
    --used-bdd-root /workspace/datasets_noleak/bdd100k_6cls_yolo \\
    --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from active_retrieval import validate_labels, place_file, filter_pool_by_class  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
RARE_CLASS_NAMES = {1: "bicycle", 3: "motorcycle", 4: "bus"}


def _get_rare_classes_in_image(lbl_path: Path, rare_classes: set) -> list:
    """Return sorted list of rare class names present in the label file."""
    present = set()
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cid = int(float(parts[0]))
                    if cid in rare_classes:
                        present.add(RARE_CLASS_NAMES.get(cid, str(cid)))
                except (ValueError, IndexError):
                    pass
    return sorted(present)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A0R control: random rare-class sampling from BDD remaining pool"
    )
    p.add_argument("--pool-root", required=True, type=Path,
                   help="BDD remaining pool root")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Output directory")
    p.add_argument("--top-k", type=int, default=5000,
                   help="Target number of images to select (default: 5000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for deterministic selection (default: 42)")
    p.add_argument("--candidate-target-classes", nargs="+", type=int, default=[1, 3, 4],
                   help="Rare class IDs to filter pool by (default: 1 3 4)")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                   help="Output mode: symlink or copy (default: symlink)")
    p.add_argument("--clean", action="store_true",
                   help="Remove --out-root before writing. Without it, fail if non-empty.")
    p.add_argument("--used-bdd-root", type=Path, default=None,
                   help="BDD30K root already in project — post-selection leakage check")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pool_root = Path(args.pool_root).resolve()
    out_root = Path(args.out_root).resolve()
    rare_classes = set(args.candidate_target_classes)
    top_k = args.top_k
    seed = args.seed

    # Clean / stale-output guard
    if args.clean:
        if out_root.exists():
            print(f"  --clean: removing existing output at {out_root}")
            shutil.rmtree(out_root)
    else:
        stale_img = out_root / "images" / "train"
        stale_lbl = out_root / "labels" / "train"
        stale = []
        if stale_img.exists() and any(stale_img.iterdir()):
            stale.append(str(stale_img))
        if stale_lbl.exists() and any(stale_lbl.iterdir()):
            stale.append(str(stale_lbl))
        if stale:
            raise RuntimeError(
                f"Output directory is non-empty: {stale}\n"
                "Pass --clean to remove and rebuild."
            )

    out_root.mkdir(parents=True, exist_ok=True)

    # Pool paths
    pool_img_dir = pool_root / "images" / "train"
    pool_lbl_dir = pool_root / "labels" / "train"

    print(f"\n{'='*60}")
    print("A0R Random Rare Control")
    print(f"  pool-root:   {pool_root}")
    print(f"  out-root:    {out_root}")
    print(f"  top-k:       {top_k}")
    print(f"  seed:        {seed}")
    print(f"  rare classes: {sorted(rare_classes)}")
    print(f"{'='*60}\n")

    all_paths = sorted(p for p in pool_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    candidate_pool_before_filter = len(all_paths)
    print(f"[1/3] Pool size: {candidate_pool_before_filter} images")

    # Filter pool to rare-class images
    filtered_paths, _ = filter_pool_by_class(pool_img_dir, pool_lbl_dir, rare_classes)
    candidate_pool_after_filter = len(filtered_paths)
    print(f"  After rare-class filter: {candidate_pool_after_filter} images")

    if candidate_pool_after_filter == 0:
        raise RuntimeError(
            f"Pool is empty after filtering for rare classes {sorted(rare_classes)}. "
            "Check that pool labels exist and contain these class IDs."
        )

    # Shuffle deterministically, take top_k
    rng = random.Random(seed)
    shuffled = list(filtered_paths)
    rng.shuffle(shuffled)

    if len(shuffled) < top_k:
        print(f"  WARNING: Only {len(shuffled)} rare images available, less than top-k={top_k}. Using all.")
    selected_paths = shuffled[:top_k]
    selected_count = len(selected_paths)
    print(f"[2/3] Selected: {selected_count} images (seed={seed})")

    # Leakage check
    if args.used_bdd_root:
        used_bdd_names: set = set()
        for split in ("train", "val", "test"):
            split_dir = args.used_bdd_root / "images" / split
            if split_dir.exists():
                for p in split_dir.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        used_bdd_names.add(p.name)
        selected_names = {p.name for p in selected_paths}
        overlap = selected_names & used_bdd_names
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: {len(overlap)} selected image(s) appear in used BDD30K: "
                f"{sorted(overlap)[:5]}"
            )
        overlap_count = 0
    else:
        overlap_count = -1  # not checked

    # Write output
    out_img_dir = out_root / "images" / "train"
    out_lbl_dir = out_root / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] Writing output ({args.mode})...")
    manifest_rows: list = []
    for rank, img_path in enumerate(selected_paths, start=1):
        lbl_path = pool_lbl_dir / f"{img_path.stem}.txt"
        place_file(img_path, out_img_dir / img_path.name, args.mode)
        if lbl_path.exists():
            place_file(lbl_path, out_lbl_dir / lbl_path.name, args.mode)
        rare_present = _get_rare_classes_in_image(lbl_path, rare_classes)
        manifest_rows.append({
            "image_name": img_path.name,
            "image": str(out_img_dir / img_path.name),
            "label": str(out_lbl_dir / lbl_path.name) if lbl_path.exists() else "",
            "source_split": "bdd_train",
            "selection_method": "random_rare",
            "rare_class_present": ",".join(rare_present),
            "rank": rank,
        })

    # dataset.yaml
    yaml_text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # random_rare_manifest.csv
    with (out_root / "random_rare_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "image", "label", "source_split",
            "selection_method", "rare_class_present", "rank",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Output count invariants
    output_img_count = sum(
        1 for p in out_img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and (p.is_file() or p.is_symlink())
    )
    output_lbl_count = sum(
        1 for p in out_lbl_dir.iterdir()
        if p.suffix.lower() == ".txt" and (p.is_file() or p.is_symlink())
    )
    manifest_row_count = len(manifest_rows)

    if output_img_count != selected_count or output_lbl_count != selected_count or manifest_row_count != selected_count:
        raise RuntimeError(
            f"OUTPUT COUNT MISMATCH: images={output_img_count}, labels={output_lbl_count}, "
            f"manifest={manifest_row_count}, selected={selected_count}. "
            "These must all be equal. Pass --clean and rerun."
        )

    # Label validation
    violations = validate_labels(out_lbl_dir, max_class_id=5)
    if violations:
        raise RuntimeError(f"Label validation failed:\n" + "\n".join(violations[:10]))

    # random_rare_stats.json
    stats = {
        "candidate_pool_before_filter": candidate_pool_before_filter,
        "candidate_pool_after_filter": candidate_pool_after_filter,
        "requested_top_k": top_k,
        "selected": selected_count,
        "seed": seed,
        "selection_method": "random_rare",
        "overlap_with_used_bdd": overlap_count,
        "output_image_count": output_img_count,
        "output_label_count": output_lbl_count,
        "manifest_row_count": manifest_row_count,
        "source_split": "bdd_train",
    }
    (out_root / "random_rare_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. Random rare data → {out_root}")
    print(f"  selected: {selected_count}")
    print(f"  source_split: bdd_train")
    print(f"  selection_method: random_rare")


if __name__ == "__main__":
    main()
