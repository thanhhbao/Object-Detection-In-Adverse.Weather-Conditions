#!/usr/bin/env python3
"""
Active learning retrieval: hard sample mining → cosine similarity → pool expansion.

Pipeline:
  1. Load YOLO checkpoint, build embedding index for all pool images (BDD remaining pool)
  2. Run inference on query images (TRAINING DATA ONLY — caller is responsible for
     passing only train-split directories), find images where model fails on target classes
  3. Extract embeddings for hard samples
  4. Cosine similarity: hard_embs @ pool_embs.T
  5. Select pool images with sim >= threshold → output dir for retraining

WARNING: --query-root must point to TRAIN-split directories only. Using val or test
splits would constitute a data-leakage violation for this experiment. The CLI enforces
no path check — the caller is responsible.

Class order (project): 0=person 1=bicycle 2=car 3=motorcycle 4=bus 5=truck

New CLI usage (P2-A1):
  python scripts/active_retrieval.py \\
    --weights   runs/phase2_final_rtdetr/weights/best.pt \\
    --pool-root /data/bdd_remaining_pool \\
    --query-root /data/xwod_6cls_yolo/images/train \\
    --query-root /data/acdc_6cls_yolo/images/train \\
    --out-root  /data/bdd_active_retrieved \\
    --top-k 5000 --similarity-threshold 0.75 \\
    --target-classes 1 3 4 --seed 42 \\
    --used-bdd-root /data/bdd100k_6cls_yolo

Legacy CLI (original interface — kept for backward compatibility):
  python scripts/active_retrieval.py \\
    --checkpoint runs/phase2_rtdetr/weights/best.pt \\
    --hard-data   datasets/xwod_6cls_yolo/images/test \\
    --pool-data   datasets/bdd100k_6cls_yolo/images/train \\
    --out-dir     datasets/active_retrieved \\
    --top-n 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Heavy ML imports — only loaded when actually running (not on --help / validate_labels calls)
try:
    import cv2  # type: ignore[import]
    _cv2_available = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _cv2_available = False

try:
    import numpy as np  # type: ignore[import]
    _np_available = True
except ImportError:
    np = None  # type: ignore[assignment]
    _np_available = False

try:
    import torch  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]
    _torch_available = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _torch_available = False

try:
    from tqdm import tqdm  # type: ignore[import]
except ImportError:
    def tqdm(it, *args, **kwargs):  # type: ignore[misc]
        return it

try:
    from ultralytics import YOLO  # type: ignore[import]
    _ultralytics_available = True
except ImportError:
    YOLO = None  # type: ignore[assignment]
    _ultralytics_available = False

_ML_DEPS_AVAILABLE = _cv2_available and _np_available and _torch_available and _ultralytics_available

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMGSZ = 640
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


# ── Embedding extractor ────────────────────────────────────────────────────────

class BackboneHook:
    """Hooks layer[layer_idx] of a YOLO model, returns GAP-pooled features."""

    def __init__(self, model: YOLO, layer_idx: int = 9):
        self.feat: torch.Tensor | None = None
        self._h = model.model.model[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: [B, C, H, W] → mean pool → [B, C]
        self.feat = out.mean(dim=[2, 3])

    def remove(self):
        self._h.remove()


def preprocess_image(path: Path, imgsz: int = IMGSZ) -> torch.Tensor | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t


def extract_embeddings(
    model: YOLO,
    hook: BackboneHook,
    paths: list[Path],
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, list[Path]]:
    """Return (N, C) L2-normalized embeddings and matched path list."""
    embs, valid_paths = [], []

    for i in tqdm(range(0, len(paths), batch_size), desc="  Embedding", ncols=80):
        batch_paths = paths[i : i + batch_size]
        tensors = []
        kept = []
        for p in batch_paths:
            t = preprocess_image(p)
            if t is not None:
                tensors.append(t)
                kept.append(p)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            model.model(batch)  # triggers hook

        feat = hook.feat  # [B, C]
        feat = F.normalize(feat, dim=1).cpu().numpy()
        embs.append(feat)
        valid_paths.extend(kept)

    return np.vstack(embs), valid_paths


# ── Hard sample finder ─────────────────────────────────────────────────────────

def has_target_in_label(label_path: Path, target_classes: set[int]) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in target_classes:
            return True
    return False


def find_hard_samples(
    model: YOLO,
    image_dir: Path,
    label_dir: Path | None,
    target_classes: set[int],
    conf_hard: float,
    batch_size: int = 1,
) -> tuple[list[Path], list[Path | None], dict[str, float]]:
    """
    Hard sample = image has target class in GT but model detects with
    max confidence < conf_hard (or no detection at all).
    If no label_dir, treat every image as a candidate and check confidence only.

    Returns: (hard_img_paths, hard_lbl_paths, hardness_scores)
    hardness_scores: {img_name -> max_conf on target class}
    """
    img_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    hard_imgs, hard_lbls = [], []
    hardness_scores: dict[str, float] = {}
    skipped = 0

    for img_path in tqdm(img_paths, desc="  Hard mining", ncols=80):
        if label_dir:
            lbl_path = label_dir / img_path.with_suffix(".txt").name
            if not has_target_in_label(lbl_path, target_classes):
                skipped += 1
                continue
        else:
            lbl_path = None

        results = model.predict(str(img_path), verbose=False, conf=0.01, imgsz=IMGSZ)
        max_conf = 0.0
        for r in results:
            if r.boxes is not None and len(r.boxes):
                for cls_t, conf_t in zip(r.boxes.cls, r.boxes.conf):
                    if int(cls_t) in target_classes:
                        max_conf = max(max_conf, float(conf_t))

        if max_conf < conf_hard:
            hard_imgs.append(img_path)
            hard_lbls.append(lbl_path)
            hardness_scores[img_path.name] = max_conf

    names = [CLASS_NAMES[c] for c in sorted(target_classes) if c < len(CLASS_NAMES)]
    print(f"  Hard samples found: {len(hard_imgs)} / {len(img_paths)} "
          f"(skipped {skipped} without GT target; target classes: {names})")
    return hard_imgs, hard_lbls, hardness_scores


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_from_pool(
    hard_embs: np.ndarray,
    pool_embs: np.ndarray,
    pool_paths: list[Path],
    sim_threshold: float,
    top_n: int,
    exclude_stems: set[str] | None = None,
) -> list[tuple[Path, float]]:
    """
    For each hard sample, find pool images with cosine sim >= threshold.
    Returns deduplicated list of (pool_path, max_sim_score) sorted by score desc.
    Limits to top_n total. Uses (-sim, name) tie-breaking for determinism.
    """
    sim_matrix = hard_embs @ pool_embs.T  # both L2-normalized → cosine sim

    selected: dict[str, tuple[Path, float]] = {}  # stem → (path, score)

    for h_idx in range(sim_matrix.shape[0]):
        sims = sim_matrix[h_idx]
        order = np.argsort(-sims)
        for p_idx in order:
            score = float(sims[p_idx])
            if score < sim_threshold:
                break
            p_path = pool_paths[p_idx]
            stem = p_path.stem
            if exclude_stems and stem in exclude_stems:
                continue
            if stem not in selected or selected[stem][1] < score:
                selected[stem] = (p_path, score)

    # Deterministic sort: (-sim, name) for tie-breaking
    ranked = sorted(selected.values(), key=lambda x: (-x[1], x[0].name))
    return ranked[:top_n]


# ── Export (legacy) ────────────────────────────────────────────────────────────

def export_retrieved(
    selected: list[tuple[Path, float]],
    pool_label_dir: Path | None,
    out_dir: Path,
) -> None:
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied_imgs, copied_lbls = 0, 0
    no_label = []

    for img_path, score in tqdm(selected, desc="  Copying", ncols=80):
        dst_img = img_out / img_path.name
        shutil.copy2(img_path, dst_img)
        copied_imgs += 1

        if pool_label_dir:
            lbl_path = pool_label_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
                copied_lbls += 1
            else:
                no_label.append(img_path.name)

    print(f"  Exported: {copied_imgs} images, {copied_lbls} labels → {out_dir}")
    if no_label:
        print(f"  WARNING: {len(no_label)} images had no matching label file")

    manifest = out_dir / "retrieval_scores.txt"
    with open(manifest, "w") as f:
        f.write("image\tsim_score\n")
        for p, s in selected:
            f.write(f"{p.name}\t{s:.4f}\n")
    print(f"  Score manifest → {manifest}")


# ── Stats ──────────────────────────────────────────────────────────────────────

def count_class_dist(label_dir: Path, target_classes: set[int]) -> dict[int, int]:
    counts = {c: 0 for c in target_classes}
    for lbl in label_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                c = int(parts[0])
                if c in target_classes:
                    counts[c] += 1
    return counts


def validate_labels(label_dir: Path, max_class_id: int = 5) -> list[str]:
    """Check all label files for invalid class IDs. Returns list of violation messages."""
    violations: list[str] = []
    for lbl in label_dir.glob("*.txt"):
        for i, line in enumerate(lbl.read_text().splitlines()):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except (ValueError, IndexError):
                violations.append(f"{lbl.name}:{i+1}: cannot parse class id: {line!r}")
                continue
            if cid < 0 or cid > max_class_id:
                violations.append(f"{lbl.name}:{i+1}: class id {cid} out of range [0..{max_class_id}]")
    return violations


def place_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


# ── New-style main (P2-A1) ─────────────────────────────────────────────────────

def parse_args_new() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active retrieval: hard mining + cosine similarity pool search (P2-A1)")
    p.add_argument("--weights", required=True, help="Phase2 best.pt checkpoint")
    p.add_argument("--pool-root", required=True, type=Path,
                   help="BDD remaining pool root (output of build_bdd_retrieval_pool.py)")
    p.add_argument("--query-root", action="append", dest="query_roots", type=Path, required=True,
                   help="TRAIN-split image directories (repeatable, >=1). Caller MUST pass only train dirs.")
    p.add_argument("--out-root", required=True, type=Path, help="Output directory")
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--similarity-threshold", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-classes", nargs="+", type=int, default=[1, 3, 4],
                   help="Class IDs to mine (default: 1=bicycle 3=motorcycle 4=bus)")
    p.add_argument("--conf-hard", type=float, default=0.25)
    p.add_argument("--layer-idx", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--cache-pool-embs", type=Path, default=None,
                   help="Path to save/load pool embeddings cache (.npz)")
    p.add_argument("--used-bdd-root", type=Path, default=None,
                   help="BDD30K root already in project — used for post-selection leakage check")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    return p.parse_args()


def parse_args_legacy() -> argparse.Namespace:
    """Legacy argument parser — kept for backward compatibility."""
    p = argparse.ArgumentParser(description="Active retrieval: hard mining + cosine similarity pool search")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hard-data", required=True)
    p.add_argument("--hard-labels", default=None)
    p.add_argument("--pool-data", required=True)
    p.add_argument("--pool-labels", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-classes", nargs="+", type=int, default=[1, 3, 4])
    p.add_argument("--conf-hard", type=float, default=0.25)
    p.add_argument("--sim-threshold", type=float, default=0.75)
    p.add_argument("--top-n", type=int, default=500)
    p.add_argument("--layer-idx", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--exclude-in-train", default=None)
    p.add_argument("--cache-pool-embs", default=None)
    _legacy_default_device = "cuda" if (_torch_available and torch.cuda.is_available()) else "cpu"
    p.add_argument("--device", default=_legacy_default_device)
    return p.parse_args()


def _detect_new_style() -> bool:
    """Return True if --weights / --pool-root / --query-root style args are present.
    Also returns True for --help / -h when no legacy-only flags are present,
    so that the default help output shows the new-style (P2-A1) interface.
    """
    argv = sys.argv[1:]
    new_style_flags = {"--weights", "--pool-root", "--query-root"}
    help_flags = {"-h", "--help"}
    legacy_only_flags = {"--checkpoint", "--hard-data", "--pool-data", "--out-dir"}
    argv_set = set(argv)
    if argv_set & new_style_flags:
        return True
    if argv_set & help_flags and not (argv_set & legacy_only_flags):
        return True
    return False


def main_new(args: argparse.Namespace) -> None:
    """P2-A1 main: new CLI interface."""
    target_classes = set(args.target_classes)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Active Retrieval (P2-A1 — failure-driven BDD retrieval)")
    print(f"  weights:    {args.weights}")
    print(f"  pool-root:  {args.pool_root}")
    print(f"  query-roots: {[str(q) for q in args.query_roots]}")
    print(f"  top-k:      {args.top_k}")
    print(f"  sim-thresh: {args.similarity_threshold}")
    print(f"  target:     {[CLASS_NAMES[c] for c in sorted(target_classes)]}")
    print(f"  seed:       {args.seed}")
    print(f"  device:     {device}")
    print(f"{'='*60}\n")

    # Pool
    pool_img_dir = args.pool_root / "images" / "train"
    pool_lbl_dir = args.pool_root / "labels" / "train"
    pool_paths = sorted(p for p in pool_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"[1/4] Pool size: {len(pool_paths)} images")

    # Load model
    print("[2/4] Loading checkpoint...")
    model = YOLO(str(args.weights))
    model.to(device)
    hook = BackboneHook(model, layer_idx=args.layer_idx)

    # Pool embeddings
    cache_path = args.cache_pool_embs
    if cache_path and cache_path.exists():
        print(f"  Loading cached pool embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        pool_embs = data["embs"]
        pool_paths = [Path(str(p)) for p in data["paths"]]
        print(f"  Loaded {len(pool_paths)} cached embeddings")
    else:
        pool_embs, pool_paths = extract_embeddings(
            model, hook, pool_paths, batch_size=args.batch_size, device=device
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, embs=pool_embs, paths=np.array([str(p) for p in pool_paths]))
            print(f"  Pool embeddings cached → {cache_path}")

    # Hard sample mining from all query dirs
    print("\n[3/4] Mining hard samples from query dirs...")
    all_hard_imgs: list[Path] = []
    all_hard_lbls: list[Path | None] = []
    all_hardness: dict[str, float] = {}
    query_count = 0

    for query_root in args.query_roots:
        query_root = query_root.resolve()
        # Infer label dir: images/train → labels/train
        lbl_dir = None
        parts = query_root.parts
        if "images" in parts:
            idx = list(parts).index("images")
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels"
            lbl_dir = Path(*lbl_parts)
            if not lbl_dir.exists():
                lbl_dir = None
        imgs_here = sorted(p for p in query_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        query_count += len(imgs_here)
        hard_imgs, hard_lbls, hardness = find_hard_samples(
            model, query_root, lbl_dir, target_classes, args.conf_hard,
        )
        all_hard_imgs.extend(hard_imgs)
        all_hard_lbls.extend(hard_lbls)
        all_hardness.update(hardness)

    if not all_hard_imgs:
        print("  No hard samples found. Try lowering --conf-hard.")
        hook.remove()
        return

    print(f"  Total hard samples across all query dirs: {len(all_hard_imgs)}")
    print("  Extracting hard sample embeddings...")
    hard_embs, all_hard_imgs = extract_embeddings(
        model, hook, all_hard_imgs, batch_size=args.batch_size, device=device
    )
    hook.remove()

    # Retrieval
    print(f"\n[4/4] Retrieving from pool (sim >= {args.similarity_threshold}, top-{args.top_k})...")
    selected = retrieve_from_pool(
        hard_embs, pool_embs, pool_paths,
        sim_threshold=args.similarity_threshold,
        top_n=args.top_k,
    )
    print(f"  Retrieved: {len(selected)} unique pool images above threshold")

    # Leakage check
    if args.used_bdd_root:
        used_bdd_names: set[str] = set()
        for split in ("train", "val", "test"):
            split_dir = args.used_bdd_root / "images" / split
            if split_dir.exists():
                for p in split_dir.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        used_bdd_names.add(p.name)
        retrieved_names = {img.name for img, _ in selected}
        overlap = retrieved_names & used_bdd_names
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: {len(overlap)} retrieved image(s) appear in used BDD30K: "
                f"{sorted(overlap)[:5]}"
            )
        overlap_count = 0
    else:
        overlap_count = -1  # not checked

    # HARD INVARIANTS
    assert len(selected) <= args.top_k, "selected_unique > top_k — invariant violated"
    retrieved_basenames = [img.name for img, _ in selected]
    assert len(retrieved_basenames) == len(set(retrieved_basenames)), "duplicate retrieved image names"

    # Write output
    out_img_dir = out_root / "images" / "train"
    out_lbl_dir = out_root / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for rank, (img_path, sim) in enumerate(selected, start=1):
        lbl_path = pool_lbl_dir / f"{img_path.stem}.txt"
        place_file(img_path, out_img_dir / img_path.name, args.mode)
        if lbl_path.exists():
            place_file(lbl_path, out_lbl_dir / lbl_path.name, args.mode)
        hardness = all_hardness.get(img_path.name, "")
        manifest_rows.append({
            "retrieved_image": str(out_img_dir / img_path.name),
            "retrieved_label": str(out_lbl_dir / lbl_path.name) if lbl_path.exists() else "",
            "source_image_name": img_path.name,
            "source_split": "bdd_train",
            "query_image": "",
            "similarity": f"{sim:.6f}",
            "hardness_score": f"{hardness:.6f}" if isinstance(hardness, float) else "",
            "rank": rank,
            "selected_reason": "similarity_retrieval",
        })

    # dataset.yaml
    yaml_text = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "nc: 6\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES))
    )
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # retrieved_manifest.csv
    with (out_root / "retrieved_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "retrieved_image", "retrieved_label", "source_image_name", "source_split",
            "query_image", "similarity", "hardness_score", "rank", "selected_reason",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # retrieval_stats.json
    source_split_counts = {"bdd_train": len(selected)}
    stats = {
        "candidate_pool_size": len(pool_paths),
        "query_count": query_count,
        "requested_top_k": args.top_k,
        "selected_unique": len(selected),
        "similarity_threshold": args.similarity_threshold,
        "seed": args.seed,
        "duplicate_candidates_removed": 0,
        "overlap_with_used_bdd": overlap_count if args.used_bdd_root else -1,
        "source_split_counts": source_split_counts,
    }
    (out_root / "retrieval_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. Retrieved data → {out_root}")
    print(f"  selected_unique: {len(selected)}")
    print(f"  source_split: all bdd_train")


def main_legacy(args: argparse.Namespace) -> None:
    """Legacy main — original CLI interface."""
    target_classes = set(args.target_classes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Active Retrieval")
    print(f"  checkpoint:     {args.checkpoint}")
    print(f"  hard data:      {args.hard_data}")
    print(f"  pool:           {args.pool_data}")
    print(f"  target classes: {[CLASS_NAMES[c] for c in sorted(target_classes)]}")
    print(f"  conf-hard:      {args.conf_hard}")
    print(f"  sim-threshold:  {args.sim_threshold}")
    print(f"  top-n:          {args.top_n}")
    print(f"  device:         {args.device}")
    print(f"{'='*60}\n")

    print("[1/4] Loading checkpoint...")
    model = YOLO(args.checkpoint)
    model.to(args.device)
    hook = BackboneHook(model, layer_idx=args.layer_idx)

    print("[2/4] Building pool embeddings...")
    pool_data_dir = Path(args.pool_data)
    pool_paths = sorted(p for p in pool_data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"  Pool size: {len(pool_paths)} images")

    cache_path = Path(args.cache_pool_embs) if args.cache_pool_embs else None
    if cache_path and cache_path.exists():
        print(f"  Loading cached pool embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        pool_embs = data["embs"]
        pool_paths = [Path(str(p)) for p in data["paths"]]
        print(f"  Loaded {len(pool_paths)} cached embeddings")
    else:
        pool_embs, pool_paths = extract_embeddings(
            model, hook, pool_paths, batch_size=args.batch_size, device=args.device
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, embs=pool_embs, paths=np.array([str(p) for p in pool_paths]))
            print(f"  Pool embeddings cached → {cache_path}")

    print("\n[3/4] Mining hard samples...")
    hard_data_dir = Path(args.hard_data)
    hard_label_dir = Path(args.hard_labels) if args.hard_labels else None

    hard_img_paths, hard_lbl_paths, _ = find_hard_samples(
        model, hard_data_dir, hard_label_dir,
        target_classes, args.conf_hard,
    )

    if not hard_img_paths:
        print("  No hard samples found. Try lowering --conf-hard.")
        hook.remove()
        return

    print("  Extracting hard sample embeddings...")
    hard_embs, hard_img_paths = extract_embeddings(
        model, hook, hard_img_paths, batch_size=args.batch_size, device=args.device
    )
    hook.remove()

    print(f"\n[4/4] Retrieving from pool (sim >= {args.sim_threshold}, top-{args.top_n})...")

    exclude_stems: set[str] | None = None
    if args.exclude_in_train:
        exclude_file = Path(args.exclude_in_train)
        if exclude_file.exists():
            exclude_stems = set(exclude_file.read_text().splitlines())
            print(f"  Excluding {len(exclude_stems)} stems already in train")

    selected = retrieve_from_pool(
        hard_embs, pool_embs, pool_paths,
        sim_threshold=args.sim_threshold,
        top_n=args.top_n,
        exclude_stems=exclude_stems,
    )
    print(f"  Retrieved: {len(selected)} pool images above threshold")

    if not selected:
        print("  Nothing retrieved. Try lowering --sim-threshold.")
        return

    pool_label_dir = Path(args.pool_labels) if args.pool_labels else None
    export_retrieved(selected, pool_label_dir, out_dir)

    if pool_label_dir:
        lbl_out = out_dir / "labels"
        if lbl_out.exists():
            dist = count_class_dist(lbl_out, target_classes)
            print("\n  Class distribution in retrieved labels:")
            for c, cnt in sorted(dist.items()):
                print(f"    {CLASS_NAMES[c]:12s} ({c}): {cnt} instances")

    scores = [s for _, s in selected]
    print(f"\n  Similarity score stats:")
    print(f"    min={min(scores):.3f}  mean={np.mean(scores):.3f}  max={max(scores):.3f}")

    print(f"\nDone. Retrieved data → {out_dir}")
    print("Next: merge this dir into your Phase 2 dataset and retrain.\n")


def main() -> None:
    if _detect_new_style():
        main_new(parse_args_new())
    else:
        main_legacy(parse_args_legacy())


if __name__ == "__main__":
    main()
