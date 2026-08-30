#!/usr/bin/env python3
"""
DINOv2-based retrieval: RT-DETR hard mining + DINOv2 cosine similarity pool search.

Pipeline:
  1. RT-DETR checkpoint → find_hard_samples_gt_aware() → hard sample paths
  2. DINOv2 model → embed pool images → pool_embs [N, D]
  3. DINOv2 model → embed hard sample images → hard_embs [H, D]
  4. Cosine similarity (DINOv2 is naturally isotropic — no centering needed)
  5. Select top-K with sim >= threshold

Class order (project): 0=person 1=bicycle 2=car 3=motorcycle 4=bus 5=truck
Rare classes: 1=bicycle, 3=motorcycle, 4=bus

Usage:
  python scripts/retrieve_dinov2.py \\
    --weights /workspace/runs/phase2_final_rtdetr/weights/best.pt \\
    --pool-root /workspace/datasets_noleak/bdd_remaining_pool \\
    --query-root /workspace/datasets_noleak/xwod_6cls_yolo/images/train \\
    --query-root /workspace/datasets_noleak/acdc_6cls_yolo/images/train \\
    --out-root /workspace/datasets_noleak/bdd_dinov2_retrieved \\
    --top-k 5000 \\
    --similarity-threshold 0.70 \\
    --target-classes 1 3 4 \\
    --candidate-target-classes 1 3 4 \\
    --mode symlink \\
    --used-bdd-root /workspace/datasets_noleak/bdd100k_6cls_yolo \\
    --seed 42 \\
    --clean
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Heavy ML imports — guarded so tests can import without GPU
try:
    import numpy as np
    _np_available = True
except ImportError:
    np = None  # type: ignore[assignment]
    _np_available = False

try:
    import torch
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _torch_available = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):  # type: ignore[misc]
        return it

try:
    from ultralytics import YOLO
    _ultralytics_available = True
except ImportError:
    YOLO = None  # type: ignore[assignment]
    _ultralytics_available = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
DINOV2_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
DINOV2_CHOICES = list(DINOV2_DIM.keys())


# ── Imports from active_retrieval ──────────────────────────────────────────────

from active_retrieval import (  # noqa: E402
    find_hard_samples_gt_aware,
    filter_pool_by_class,
    validate_query_root,
    _pool_fingerprint,
    load_pool_cache,
    save_pool_cache,
    validate_labels,
    place_file,
)


def _infer_query_dataset(query_root: Path) -> str:
    path_str = str(query_root).lower()
    if "xwod" in path_str:
        return "xwod"
    if "acdc" in path_str:
        return "acdc"
    return query_root.parent.name


# ── DINOv2 loading ─────────────────────────────────────────────────────────────

def load_dinov2(model_name: str, device: str):
    """Load DINOv2 model from torch.hub, with transformers fallback."""
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
    except Exception:
        # Fallback: try transformers
        from transformers import AutoModel, AutoImageProcessor  # type: ignore[import]
        model = AutoModel.from_pretrained(f'facebook/{model_name}')
    model.eval()
    model.to(device)
    return model


# ── DINOv2 embedding extraction ────────────────────────────────────────────────

def extract_dinov2_embeddings(
    dinov2_model,
    paths: list,
    batch_size: int,
    device: str,
    imgsz: int = 224,
) -> "tuple[np.ndarray, list]":
    """
    Extract L2-normalized DINOv2 embeddings for a list of image paths.

    DINOv2 expects 224x224 RGB, normalized with ImageNet mean/std.
    Returns (N, D) array of L2-normalized embeddings and matched paths.
    """
    import cv2  # type: ignore[import]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    embs = []
    valid_paths = []

    for i in tqdm(range(0, len(paths), batch_size), desc="  DINOv2 embed", ncols=80):
        batch_paths = paths[i: i + batch_size]
        tensors = []
        kept = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (imgsz, imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # Normalize with ImageNet stats
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            tensors.append(t)
            kept.append(p)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = dinov2_model(batch)
            # DINOv2 returns CLS token as first token or as a dict
            if isinstance(out, dict):
                feat = out.get("last_hidden_state", out.get("pooler_output"))
                if feat is not None and feat.ndim == 3:
                    feat = feat[:, 0]  # CLS token
            elif isinstance(out, torch.Tensor):
                if out.ndim == 3:
                    feat = out[:, 0]  # CLS token
                else:
                    feat = out
            else:
                # torch.hub DINOv2 typically returns [B, D] directly via forward_features
                try:
                    feat = dinov2_model.forward_features(batch)
                    if isinstance(feat, dict):
                        feat = feat["x_norm_clstoken"]
                    elif feat.ndim == 3:
                        feat = feat[:, 0]
                except Exception:
                    feat = out

        feat = F.normalize(feat.float(), dim=1)
        embs.append(feat.cpu().numpy())
        valid_paths.extend(kept)

    if not embs:
        return np.zeros((0, DINOV2_DIM.get("dinov2_vitb14", 768)), dtype=np.float32), []
    return np.vstack(embs), valid_paths


# ── Pool embedding cache (DINOv2-specific metadata) ───────────────────────────

def _dinov2_cache_meta(
    weights_path: Path,
    dinov2_model_name: str,
    pool_paths: list,
    imgsz: int = 224,
) -> dict:
    weights_path = Path(weights_path)
    try:
        size = weights_path.stat().st_size
        with open(weights_path, "rb") as f:
            partial = f.read(65536)
        sha = hashlib.sha256(partial).hexdigest()[:16]
    except OSError:
        size, sha = -1, "unknown"
    return {
        "checkpoint": str(weights_path),
        "checkpoint_size": size,
        "checkpoint_sha256_prefix": sha,
        "dinov2_model": dinov2_model_name,
        "embedding_layers": ["dinov2_cls_token"],
        "imgsz": imgsz,
        "pool_count": len(pool_paths),
        "pool_fingerprint": _pool_fingerprint(pool_paths),
        "version": 1,
    }


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_dinov2(
    hard_embs: "np.ndarray",
    hard_paths: list,
    hardness_scores: dict,
    img_to_dataset: dict,
    pool_embs: "np.ndarray",
    pool_paths: list,
    sim_threshold: float,
    top_n: int,
) -> "tuple[list[dict], dict]":
    """
    Cosine similarity retrieval using DINOv2 embeddings.
    Both hard_embs and pool_embs are L2-normalized — dot product = cosine similarity.
    No centering: DINOv2 is naturally isotropic.
    """
    sim_matrix = hard_embs @ pool_embs.T  # [H, P]

    best_per_stem: dict = {}
    candidate_hits_above_threshold = 0

    for h_idx, hard_path in enumerate(hard_paths):
        canonical_key = str(hard_path.resolve())
        sims = sim_matrix[h_idx]
        order = np.argsort(-sims)
        for p_idx in order:
            score = float(sims[p_idx])
            if score < sim_threshold:
                break
            candidate_hits_above_threshold += 1
            p_path = pool_paths[p_idx]
            stem = p_path.stem
            if stem not in best_per_stem or best_per_stem[stem]["sim"] < score:
                best_per_stem[stem] = {
                    "pool_path": p_path,
                    "sim": score,
                    "query_path": hard_path,
                    "query_dataset": img_to_dataset.get(canonical_key, ""),
                    "hardness_score": hardness_scores.get(canonical_key, 0.0),
                }

    unique_before_topk = len(best_per_stem)
    duplicate_removed = candidate_hits_above_threshold - unique_before_topk

    # Deterministic sort: (-sim, pool_name)
    ranked = sorted(best_per_stem.values(), key=lambda x: (-x["sim"], x["pool_path"].name))
    selected = ranked[:top_n]

    stats = {
        "candidate_hits_above_threshold": candidate_hits_above_threshold,
        "unique_candidates_before_top_k": unique_before_topk,
        "duplicate_candidate_hits_removed": duplicate_removed,
        "selected_unique": len(selected),
    }
    return selected, stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 retrieval: RT-DETR hard mining + DINOv2 cosine similarity pool search"
    )
    p.add_argument("--weights", required=True, help="RT-DETR checkpoint for hard mining")
    p.add_argument("--pool-root", required=True, type=Path,
                   help="BDD remaining pool (output of build_bdd_retrieval_pool.py)")
    p.add_argument("--query-root", action="append", dest="query_roots", type=Path, required=True,
                   help="Train-split image directories (repeatable). Must be train splits only.")
    p.add_argument("--out-root", required=True, type=Path, help="Output directory")
    p.add_argument("--dinov2-model", default="dinov2_vitb14", choices=DINOV2_CHOICES,
                   help="DINOv2 model variant (default: dinov2_vitb14)")
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--similarity-threshold", type=float, default=0.70,
                   help="Cosine similarity threshold (DINOv2 needs lower than RT-DETR centered, default: 0.70)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-classes", nargs="+", type=int, default=[1, 3, 4],
                   help="Class IDs for hard mining (default: 1=bicycle 3=motorcycle 4=bus)")
    p.add_argument("--conf-hard", type=float, default=0.25)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--candidate-target-classes", nargs="+", type=int, default=[1, 3, 4],
                   help="Filter pool by label; -1 to disable (default: 1 3 4)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--cache-pool-embs", type=Path, default=None,
                   help="Path to save/load DINOv2 pool embedding cache (.npz)")
    p.add_argument("--used-bdd-root", type=Path, default=None,
                   help="BDD30K root already in project — post-selection leakage check")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--clean", action="store_true",
                   help="Remove --out-root before writing. Without it, fail if non-empty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    target_classes = set(args.target_classes)
    device = args.device or ("cuda" if (_torch_available and torch.cuda.is_available()) else "cpu")
    _ = random.Random(args.seed)  # seed for reproducibility
    out_root = args.out_root.resolve()
    pool_root = Path(args.pool_root).resolve()
    dinov2_model_name = args.dinov2_model
    embedding_dim = DINOV2_DIM.get(dinov2_model_name, 768)

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

    # Candidate class filter setup
    candidate_target_classes_raw = args.candidate_target_classes
    use_class_filter = candidate_target_classes_raw != [-1]
    candidate_target_classes = set(candidate_target_classes_raw) if use_class_filter else None

    print(f"\n{'='*60}")
    print("DINOv2 Retrieval (RT-DETR hard mining + DINOv2 cosine similarity)")
    print(f"  weights:              {args.weights}")
    print(f"  pool-root:            {pool_root}")
    print(f"  query-roots:          {[str(q) for q in args.query_roots]}")
    print(f"  top-k:                {args.top_k}")
    print(f"  similarity-threshold: {args.similarity_threshold}")
    print(f"  target:               {[CLASS_NAMES[c] for c in sorted(target_classes)]}")
    print(f"  match-iou:            {args.match_iou}")
    print(f"  seed:                 {args.seed}")
    print(f"  device:               {device}")
    print(f"{'='*60}\n")

    print(f"DINOv2 model: {dinov2_model_name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Centering applied: False")

    # Validate query roots (train-only enforcement)
    validated_roots = [validate_query_root(qr) for qr in args.query_roots]
    args.query_roots = validated_roots

    # Pool paths
    pool_img_dir = pool_root / "images" / "train"
    pool_lbl_dir = pool_root / "labels" / "train"
    pool_paths = sorted(p for p in pool_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    pool_count_before_filter = len(pool_paths)
    print(f"[1/4] Pool size: {len(pool_paths)} images")

    # Candidate class filter
    if use_class_filter and candidate_target_classes:
        print(f"  Filtering pool to images with classes {sorted(candidate_target_classes)}...")
        pool_paths, pool_count_before_filter = filter_pool_by_class(
            pool_img_dir, pool_lbl_dir, candidate_target_classes
        )
        print(f"  Pool after class filter: {len(pool_paths)} / {pool_count_before_filter}")
    pool_count_after_filter = len(pool_paths)
    print(f"Pool size (after class filter): {pool_count_after_filter}")

    # Load RT-DETR model for hard mining
    print("[2/4] Loading RT-DETR checkpoint for hard mining...")
    model = YOLO(str(args.weights))
    model.to(device)

    # Hard mining
    print("\n[3/4] GT-aware hard mining from query dirs...")
    all_hard_imgs: list = []
    all_hardness: dict = {}
    all_img_to_dataset: dict = {}
    query_count = 0

    for query_root_path in args.query_roots:
        query_root_path = query_root_path.resolve()
        parts = query_root_path.parts
        lbl_dir = None
        if "images" in parts:
            idx = list(parts).index("images")
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels"
            lbl_dir = Path(*lbl_parts)
            if not lbl_dir.exists():
                lbl_dir = None

        imgs_here = sorted(p for p in query_root_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        query_count += len(imgs_here)
        query_dataset = _infer_query_dataset(query_root_path)

        if lbl_dir is not None:
            hard_imgs, hardness, img_to_ds = find_hard_samples_gt_aware(
                model, query_root_path, lbl_dir, target_classes,
                conf_hard=args.conf_hard, match_iou=args.match_iou,
                query_dataset=query_dataset,
            )
        else:
            print(f"  WARNING: No label dir found for {query_root_path} — skipping GT-aware mining.")
            hard_imgs, hardness, img_to_ds = [], {}, {}

        all_hard_imgs.extend(hard_imgs)
        all_hardness.update(hardness)
        all_img_to_dataset.update(img_to_ds)

    if not all_hard_imgs:
        print("  No hard samples found. Try lowering --conf-hard or --match-iou.")
        return

    print(f"  Total hard samples across all query dirs: {len(all_hard_imgs)}")
    print(f"Hard samples found: {len(all_hard_imgs)}")

    # Load DINOv2 model for embeddings
    print(f"\n[4/4] Loading DINOv2 ({dinov2_model_name}) and extracting embeddings...")
    dinov2 = load_dinov2(dinov2_model_name, device)

    # Pool embeddings (with cache)
    cache_path = args.cache_pool_embs
    pool_embs = None
    if cache_path:
        expected_meta = _dinov2_cache_meta(args.weights, dinov2_model_name, pool_paths)
        cached = load_pool_cache(cache_path, expected_meta)
        if cached is not None:
            pool_embs, pool_paths = cached

    if pool_embs is None:
        print("  Extracting DINOv2 pool embeddings...")
        pool_embs, pool_paths = extract_dinov2_embeddings(
            dinov2, pool_paths, batch_size=args.batch_size, device=device
        )
        if cache_path:
            expected_meta = _dinov2_cache_meta(args.weights, dinov2_model_name, pool_paths)
            save_pool_cache(cache_path, pool_embs, pool_paths, expected_meta)

    # Determine actual embedding dim
    if pool_embs is not None and pool_embs.ndim == 2 and pool_embs.shape[0] > 0:
        embedding_dim = pool_embs.shape[1]

    # Hard sample embeddings
    print("  Extracting DINOv2 hard sample embeddings...")
    hard_embs, all_hard_imgs = extract_dinov2_embeddings(
        dinov2, all_hard_imgs, batch_size=args.batch_size, device=device
    )

    # Retrieval
    print(f"  Retrieving (sim >= {args.similarity_threshold}, top-{args.top_k})...")
    selected, ret_stats = retrieve_dinov2(
        hard_embs=hard_embs,
        hard_paths=all_hard_imgs,
        hardness_scores=all_hardness,
        img_to_dataset=all_img_to_dataset,
        pool_embs=pool_embs,
        pool_paths=pool_paths,
        sim_threshold=args.similarity_threshold,
        top_n=args.top_k,
    )
    print(f"  Candidate hits above threshold: {ret_stats['candidate_hits_above_threshold']}")
    print(f"  Unique candidates (before top-k): {ret_stats['unique_candidates_before_top_k']}")
    print(f"  Duplicates removed: {ret_stats['duplicate_candidate_hits_removed']}")
    print(f"  Selected unique: {ret_stats['selected_unique']}")

    # Leakage check
    if args.used_bdd_root:
        used_bdd_names: set = set()
        for split in ("train", "val", "test"):
            split_dir = args.used_bdd_root / "images" / split
            if split_dir.exists():
                for p in split_dir.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        used_bdd_names.add(p.name)
        retrieved_names = {entry["pool_path"].name for entry in selected}
        overlap = retrieved_names & used_bdd_names
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: {len(overlap)} retrieved image(s) appear in used BDD30K: "
                f"{sorted(overlap)[:5]}"
            )
        overlap_count = 0
    else:
        overlap_count = -1  # not checked

    # Hard invariants
    assert ret_stats["selected_unique"] <= args.top_k, "selected_unique > top_k — invariant violated"
    retrieved_basenames = [entry["pool_path"].name for entry in selected]
    assert len(retrieved_basenames) == len(set(retrieved_basenames)), "duplicate retrieved image names"

    # Write output
    out_img_dir = out_root / "images" / "train"
    out_lbl_dir = out_root / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list = []
    for rank, entry in enumerate(selected, start=1):
        img_path = entry["pool_path"]
        sim = entry["sim"]
        query_path = entry["query_path"]
        query_dataset = entry["query_dataset"]
        hardness = entry["hardness_score"]
        lbl_path = pool_lbl_dir / f"{img_path.stem}.txt"
        place_file(img_path, out_img_dir / img_path.name, args.mode)
        if lbl_path.exists():
            place_file(lbl_path, out_lbl_dir / lbl_path.name, args.mode)
        manifest_rows.append({
            "retrieved_image": str(out_img_dir / img_path.name),
            "retrieved_label": str(out_lbl_dir / lbl_path.name) if lbl_path.exists() else "",
            "source_image_name": img_path.name,
            "source_split": "bdd_train",
            "query_image": str(query_path) if query_path else "",
            "query_dataset": query_dataset,
            "similarity": f"{sim:.6f}",
            "hardness_score": f"{hardness:.6f}" if isinstance(hardness, float) else "",
            "rank": rank,
            "selected_reason": "dinov2_cosine_retrieval",
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

    # retrieved_manifest.csv
    with (out_root / "retrieved_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "retrieved_image", "retrieved_label", "source_image_name", "source_split",
            "query_image", "query_dataset", "similarity", "hardness_score", "rank", "selected_reason",
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
    selected_unique = len(selected)

    if output_img_count != selected_unique:
        raise RuntimeError(
            f"OUTPUT COUNT MISMATCH: wrote {output_img_count} images but selected_unique={selected_unique}. "
            "Pass --clean and rerun."
        )
    if output_lbl_count != selected_unique:
        raise RuntimeError(
            f"OUTPUT LABEL COUNT MISMATCH: wrote {output_lbl_count} labels but selected_unique={selected_unique}."
        )
    if manifest_row_count != selected_unique:
        raise RuntimeError(
            f"MANIFEST COUNT MISMATCH: {manifest_row_count} rows but selected_unique={selected_unique}."
        )

    # Label validation
    violations = validate_labels(out_lbl_dir, max_class_id=5)
    if violations:
        raise RuntimeError(f"Label validation failed:\n" + "\n".join(violations[:10]))

    # retrieval_stats.json
    stats = {
        "retrieval_method": "dinov2_cosine",
        "dinov2_model": dinov2_model_name,
        "embedding_dim": embedding_dim,
        "centering_applied": False,
        "candidate_pool_size": pool_count_after_filter,
        "candidate_pool_before_class_filter": pool_count_before_filter,
        "candidate_pool_after_class_filter": pool_count_after_filter,
        "query_roots": [str(q) for q in args.query_roots],
        "query_split": "train",
        "query_count": query_count,
        "hard_query_count": len(all_hard_imgs),
        "requested_top_k": args.top_k,
        "selected_unique": selected_unique,
        "similarity_threshold": args.similarity_threshold,
        "seed": args.seed,
        "candidate_hits_above_threshold": ret_stats["candidate_hits_above_threshold"],
        "unique_candidates_before_top_k": ret_stats["unique_candidates_before_top_k"],
        "duplicate_candidate_hits_removed": ret_stats["duplicate_candidate_hits_removed"],
        "duplicate_candidates_removed": ret_stats["duplicate_candidate_hits_removed"],
        "overlap_with_used_bdd": overlap_count,
        "source_split_counts": {"bdd_train": selected_unique},
        "output_image_count": output_img_count,
        "output_label_count": output_lbl_count,
        "manifest_row_count": manifest_row_count,
    }
    (out_root / "retrieval_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. DINOv2 retrieved data → {out_root}")
    print(f"  selected_unique: {selected_unique}")
    print(f"  dinov2_model: {dinov2_model_name}  dim: {embedding_dim}")
    print(f"  source_split: all bdd_train")


if __name__ == "__main__":
    main()
