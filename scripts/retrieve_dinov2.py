#!/usr/bin/env python3
"""
DINOv2-based retrieval: RT-DETR hard mining + crop-level DINOv2 + per-class quota.

Default pipeline (--use-quota, the default):
  1. RT-DETR checkpoint → find_hard_samples_gt_aware() → hard sample paths
  2. DINOv2 → embed POOL whole images → pool_embs [N, D]
  3. DINOv2 → embed GT OBJECT CROPS from hard samples → crop_embs [Nc, D]
     ACDC hard samples are duplicated (--acdc-query-weight) to boost adverse-domain pull.
  4. Per-pool score = max_crop_sim × (rare_ratio ^ alpha)  (rare_ratio from GT labels)
  5. Per-class quota fill: bicycle=2000, motorcycle=2000, bus=1000 → deduplicated → fill to top-k

Legacy pipeline (--no-quota):
  whole-image hard-sample embedding + global top-K (original behaviour)

Strict fairness: always outputs exactly --top-k unique images or raises RuntimeError.

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
from collections import Counter
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

# HuggingFace model IDs for fallback when torch.hub is unavailable
_DINOV2_HF_IDS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}


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

class _HFDINOv2Wrapper:
    """Wraps a HuggingFace DINOv2 model to match the torch.hub forward contract.

    torch.hub DINOv2 returns [B, D] directly.
    HuggingFace DINOv2 returns a BaseModelOutputWithPooling;
    the CLS token is last_hidden_state[:, 0] — shape [B, D].
    """

    def __init__(self, hf_model):
        self._model = hf_model

    def __call__(self, pixel_values):
        outputs = self._model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]  # CLS token → [B, D]

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model.to(device)
        return self


def load_dinov2(model_name: str, device: str):
    """Load DINOv2 model. Primary: torch.hub. Fallback: HuggingFace transformers.

    Both backends are wrapped to return [B, D] float tensors from a [B, 3, H, W] input.
    Raises RuntimeError with both failure reasons if neither backend succeeds.
    """
    hub_err: Exception | None = None
    hf_err: Exception | None = None

    # Primary: torch.hub (facebookresearch/dinov2)
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
        model.eval().to(device)
        return model
    except Exception as e:
        hub_err = e

    # Fallback: HuggingFace transformers with correct model ID mapping
    try:
        from transformers import AutoModel  # type: ignore[import]
        hf_id = _DINOV2_HF_IDS.get(model_name)
        if hf_id is None:
            raise ValueError(f"No HuggingFace ID mapping for model_name={model_name!r}")
        hf_model = AutoModel.from_pretrained(hf_id)
        wrapper = _HFDINOv2Wrapper(hf_model)
        wrapper.eval().to(device)
        print(f"  torch.hub failed ({hub_err}); loaded DINOv2 from HuggingFace ({hf_id})")
        return wrapper
    except Exception as e:
        hf_err = e

    raise RuntimeError(
        f"Failed to load DINOv2 model '{model_name}' from both backends.\n"
        f"  torch.hub error:     {hub_err}\n"
        f"  HuggingFace error:   {hf_err}\n"
        "Install torch.hub access (internet + facebookresearch/dinov2) or "
        "'pip install transformers' with network access."
    )


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
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            tensors.append(t)
            kept.append(p)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = dinov2_model(batch)
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


# ── Pool rare-class statistics ────────────────────────────────────────────────

def compute_pool_rare_stats(
    pool_paths: list,
    pool_lbl_dir: "Path",
    rare_classes: set,
) -> dict:
    """
    For each pool image, compute rare-class statistics from its label file.
    Returns dict keyed by image stem:
      rare_count, total_count, rare_ratio, class_ids (set of int).
    """
    stats: dict = {}
    for img_path in pool_paths:
        lbl = pool_lbl_dir / f"{img_path.stem}.txt"
        total, rare = 0, 0
        class_ids: set = set()
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                class_ids.add(cls_id)
                total += 1
                if cls_id in rare_classes:
                    rare += 1
        stats[img_path.stem] = {
            "rare_count": rare,
            "total_count": total,
            "rare_ratio": rare / total if total > 0 else 0.0,
            "class_ids": class_ids,
        }
    return stats


# ── Crop-level query embedding ─────────────────────────────────────────────────

def extract_crop_embeddings(
    dinov2_model,
    hard_imgs: list,
    query_lbl_dirs: dict,
    img_to_dataset: dict,
    target_classes: set,
    batch_size: int,
    device: str,
    padding: float = 0.10,
    imgsz: int = 224,
    acdc_weight: int = 2,
) -> "tuple[np.ndarray, list, list, list]":
    """
    Extract DINOv2 embeddings for GT object crops from hard samples.

    Each target-class bounding box is cropped (+ padding) and embedded.
    ACDC-sourced images are duplicated acdc_weight times so they pull
    more of the retrieval budget toward adverse-condition exemplars.

    Returns (crop_embs [N,D], crop_srcs [N], crop_classes [N], crop_datasets [N]).
    """
    import cv2  # type: ignore[import]
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    crop_tensors: list = []
    crop_srcs:    list = []
    crop_cls_list: list = []
    crop_ds_list:  list = []

    def _add_image(img_path: "Path") -> None:
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]
        lbl_dir = query_lbl_dirs.get(img_path.stem)
        if lbl_dir is None:
            return
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            return
        canonical = str(img_path.resolve())
        ds = img_to_dataset.get(canonical, "")
        for line in lbl.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in target_classes:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2 - padding * bw) * w)
            y1 = int((cy - bh / 2 - padding * bh) * h)
            x2 = int((cx + bw / 2 + padding * bw) * w)
            y2 = int((cy + bh / 2 + padding * bh) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            crop_r = cv2.resize(crop, (imgsz, imgsz))
            crop_rgb = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            crop_tensors.append(t)
            crop_srcs.append(img_path)
            crop_cls_list.append(cls_id)
            crop_ds_list.append(ds)

    for img_path in hard_imgs:
        canonical = str(img_path.resolve())
        ds = img_to_dataset.get(canonical, "")
        _add_image(img_path)
        if ds == "acdc":
            for _ in range(acdc_weight - 1):
                _add_image(img_path)

    if not crop_tensors:
        return np.zeros((0, 768), dtype=np.float32), [], [], []

    embs = []
    for i in tqdm(range(0, len(crop_tensors), batch_size), desc="  DINOv2 crop embed", ncols=80):
        batch = torch.stack(crop_tensors[i: i + batch_size]).to(device)
        with torch.no_grad():
            out = dinov2_model(batch)
            if isinstance(out, dict):
                feat = out.get("last_hidden_state", out.get("pooler_output"))
                if feat is not None and feat.ndim == 3:
                    feat = feat[:, 0]
            elif isinstance(out, torch.Tensor):
                feat = out[:, 0] if out.ndim == 3 else out
            else:
                feat = out
        feat = F.normalize(feat.float(), dim=1)
        embs.append(feat.cpu().numpy())

    return np.vstack(embs), crop_srcs, crop_cls_list, crop_ds_list


# ── DINOv2 smoke check ─────────────────────────────────────────────────────────

def dinov2_smoke_check(dinov2_model, sample_image_path: Path, device: str) -> None:
    """
    Embed one real image through DINOv2. Hard-fail if:
      - image cannot be read
      - embedding contains non-finite values
      - L2 norm is not ≈ 1.0
    """
    print(f"  DINOv2 smoke check on {sample_image_path.name}...")
    embs, valid = extract_dinov2_embeddings(
        dinov2_model, [sample_image_path], batch_size=1, device=device
    )
    if len(valid) == 0:
        raise RuntimeError(f"DINO SMOKE: failed to embed {sample_image_path}")
    emb = embs[0]
    if not np.all(np.isfinite(emb)):
        raise RuntimeError("DINO SMOKE: embedding contains non-finite values (NaN or Inf)")
    norm = float(np.linalg.norm(emb))
    if abs(norm - 1.0) > 1e-3:
        raise RuntimeError(f"DINO SMOKE: L2 norm = {norm:.6f}, expected ≈ 1.0")
    print(f"  DINOv2 smoke check: PASS (dim={len(emb)}, norm={norm:.6f})")


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
    Global top-N retrieval using DINOv2 cosine similarity.

    Selection logic:
      1. Compute [H, P] cosine similarity matrix (L2-normalized embeds → dot = cosine).
      2. For each pool candidate p, score = max_h(sim[h, p]).
      3. Rank ALL pool candidates by score descending; tie-break by pool filename.
      4. Select the top_n candidates.

    sim_threshold is DIAGNOSTIC ONLY — it does not gate or reduce the selected count.
    If unique_pool_candidates < top_n, raises RuntimeError immediately.
    """
    sim_matrix = hard_embs @ pool_embs.T  # [H, P]

    # Per-pool-candidate: max cosine sim over all hard queries
    max_sim_per_pool = sim_matrix.max(axis=0)   # [P]
    argmax_per_pool = sim_matrix.argmax(axis=0)  # [P]

    # Deduplicate by stem (pool should already be unique, but guard)
    stem_to_entry: dict = {}
    for p_idx, p_path in enumerate(pool_paths):
        stem = p_path.stem
        score = float(max_sim_per_pool[p_idx])
        if stem not in stem_to_entry or stem_to_entry[stem]["sim"] < score:
            h_idx = int(argmax_per_pool[p_idx])
            hard_path = hard_paths[h_idx]
            canonical_key = str(hard_path.resolve())
            stem_to_entry[stem] = {
                "pool_path": p_path,
                "sim": score,
                "query_path": hard_path,
                "query_dataset": img_to_dataset.get(canonical_key, ""),
                "hardness_score": hardness_scores.get(canonical_key, 0.0),
            }

    unique_pool = len(stem_to_entry)

    # Hard-fail immediately if pool is too small to guarantee exact top_n
    if unique_pool < top_n:
        raise RuntimeError(
            f"EXACT-{top_n} FAIL: rare-class-filtered pool has only {unique_pool} unique candidates, "
            f"need {top_n}. Check --pool-root and --candidate-target-classes."
        )

    # Threshold is diagnostic only: count candidates above threshold
    hits_above_threshold = int(np.sum(max_sim_per_pool >= sim_threshold))

    # Deterministic sort: highest sim first; tie-break by filename
    ranked = sorted(stem_to_entry.values(), key=lambda x: (-x["sim"], x["pool_path"].name))
    selected = ranked[:top_n]
    assert len(selected) == top_n  # invariant

    # Similarity stats
    all_sims = sim_matrix.flatten()
    selected_sims = np.array([e["sim"] for e in selected], dtype=np.float64)
    contributing_queries = {str(e["query_path"].resolve()) for e in selected if e["query_path"]}
    ds_counter = Counter(e["query_dataset"] for e in selected)

    stats = {
        "candidate_hits_above_threshold": hits_above_threshold,
        "threshold_is_diagnostic_only": True,
        "unique_candidates_before_top_k": unique_pool,
        "duplicate_candidate_hits_removed": 0,  # dedup is by stem, not by hit count now
        "selected_unique": len(selected),
        "unique_contributing_hard_queries": len(contributing_queries),
        "selected_by_query_dataset": dict(ds_counter),
        # All-pair similarity distribution
        "sim_all_min": float(np.min(all_sims)),
        "sim_all_median": float(np.median(all_sims)),
        "sim_all_mean": float(np.mean(all_sims)),
        "sim_all_max": float(np.max(all_sims)),
        # Selected similarity distribution
        "sim_selected_min": float(np.min(selected_sims)),
        "sim_selected_p05": float(np.percentile(selected_sims, 5)),
        "sim_selected_median": float(np.median(selected_sims)),
        "sim_selected_p95": float(np.percentile(selected_sims, 95)),
        "sim_selected_max": float(np.max(selected_sims)),
    }
    return selected, stats


# ── Per-class quota retrieval with rare-density scoring ───────────────────────

def retrieve_dinov2_quota(
    crop_embs: "np.ndarray",
    crop_srcs: list,
    crop_datasets: list,
    pool_embs: "np.ndarray",
    pool_paths: list,
    pool_rare_stats: dict,
    class_quotas: dict,
    total_k: int,
    rare_density_alpha: float,
    sim_threshold: float,
    img_to_dataset: dict,
    rng: "random.Random",
) -> "tuple[list[dict], dict]":
    """
    Per-class quota retrieval with rare-density scoring.

    Score formula: final_score = max_crop_sim × (rare_ratio ^ alpha)
    If rare_ratio == 0 the candidate still participates (score ≈ sim × epsilon).

    Quota fill order:
      1. For each class (sorted by ascending quota), pick top candidates that
         contain that class and have not yet been selected.
      2. Fill any shortfall with the globally highest-score unselected candidates.

    Guarantees exactly total_k unique images or raises RuntimeError.
    """
    # [Nc, P] cosine similarity (already L2-normalised → dot = cosine)
    sim_matrix = crop_embs @ pool_embs.T
    max_sim_per_pool = sim_matrix.max(axis=0)    # [P]
    argmax_per_pool  = sim_matrix.argmax(axis=0) # [P]

    # Build per-pool entry with final_score
    pool_entries: dict = {}
    for p_idx, p_path in enumerate(pool_paths):
        stem = p_path.stem
        sim = float(max_sim_per_pool[p_idx])
        s = pool_rare_stats.get(stem, {"rare_count": 0, "total_count": 0,
                                       "rare_ratio": 0.0, "class_ids": set()})
        rr = s["rare_ratio"]
        final_score = sim * (rr ** rare_density_alpha) if rr > 0 else sim * 1e-3
        c_src = crop_srcs[int(argmax_per_pool[p_idx])]
        canonical = str(c_src.resolve())
        ds = img_to_dataset.get(canonical, crop_datasets[int(argmax_per_pool[p_idx])])
        if stem not in pool_entries or pool_entries[stem]["final_score"] < final_score:
            pool_entries[stem] = {
                "pool_path": p_path,
                "sim": sim,
                "final_score": final_score,
                "rare_ratio": rr,
                "rare_count": s["rare_count"],
                "class_ids": s["class_ids"],
                "query_path": c_src,
                "query_dataset": ds,
                "hardness_score": 0.0,
            }

    unique_pool = len(pool_entries)
    if unique_pool < total_k:
        raise RuntimeError(
            f"EXACT-{total_k} FAIL: rare-class pool has only {unique_pool} unique "
            f"candidates, need {total_k}. Check --pool-root and --candidate-target-classes."
        )

    selected_stems: set = set()
    selected: list = []

    # Quota fill: process classes from smallest quota first to avoid starvation
    for cls_id in sorted(class_quotas, key=lambda c: class_quotas[c]):
        quota = class_quotas[cls_id]
        candidates = [
            entry for stem, entry in pool_entries.items()
            if stem not in selected_stems and cls_id in entry["class_ids"]
        ]
        candidates.sort(key=lambda x: (-x["final_score"], x["pool_path"].name))
        for entry in candidates[:quota]:
            selected_stems.add(entry["pool_path"].stem)
            selected.append({**entry, "selected_reason": f"quota_cls{cls_id}"})

    # Global fill to reach total_k
    if len(selected) < total_k:
        remaining = [
            entry for stem, entry in pool_entries.items()
            if stem not in selected_stems
        ]
        remaining.sort(key=lambda x: (-x["final_score"], x["pool_path"].name))
        for entry in remaining[:total_k - len(selected)]:
            selected_stems.add(entry["pool_path"].stem)
            selected.append({**entry, "selected_reason": "score_fill"})

    if len(selected) < total_k:
        raise RuntimeError(
            f"EXACT-{total_k} FAIL: only {len(selected)} after quota+fill. "
            "Expand pool or lower quotas."
        )
    selected = selected[:total_k]
    assert len(selected) == total_k

    sel_sims = np.array([e["sim"] for e in selected], dtype=np.float64)
    ds_counter = Counter(e["query_dataset"] for e in selected)
    reason_counter = Counter(e["selected_reason"] for e in selected)
    hits = int(np.sum(max_sim_per_pool >= sim_threshold))

    stats_out = {
        "candidate_hits_above_threshold": hits,
        "threshold_is_diagnostic_only": True,
        "unique_candidates_before_top_k": unique_pool,
        "duplicate_candidate_hits_removed": 0,
        "selected_unique": len(selected),
        "unique_contributing_hard_queries": len(
            {str(e["query_path"].resolve()) for e in selected if e["query_path"]}
        ),
        "selected_by_query_dataset": dict(ds_counter),
        "selected_by_reason": dict(reason_counter),
        "sim_all_min": float(np.min(max_sim_per_pool)),
        "sim_all_median": float(np.median(max_sim_per_pool)),
        "sim_all_mean": float(np.mean(max_sim_per_pool)),
        "sim_all_max": float(np.max(max_sim_per_pool)),
        "sim_selected_min": float(np.min(sel_sims)),
        "sim_selected_p05": float(np.percentile(sel_sims, 5)),
        "sim_selected_median": float(np.median(sel_sims)),
        "sim_selected_p95": float(np.percentile(sel_sims, 95)),
        "sim_selected_max": float(np.max(sel_sims)),
    }
    return selected, stats_out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 retrieval: RT-DETR hard mining + global top-K cosine similarity"
    )
    p.add_argument("--weights", required=True, help="RT-DETR checkpoint for hard mining")
    p.add_argument("--pool-root", required=True, type=Path,
                   help="BDD remaining pool (output of build_bdd_retrieval_pool.py)")
    p.add_argument("--query-root", action="append", dest="query_roots", type=Path, required=True,
                   help="Train-split image directories (repeatable). Must be train splits only.")
    p.add_argument("--out-root", required=True, type=Path, help="Output directory")
    p.add_argument("--dinov2-model", default="dinov2_vitb14", choices=DINOV2_CHOICES,
                   help="DINOv2 model variant (default: dinov2_vitb14)")
    p.add_argument("--top-k", type=int, default=5000,
                   help="Exact number of images to select (guaranteed; hard-fail if pool smaller)")
    p.add_argument("--similarity-threshold", type=float, default=0.70,
                   help="Cosine similarity threshold — DIAGNOSTIC ONLY, does not filter (default: 0.70)")
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
    # ── Quota / rare-density options ──────────────────────────────────────
    p.add_argument("--rare-density-alpha", type=float, default=0.5,
                   help="Exponent for rare_ratio in final_score = sim × rare_ratio^alpha (default: 0.5)")
    p.add_argument("--acdc-query-weight", type=int, default=2,
                   help="Repeat ACDC crops this many times in query set (default: 2)")
    p.add_argument("--quota-bicycle", type=int, default=2000)
    p.add_argument("--quota-motorcycle", type=int, default=2000)
    p.add_argument("--quota-bus", type=int, default=1000)
    p.add_argument("--use-quota", action="store_true", default=True,
                   help="Use per-class quota + rare-density scoring (default: True)")
    p.add_argument("--no-quota", dest="use_quota", action="store_false",
                   help="Fall back to global top-K without quota or density scoring")
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
    print("DINOv2 Retrieval (RT-DETR hard mining + global top-K cosine similarity)")
    print(f"  weights:              {args.weights}")
    print(f"  pool-root:            {pool_root}")
    print(f"  query-roots:          {[str(q) for q in args.query_roots]}")
    print(f"  top-k:                {args.top_k}  [guaranteed exact]")
    print(f"  similarity-threshold: {args.similarity_threshold}  [diagnostic only]")
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
    print(f"[1/5] Pool size: {len(pool_paths)} images")

    # Candidate class filter
    if use_class_filter and candidate_target_classes:
        print(f"  Filtering pool to images with classes {sorted(candidate_target_classes)}...")
        pool_paths, pool_count_before_filter = filter_pool_by_class(
            pool_img_dir, pool_lbl_dir, candidate_target_classes
        )
        print(f"  Pool after class filter: {len(pool_paths)} / {pool_count_before_filter}")
    pool_count_after_filter = len(pool_paths)
    print(f"Pool size (after class filter): {pool_count_after_filter}")

    if pool_count_after_filter < args.top_k:
        raise RuntimeError(
            f"EXACT-{args.top_k} FAIL: rare-class-filtered pool has only "
            f"{pool_count_after_filter} candidates, need {args.top_k}. "
            "Check --pool-root and --candidate-target-classes."
        )

    # Load DINOv2 model BEFORE hard mining so smoke check uses a real XWOD image
    print(f"\n[2/5] Loading DINOv2 ({dinov2_model_name})...")
    dinov2 = load_dinov2(dinov2_model_name, device)

    # DINOv2 smoke check: embed one real train image before any expensive work
    smoke_img: Path | None = None
    for qr in args.query_roots:
        for p in sorted(qr.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS and (p.is_file() or p.is_symlink()):
                smoke_img = p
                break
        if smoke_img:
            break
    if smoke_img:
        dinov2_smoke_check(dinov2, smoke_img, device)
    else:
        print("  WARNING: no real query image found for DINOv2 smoke check — skipping")

    # Load RT-DETR model for hard mining
    print("\n[3/5] Loading RT-DETR checkpoint for hard mining...")
    model = YOLO(str(args.weights))
    model.to(device)

    # Hard mining
    print("\n[4/5] GT-aware hard mining from query dirs...")
    all_hard_imgs: list = []
    all_hardness: dict = {}
    all_img_to_dataset: dict = {}
    query_lbl_dirs: dict = {}  # stem -> Path  (used for crop embedding)
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
            for p in hard_imgs:
                query_lbl_dirs[p.stem] = lbl_dir
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

    # Pool and query embeddings
    print(f"\n[5/5] Extracting DINOv2 embeddings...")
    pool_embs = None
    cache_path = args.cache_pool_embs
    if cache_path:
        expected_meta = _dinov2_cache_meta(args.weights, dinov2_model_name, pool_paths)
        cached = load_pool_cache(cache_path, expected_meta)
        if cached is not None:
            pool_embs, pool_paths = cached

    if pool_embs is None:
        print("  Extracting DINOv2 pool embeddings (whole-image)...")
        pool_embs, pool_paths = extract_dinov2_embeddings(
            dinov2, pool_paths, batch_size=args.batch_size, device=device
        )
        if cache_path:
            expected_meta = _dinov2_cache_meta(args.weights, dinov2_model_name, pool_paths)
            save_pool_cache(cache_path, pool_embs, pool_paths, expected_meta)

    if pool_embs is not None and pool_embs.ndim == 2 and pool_embs.shape[0] > 0:
        embedding_dim = pool_embs.shape[1]

    if args.use_quota:
        # ── NEW: crop-level queries + rare-density + per-class quota ──────
        print(f"  Mode: crop-level query embedding + rare-density scoring + per-class quota")
        print(f"  Extracting DINOv2 crop embeddings (acdc_weight={args.acdc_query_weight})...")
        crop_embs, crop_srcs, crop_cls_list, crop_ds_list = extract_crop_embeddings(
            dinov2, all_hard_imgs, query_lbl_dirs, all_img_to_dataset,
            target_classes, args.batch_size, device,
            acdc_weight=args.acdc_query_weight,
        )
        if crop_embs.shape[0] == 0:
            raise RuntimeError(
                "No crop embeddings extracted — no GT boxes for target classes found in hard samples. "
                "Check --target-classes and query label dirs."
            )
        print(f"  Crop embeddings: {crop_embs.shape[0]} crops from {len(all_hard_imgs)} hard images")

        # Pool rare stats (needed for density scoring and quota)
        print("  Computing pool rare-class statistics...")
        pool_rare_stats = compute_pool_rare_stats(
            pool_paths, pool_lbl_dir, set(args.target_classes)
        )
        rare_multi = sum(
            1 for s in pool_rare_stats.values()
            if s["rare_count"] >= 2 or s["rare_ratio"] >= 0.3
        )
        print(f"  Pool images with rare_count≥2 or rare_ratio≥0.3: {rare_multi}/{len(pool_paths)}")

        class_quotas = {
            1: args.quota_bicycle,
            3: args.quota_motorcycle,
            4: args.quota_bus,
        }
        print(f"  Class quotas: bicycle={class_quotas[1]} motorcycle={class_quotas[3]} bus={class_quotas[4]}")
        print(f"  Rare-density alpha: {args.rare_density_alpha}")

        selected, ret_stats = retrieve_dinov2_quota(
            crop_embs=crop_embs,
            crop_srcs=crop_srcs,
            crop_datasets=crop_ds_list,
            pool_embs=pool_embs,
            pool_paths=pool_paths,
            pool_rare_stats=pool_rare_stats,
            class_quotas=class_quotas,
            total_k=args.top_k,
            rare_density_alpha=args.rare_density_alpha,
            sim_threshold=args.similarity_threshold,
            img_to_dataset=all_img_to_dataset,
            rng=random.Random(args.seed),
        )
        print(f"  Selected by reason: {ret_stats['selected_by_reason']}")
    else:
        # ── Legacy: whole-image queries + global top-K ──────────────────
        print("  Mode: whole-image query embedding + global top-K (legacy, --no-quota)")
        hard_embs, all_hard_imgs = extract_dinov2_embeddings(
            dinov2, all_hard_imgs, batch_size=args.batch_size, device=device
        )
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

    print(f"  Unique pool candidates: {ret_stats['unique_candidates_before_top_k']}")
    print(f"  Candidates above threshold (diagnostic): {ret_stats['candidate_hits_above_threshold']}")
    print(f"  Unique contributing hard queries: {ret_stats['unique_contributing_hard_queries']}")
    print(f"  Selected unique: {ret_stats['selected_unique']}")
    print(f"  Selected similarity: "
          f"min={ret_stats['sim_selected_min']:.4f} "
          f"p05={ret_stats['sim_selected_p05']:.4f} "
          f"median={ret_stats['sim_selected_median']:.4f} "
          f"p95={ret_stats['sim_selected_p95']:.4f} "
          f"max={ret_stats['sim_selected_max']:.4f}")
    print(f"  Provenance: {ret_stats['selected_by_query_dataset']}")

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
    assert ret_stats["selected_unique"] == args.top_k, \
        f"selected_unique={ret_stats['selected_unique']} != top_k={args.top_k} — invariant violated"
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
        hardness = entry.get("hardness_score", 0.0)
        sel_reason = entry.get("selected_reason", "dinov2_cosine_global_topk")
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
            "hardness_score": f"{hardness:.6f}" if isinstance(hardness, (int, float)) else "",
            "rank": rank,
            "selected_reason": sel_reason,
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
        raise RuntimeError("Label validation failed:\n" + "\n".join(violations[:10]))

    # retrieval_stats.json
    stats = {
        "retrieval_method": "dinov2_crop_quota_rare_density" if args.use_quota else "dinov2_cosine_global_topk",
        "use_quota": args.use_quota,
        "rare_density_alpha": args.rare_density_alpha if args.use_quota else None,
        "acdc_query_weight": args.acdc_query_weight if args.use_quota else None,
        "class_quotas": {"1_bicycle": args.quota_bicycle, "3_motorcycle": args.quota_motorcycle,
                         "4_bus": args.quota_bus} if args.use_quota else None,
        "dinov2_model": dinov2_model_name,
        "embedding_dim": embedding_dim,
        "centering_applied": False,
        "threshold_is_diagnostic_only": True,
        "similarity_threshold_diagnostic": args.similarity_threshold,
        "candidate_pool_size": pool_count_after_filter,
        "candidate_pool_before_class_filter": pool_count_before_filter,
        "candidate_pool_after_class_filter": pool_count_after_filter,
        "query_roots": [str(q) for q in args.query_roots],
        "query_split": "train",
        "query_count": query_count,
        "hard_query_count": len(all_hard_imgs),
        "requested_top_k": args.top_k,
        "selected_unique": selected_unique,
        "seed": args.seed,
        "candidate_hits_above_threshold": ret_stats["candidate_hits_above_threshold"],
        "unique_candidates_before_top_k": ret_stats["unique_candidates_before_top_k"],
        "duplicate_candidate_hits_removed": ret_stats["duplicate_candidate_hits_removed"],
        "unique_contributing_hard_queries": ret_stats["unique_contributing_hard_queries"],
        "selected_by_query_dataset": ret_stats["selected_by_query_dataset"],
        "selected_by_reason": ret_stats.get("selected_by_reason", {}),
        "sim_all_min": ret_stats["sim_all_min"],
        "sim_all_median": ret_stats["sim_all_median"],
        "sim_all_mean": ret_stats["sim_all_mean"],
        "sim_all_max": ret_stats["sim_all_max"],
        "sim_selected_min": ret_stats["sim_selected_min"],
        "sim_selected_p05": ret_stats["sim_selected_p05"],
        "sim_selected_median": ret_stats["sim_selected_median"],
        "sim_selected_p95": ret_stats["sim_selected_p95"],
        "sim_selected_max": ret_stats["sim_selected_max"],
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
    print(f"  selected_unique: {selected_unique}  (exactly top_k={args.top_k})")
    print(f"  dinov2_model: {dinov2_model_name}  dim: {embedding_dim}")
    print(f"  source_split: all bdd_train")


if __name__ == "__main__":
    main()
