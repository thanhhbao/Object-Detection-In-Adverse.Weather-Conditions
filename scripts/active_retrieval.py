#!/usr/bin/env python3
"""
Active learning retrieval: hard sample mining → cosine similarity → pool expansion.

Pipeline:
  1. Load YOLO/RT-DETR checkpoint, build embedding index for all pool images
  2. Run GT-aware hard mining on query images (TRAINING DATA ONLY)
  3. Extract multi-scale embeddings for hard samples
  4. Cosine similarity: hard_embs @ pool_embs.T
  5. Select pool images with sim >= threshold → output dir for retraining

WARNING: --query-root must point to TRAIN-split directories only. Using val or test
splits would constitute a data-leakage violation. The CLI enforces this via
validate_query_root().

Class order (project): 0=person 1=bicycle 2=car 3=motorcycle 4=bus 5=truck

New CLI usage (P2-A1 — RT-DETR aware):
  python scripts/active_retrieval.py \\
    --weights   runs/phase2_final_rtdetr/weights/best.pt \\
    --pool-root /data/bdd_remaining_pool \\
    --query-root /data/xwod_6cls_yolo/images/train \\
    --query-root /data/acdc_6cls_yolo/images/train \\
    --out-root  /data/bdd_active_retrieved \\
    --top-k 5000 --similarity-threshold 0.75 \\
    --target-classes 1 3 4 \\
    --candidate-target-classes 1 3 4 \\
    --embedding-layers 21 24 27 \\
    --match-iou 0.5 \\
    --seed 42 \\
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


# ── Legacy backbone hook ───────────────────────────────────────────────────────

class BackboneHook:
    """Hooks layer[layer_idx] of a YOLO model, returns GAP-pooled features.
    Kept for legacy mode backward compatibility.
    """

    def __init__(self, model: "YOLO", layer_idx: int = 9):
        self.feat = None
        self._h = model.model.model[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: [B, C, H, W] → mean pool → [B, C]
        self.feat = out.mean(dim=[2, 3])

    def remove(self):
        self._h.remove()


# ── Multi-scale hook (new-style, P2-A1) ───────────────────────────────────────

class MultiScaleHook:
    """Hook multiple layers, GAP each output, concatenate → L2-normalize.

    Designed for RT-DETR-L layers 21 (P3/RepC3), 24 (P4/RepC3), 27 (P5/RepC3),
    each 256-channel after RepC3. After GAP+concat: 768-dim embedding.
    """

    def __init__(self, model: "YOLO", layer_indices: list):
        self.feats: dict = {}
        self._handles = []
        for idx in layer_indices:
            layer = model.model.model[idx]
            h = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(h)

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            if _torch_available and torch is not None:
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    self.feats[idx] = out.mean(dim=[2, 3])  # GAP → [B, C]
                elif isinstance(out, (list, tuple)) and len(out) > 0:
                    first = out[0]
                    if isinstance(first, torch.Tensor) and first.ndim == 4:
                        self.feats[idx] = first.mean(dim=[2, 3])
        return hook

    def get_embedding(self, layer_indices: list) -> "torch.Tensor":
        parts = []
        for idx in layer_indices:
            if idx not in self.feats:
                raise RuntimeError(
                    f"Layer {idx} hook did not produce output. Check layer index."
                )
            parts.append(self.feats[idx])
        cat = torch.cat(parts, dim=1)  # [B, sum(C_i)]
        return F.normalize(cat, dim=1)

    def clear(self):
        self.feats.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# ── Image preprocessing ────────────────────────────────────────────────────────

def preprocess_image(path: Path, imgsz: int = IMGSZ) -> "torch.Tensor | None":
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t


# ── Legacy embedding extractor ─────────────────────────────────────────────────

def extract_embeddings(
    model: "YOLO",
    hook: "BackboneHook",
    paths: list,
    batch_size: int = 32,
    device: str = "cuda",
) -> "tuple[np.ndarray, list]":
    """Legacy: Return (N, C) L2-normalized embeddings and matched path list."""
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


# ── Multi-scale embedding extractor (new-style) ────────────────────────────────

def extract_embeddings_multi(
    model: "YOLO",
    hook: "MultiScaleHook",
    layer_indices: list,
    paths: list,
    batch_size: int = 32,
    device: str = "cuda",
    has_rtdetr: bool = False,
) -> "tuple[np.ndarray, list]":
    """New-style: multi-scale GAP embeddings, L2-normalized. Returns (N, D) array."""
    embs, valid_paths = [], []
    first_batch = True

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
            model.model(batch)  # triggers hooks

        if first_batch:
            shapes = {idx: list(hook.feats[idx].shape) for idx in layer_indices if idx in hook.feats}
            total_dim = sum(v[-1] for v in shapes.values())
            model_type = "RT-DETR (RTDETRDecoder)" if has_rtdetr else "YOLO/other"
            print(f"  Embedding model type: {model_type}")
            print(f"  Embedding layers: {layer_indices}")
            print(f"  Layer output shapes (after GAP): {shapes}")
            print(f"  Final embedding dimension: {total_dim}")
            first_batch = False

        emb = hook.get_embedding(layer_indices)  # [B, D] L2-normalized
        embs.append(emb.cpu().numpy())
        hook.clear()
        valid_paths.extend(kept)

    if not embs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.vstack(embs), valid_paths


# ── Hard sample finder (legacy) ────────────────────────────────────────────────

def has_target_in_label(label_path: Path, target_classes: set) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in target_classes:
            return True
    return False


def find_hard_samples(
    model: "YOLO",
    image_dir: Path,
    label_dir: "Path | None",
    target_classes: set,
    conf_hard: float,
    batch_size: int = 1,
) -> "tuple[list, list, dict]":
    """
    Legacy hard sample mining: image has target class in GT but model detects with
    max confidence < conf_hard (or no detection at all).

    Returns: (hard_img_paths, hard_lbl_paths, hardness_scores)
    hardness_scores: {img_name -> max_conf on target class}
    """
    img_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    hard_imgs, hard_lbls = [], []
    hardness_scores: dict = {}
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


# ── GT-aware IoU utilities ─────────────────────────────────────────────────────

def compute_iou(
    gt_box_xywh_norm: list,
    pred_boxes_xyxy_abs: "np.ndarray",
    img_w: int,
    img_h: int,
) -> "np.ndarray":
    """Compute IoU between one GT box (YOLO xywh normalized) and N pred boxes (xyxy absolute)."""
    cx, cy, bw, bh = gt_box_xywh_norm
    gx1 = (cx - bw / 2) * img_w
    gy1 = (cy - bh / 2) * img_h
    gx2 = (cx + bw / 2) * img_w
    gy2 = (cy + bh / 2) * img_h

    px1 = pred_boxes_xyxy_abs[:, 0]
    py1 = pred_boxes_xyxy_abs[:, 1]
    px2 = pred_boxes_xyxy_abs[:, 2]
    py2 = pred_boxes_xyxy_abs[:, 3]

    ix1 = np.maximum(gx1, px1)
    iy1 = np.maximum(gy1, py1)
    ix2 = np.minimum(gx2, px2)
    iy2 = np.minimum(gy2, py2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    gt_area = (gx2 - gx1) * (gy2 - gy1)
    pred_area = (px2 - px1) * (py2 - py1)
    union = gt_area + pred_area - inter
    return np.where(union > 0, inter / union, 0.0)


def is_gt_hard(
    gt_cls: int,
    gt_box: list,
    pred_classes: list,
    pred_confs: list,
    pred_boxes_xyxy: "np.ndarray",
    img_w: int,
    img_h: int,
    conf_hard: float,
    match_iou: float,
) -> "tuple[bool, float]":
    """
    A GT box is hard if:
    A. No same-class prediction with IoU >= match_iou exists (missed GT → matched_conf = 0)
    OR
    B. Best same-class prediction IoU >= match_iou but confidence < conf_hard (weak detection)

    Returns: (is_hard, matched_confidence)
    matched_confidence = 0.0 if missed, else best-matched prediction confidence
    """
    same_class_mask = np.array([c == gt_cls for c in pred_classes], dtype=bool)
    if not np.any(same_class_mask):
        return True, 0.0  # missed GT entirely

    sc_boxes = pred_boxes_xyxy[same_class_mask]
    sc_confs = np.array(pred_confs)[same_class_mask]
    ious = compute_iou(gt_box, sc_boxes, img_w, img_h)
    matched = ious >= match_iou
    if not np.any(matched):
        return True, 0.0  # no matching prediction → missed

    best_conf = float(sc_confs[matched].max())
    if best_conf < conf_hard:
        return True, best_conf  # weak detection
    return False, best_conf  # detected well


def find_hard_samples_gt_aware(
    model: "YOLO",
    image_dir: Path,
    label_dir: Path,
    target_classes: set,
    conf_hard: float,
    match_iou: float,
    query_dataset: str = "",
) -> "tuple[list, dict, dict]":
    """
    GT-aware hard sample mining with per-GT-box IoU matching.

    Hardness score = max over hard GTs of (1 - matched_confidence)
    A completely missed GT has matched_confidence=0, hardness=1.0.

    Returns: (hard_img_paths, hardness_scores, img_to_dataset)
    img_to_dataset: {img_name -> query_dataset string}
    """
    img_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    hard_imgs = []
    hardness_scores: dict = {}
    img_to_dataset: dict = {}
    skipped_no_gt = 0
    skipped_easy = 0

    for img_path in tqdm(img_paths, desc=f"  Hard mining ({query_dataset or image_dir.name})", ncols=80):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            skipped_no_gt += 1
            continue

        # Parse GT boxes for target classes
        gt_boxes = []  # (cls, [cx, cy, w, h])
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                if cls in target_classes:
                    gt_boxes.append((cls, [float(x) for x in parts[1:5]]))

        if not gt_boxes:
            skipped_no_gt += 1
            continue

        # Run inference low-conf to see all detections
        results = model.predict(str(img_path), verbose=False, conf=0.01, imgsz=IMGSZ)
        pred_classes: list = []
        pred_confs: list = []
        pred_boxes_xyxy: list = []
        img_h_pred = IMGSZ
        img_w_pred = IMGSZ

        for r in results:
            if r.boxes is not None and len(r.boxes):
                img_h_pred, img_w_pred = r.orig_shape
                pred_classes = [int(c) for c in r.boxes.cls.cpu().tolist()]
                pred_confs = r.boxes.conf.cpu().tolist()
                pred_boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            else:
                img_h_pred, img_w_pred = IMGSZ, IMGSZ

        pred_boxes_xyxy_arr = np.array(pred_boxes_xyxy) if len(pred_boxes_xyxy) > 0 else np.zeros((0, 4))

        # Check each GT box
        max_hardness = 0.0
        image_is_hard = False
        for gt_cls, gt_box in gt_boxes:
            hard, matched_conf = is_gt_hard(
                gt_cls, gt_box, pred_classes, pred_confs,
                pred_boxes_xyxy_arr, img_w_pred, img_h_pred,
                conf_hard=conf_hard, match_iou=match_iou,
            )
            if hard:
                image_is_hard = True
                hardness = 1.0 - matched_conf  # 1.0 = completely missed
                max_hardness = max(max_hardness, hardness)

        if image_is_hard:
            hard_imgs.append(img_path)
            hardness_scores[img_path.name] = max_hardness
            img_to_dataset[img_path.name] = query_dataset
        else:
            skipped_easy += 1

    names = [CLASS_NAMES[c] for c in sorted(target_classes) if c < len(CLASS_NAMES)]
    print(f"  Hard: {len(hard_imgs)}/{len(img_paths)} "
          f"(skipped {skipped_no_gt} no-GT, {skipped_easy} easy; classes: {names})")
    return hard_imgs, hardness_scores, img_to_dataset


# ── Candidate class filter ─────────────────────────────────────────────────────

def filter_pool_by_class(
    pool_img_dir: Path,
    pool_lbl_dir: Path,
    target_classes: set,
) -> "tuple[list, int]":
    """Return pool images that contain at least one target class in their label."""
    all_paths = sorted(p for p in pool_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    before = len(all_paths)
    filtered = []
    for img in all_paths:
        lbl = pool_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if parts and int(float(parts[0])) in target_classes:
                    filtered.append(img)
                    break
    return filtered, before


# ── Legacy retrieval ───────────────────────────────────────────────────────────

def retrieve_from_pool(
    hard_embs: "np.ndarray",
    pool_embs: "np.ndarray",
    pool_paths: list,
    sim_threshold: float,
    top_n: int,
    exclude_stems: "set | None" = None,
) -> "list[tuple[Path, float]]":
    """
    Legacy: For each hard sample, find pool images with cosine sim >= threshold.
    Returns deduplicated list of (pool_path, max_sim_score) sorted by score desc.
    Limits to top_n total. Uses (-sim, name) tie-breaking for determinism.
    """
    sim_matrix = hard_embs @ pool_embs.T  # both L2-normalized → cosine sim

    selected: dict = {}  # stem → (path, score)

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


# ── New-style retrieval with provenance ────────────────────────────────────────

def retrieve_from_pool_with_provenance(
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
    New-style retrieval with query provenance tracking.

    Returns:
      selected: list of dicts with keys:
        pool_path, sim, query_path, query_dataset, hardness_score
      stats: dict with dedup statistics
    """
    sim_matrix = hard_embs @ pool_embs.T  # [H, P] cosine similarities

    # stem → {pool_path, sim, query_path, query_dataset, hardness_score}
    best_per_stem: dict = {}
    candidate_hits_above_threshold = 0

    for h_idx, hard_path in enumerate(hard_paths):
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
                    "query_dataset": img_to_dataset.get(hard_path.name, ""),
                    "hardness_score": hardness_scores.get(hard_path.name, 0.0),
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


# ── Query root validation ──────────────────────────────────────────────────────

def validate_query_root(qr: Path) -> Path:
    """Require query-root to be */images/train. Fail if val or test."""
    qr = qr.resolve()
    parts = qr.parts
    if "images" not in parts:
        raise SystemExit(
            f"ERROR: --query-root must be inside an 'images/' directory: {qr}"
        )
    idx = list(parts).index("images")
    split = parts[idx + 1] if idx + 1 < len(parts) else ""
    if split in ("val", "test"):
        raise SystemExit(
            f"ERROR: --query-root points to '{split}' split: {qr}\n"
            "Querying val/test splits is a leakage violation. Pass only train splits."
        )
    if split != "train":
        raise SystemExit(
            f"ERROR: --query-root last path component under 'images/' must be 'train', "
            f"got '{split}': {qr}"
        )
    lbl_dir = Path(*list(parts)[:idx]) / "labels" / "train"
    if not lbl_dir.exists():
        raise SystemExit(f"ERROR: Expected label dir does not exist: {lbl_dir}")
    return qr


# ── Cache metadata helpers ─────────────────────────────────────────────────────

def _cache_fingerprint(
    weights_path: Path,
    layer_indices: list,
    imgsz: int,
    pool_count: int,
) -> dict:
    import hashlib
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
        "embedding_layers": layer_indices,
        "imgsz": imgsz,
        "pool_count": pool_count,
        "version": 2,
    }


def load_pool_cache(
    cache_path: Path,
    expected_meta: dict,
) -> "tuple[np.ndarray, list] | None":
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        if "meta" not in data:
            print("  Cache missing metadata — rebuilding.")
            return None
        cached_meta = json.loads(str(data["meta"]))
        mismatches = {k for k in expected_meta if cached_meta.get(k) != expected_meta[k]}
        if mismatches:
            print(f"  Cache metadata mismatch on {mismatches} — rebuilding.")
            return None
        pool_embs = data["embs"]
        pool_paths = [Path(str(p)) for p in data["paths"]]
        print(f"  Loaded {len(pool_paths)} cached embeddings (metadata verified).")
        return pool_embs, pool_paths
    except Exception as e:
        print(f"  Cache load failed ({e}) — rebuilding.")
        return None


def save_pool_cache(
    cache_path: Path,
    embs: "np.ndarray",
    paths: list,
    meta: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez(
        cache_path,
        embs=embs,
        paths=np.array([str(p) for p in paths]),
        meta=np.array(meta_json),
    )
    print(f"  Pool embeddings cached → {cache_path}")


# ── Export (legacy) ────────────────────────────────────────────────────────────

def export_retrieved(
    selected: "list[tuple[Path, float]]",
    pool_label_dir: "Path | None",
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

def count_class_dist(label_dir: Path, target_classes: set) -> dict:
    counts = {c: 0 for c in target_classes}
    for lbl in label_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                c = int(parts[0])
                if c in target_classes:
                    counts[c] += 1
    return counts


def validate_labels(label_dir: Path, max_class_id: int = 5) -> list:
    """Check all label files for invalid class IDs. Returns list of violation messages."""
    violations: list = []
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
    p = argparse.ArgumentParser(
        description="Active retrieval: GT-aware hard mining + cosine similarity pool search (P2-A1, RT-DETR aware)"
    )
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
    p.add_argument("--match-iou", type=float, default=0.5,
                   help="IoU threshold for GT-aware hard mining (default: 0.5)")
    p.add_argument("--embedding-layers", nargs="+", type=int, default=[21, 24, 27],
                   help="Layer indices to hook for multi-scale embedding (default: 21 24 27 for RT-DETR-L P3/P4/P5)")
    p.add_argument("--candidate-target-classes", nargs="+", type=int, default=[1, 3, 4],
                   help="Filter pool by label content; use -1 to disable (default: 1 3 4)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--cache-pool-embs", type=Path, default=None,
                   help="Path to save/load pool embeddings cache (.npz) with metadata validation")
    p.add_argument("--used-bdd-root", type=Path, default=None,
                   help="BDD30K root already in project — used for post-selection leakage check")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--smoke-check", action="store_true",
                   help="Load model, run one forward on dummy image, print shapes, exit 0.")
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
    """Return True if --weights / --pool-root / --query-root style args are present."""
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


def _infer_query_dataset(query_root: Path) -> str:
    """Infer dataset name from path: look for 'xwod' or 'acdc' (case-insensitive)."""
    path_str = str(query_root).lower()
    if "xwod" in path_str:
        return "xwod"
    if "acdc" in path_str:
        return "acdc"
    return query_root.parent.name


def main_new(args: argparse.Namespace) -> None:
    """P2-A1 main: new CLI interface with RT-DETR multi-scale embedding."""
    target_classes = set(args.target_classes)
    embedding_layers = args.embedding_layers
    match_iou = args.match_iou
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    _ = rng  # seed used for future random operations
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Candidate target class filter
    candidate_target_classes_raw = args.candidate_target_classes
    use_class_filter = candidate_target_classes_raw != [-1]
    candidate_target_classes = set(candidate_target_classes_raw) if use_class_filter else None

    print(f"\n{'='*60}")
    print("Active Retrieval (P2-A1 — RT-DETR aware, GT-matched, failure-driven)")
    print(f"  weights:         {args.weights}")
    print(f"  pool-root:       {args.pool_root}")
    print(f"  query-roots:     {[str(q) for q in args.query_roots]}")
    print(f"  top-k:           {args.top_k}")
    print(f"  sim-thresh:      {args.similarity_threshold}")
    print(f"  target:          {[CLASS_NAMES[c] for c in sorted(target_classes)]}")
    print(f"  embedding-layers: {embedding_layers}")
    print(f"  match-iou:       {match_iou}")
    print(f"  seed:            {args.seed}")
    print(f"  device:          {device}")
    print(f"{'='*60}\n")

    # Validate query roots (train-only enforcement)
    validated_roots = []
    for qr in args.query_roots:
        validated_roots.append(validate_query_root(qr))
    args.query_roots = validated_roots

    # Smoke check
    if args.smoke_check:
        print("[SMOKE] Loading model for smoke check...")
        model = YOLO(str(args.weights))
        model.to(device)

        try:
            from ultralytics.nn.modules.head import RTDETRDecoder  # type: ignore[import]
            has_rtdetr = any(isinstance(m, RTDETRDecoder) for m in model.model.model)
        except ImportError:
            has_rtdetr = False

        hook = MultiScaleHook(model, embedding_layers)
        dummy = torch.zeros(1, 3, IMGSZ, IMGSZ).to(device)
        with torch.no_grad():
            model.model(dummy)
        shapes = {idx: list(hook.feats[idx].shape) for idx in embedding_layers if idx in hook.feats}
        total_dim = sum(v[-1] for v in shapes.values())
        model_type = "RT-DETR (RTDETRDecoder)" if has_rtdetr else "YOLO/other"
        print(f"  Model type: {model_type}")
        print(f"  Embedding layers: {embedding_layers}")
        print(f"  Layer output shapes (after GAP): {shapes}")
        print(f"  Final embedding dimension: {total_dim}")
        hook.remove()
        print("SMOKE CHECK: PASS")
        raise SystemExit(0)

    # Pool paths
    pool_img_dir = args.pool_root / "images" / "train"
    pool_lbl_dir = args.pool_root / "labels" / "train"
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

    # Load model
    print("[2/4] Loading checkpoint...")
    model = YOLO(str(args.weights))
    model.to(device)

    # Detect RT-DETR
    has_rtdetr = False
    try:
        from ultralytics.nn.modules.head import RTDETRDecoder  # type: ignore[import]
        has_rtdetr = any(isinstance(m, RTDETRDecoder) for m in model.model.model)
    except ImportError:
        pass

    if embedding_layers != [9]:  # multi-layer mode
        if not has_rtdetr:
            print(f"  WARNING: --embedding-layers {embedding_layers} specified but no RTDETRDecoder found.")
            print("  For YOLO models, use --embedding-layers 9 (single layer).")
    model_type_str = "RT-DETR-L (RTDETRDecoder detected)" if has_rtdetr else "YOLO/other"
    print(f"  Model type: {model_type_str}")

    hook = MultiScaleHook(model, embedding_layers)

    # Pool embeddings (with cache metadata validation)
    cache_path = args.cache_pool_embs
    pool_embs = None
    if cache_path:
        expected_meta = _cache_fingerprint(args.weights, embedding_layers, IMGSZ, len(pool_paths))
        cached = load_pool_cache(cache_path, expected_meta)
        if cached is not None:
            pool_embs, pool_paths = cached
    if pool_embs is None:
        pool_embs, pool_paths = extract_embeddings_multi(
            model, hook, embedding_layers, pool_paths,
            batch_size=args.batch_size, device=device, has_rtdetr=has_rtdetr,
        )
        if cache_path:
            expected_meta = _cache_fingerprint(args.weights, embedding_layers, IMGSZ, len(pool_paths))
            save_pool_cache(cache_path, pool_embs, pool_paths, expected_meta)

    # Hard sample mining from all query dirs
    print("\n[3/4] GT-aware hard mining from query dirs...")
    all_hard_imgs: list = []
    all_hardness: dict = {}
    all_img_to_dataset: dict = {}
    query_count = 0

    for query_root in args.query_roots:
        query_root = query_root.resolve()
        # Infer label dir: images/train → labels/train
        parts = query_root.parts
        lbl_dir = None
        if "images" in parts:
            idx = list(parts).index("images")
            lbl_parts = list(parts)
            lbl_parts[idx] = "labels"
            lbl_dir = Path(*lbl_parts)
            if not lbl_dir.exists():
                lbl_dir = None

        imgs_here = sorted(p for p in query_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        query_count += len(imgs_here)
        query_dataset = _infer_query_dataset(query_root)

        if lbl_dir is not None:
            hard_imgs, hardness, img_to_ds = find_hard_samples_gt_aware(
                model, query_root, lbl_dir, target_classes,
                conf_hard=args.conf_hard, match_iou=match_iou,
                query_dataset=query_dataset,
            )
        else:
            print(f"  WARNING: No label dir found for {query_root} — skipping GT-aware mining.")
            hard_imgs, hardness, img_to_ds = [], {}, {}

        all_hard_imgs.extend(hard_imgs)
        all_hardness.update(hardness)
        all_img_to_dataset.update(img_to_ds)

    if not all_hard_imgs:
        print("  No hard samples found. Try lowering --conf-hard or --match-iou.")
        hook.remove()
        return

    print(f"  Total hard samples across all query dirs: {len(all_hard_imgs)}")
    print("  Extracting hard sample embeddings...")
    hook.clear()  # clear any stale feats from predict calls during mining
    hard_embs, all_hard_imgs = extract_embeddings_multi(
        model, hook, embedding_layers, all_hard_imgs,
        batch_size=args.batch_size, device=device, has_rtdetr=has_rtdetr,
    )
    hook.remove()

    # Compute total embedding dim
    embedding_dim = hard_embs.shape[1] if hard_embs.ndim == 2 else 0

    # Retrieval with provenance
    print(f"\n[4/4] Retrieving from pool (sim >= {args.similarity_threshold}, top-{args.top_k})...")
    selected, ret_stats = retrieve_from_pool_with_provenance(
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
            "query_image", "query_dataset", "similarity", "hardness_score", "rank", "selected_reason",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # retrieval_stats.json
    source_split_counts = {"bdd_train": len(selected)}
    stats = {
        "candidate_pool_size": pool_count_after_filter,
        "candidate_pool_before_class_filter": pool_count_before_filter,
        "candidate_pool_after_class_filter": pool_count_after_filter,
        "query_roots": [str(q) for q in args.query_roots],
        "query_split": "train",
        "query_count": query_count,
        "hard_query_count": len(all_hard_imgs),
        "requested_top_k": args.top_k,
        "selected_unique": len(selected),
        "similarity_threshold": args.similarity_threshold,
        "seed": args.seed,
        "candidate_hits_above_threshold": ret_stats["candidate_hits_above_threshold"],
        "unique_candidates_before_top_k": ret_stats["unique_candidates_before_top_k"],
        "duplicate_candidate_hits_removed": ret_stats["duplicate_candidate_hits_removed"],
        # Legacy key kept for existing test compatibility
        "duplicate_candidates_removed": ret_stats["duplicate_candidate_hits_removed"],
        "overlap_with_used_bdd": overlap_count,
        "source_split_counts": source_split_counts,
        "embedding_layers": embedding_layers,
        "embedding_dim": embedding_dim,
    }
    (out_root / "retrieval_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. Retrieved data → {out_root}")
    print(f"  selected_unique: {len(selected)}")
    print(f"  embedding_layers: {embedding_layers}  dim: {embedding_dim}")
    print(f"  source_split: all bdd_train")


def main_legacy(args: argparse.Namespace) -> None:
    """Legacy main — original CLI interface."""
    target_classes = set(args.target_classes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Active Retrieval")
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

    exclude_stems: "set | None" = None
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
    print("\n  Similarity score stats:")
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
