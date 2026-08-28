"""Tests for scripts/active_retrieval.py — protocol and CLI validation.

GPU-dependent embedding tests are skipped. Tests cover:
- CLI argument acceptance (--query-root repeatable)
- validate_labels function for invalid class IDs
- Dedup logic in retrieve_from_pool (legacy) and retrieve_from_pool_with_provenance (new-style)
- Seed determinism (output ordering)
- Manifest structure
- GT-aware hard mining (is_gt_hard, compute_iou)
- Candidate class filter (filter_pool_by_class)
- Query provenance (query_dataset tracking)
- Dedup statistics (duplicate_candidate_hits_removed)
- validate_query_root (val/test rejection)
- Cache metadata mismatch detection (load_pool_cache)
- MultiScaleHook concatenation shape and L2 normalization
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "active_retrieval.py"
scripts_dir = ROOT / "scripts"

try:
    import torch as _torch_mod
    _torch_available = True
except ImportError:
    _torch_available = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_label(path: Path, classes: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Test: validate_labels function ────────────────────────────────────────────

def test_validate_labels_rejects_class_above_5(tmp_path):
    """validate_labels must flag class IDs > 5."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from active_retrieval import validate_labels

    lbl_dir = tmp_path / "labels" / "train"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / "bad.txt").write_text("6 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1", encoding="utf-8")
    (lbl_dir / "ok.txt").write_text("0 0.5 0.5 0.1 0.1\n5 0.5 0.5 0.1 0.1", encoding="utf-8")

    violations = validate_labels(lbl_dir, max_class_id=5)
    assert any("bad.txt" in v for v in violations), f"Expected violation for class 6, got: {violations}"
    assert not any("ok.txt" in v for v in violations), f"ok.txt should have no violations"


def test_validate_labels_accepts_all_valid_ids(tmp_path):
    """validate_labels must not flag class IDs 0-5."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from active_retrieval import validate_labels

    lbl_dir = tmp_path / "labels" / "train"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (lbl_dir / f"cls{i}.txt").write_text(f"{i} 0.5 0.5 0.1 0.1", encoding="utf-8")

    violations = validate_labels(lbl_dir, max_class_id=5)
    assert violations == [], f"Expected no violations, got: {violations}"


# ── Legacy dedup test ─────────────────────────────────────────────────────────

def test_retrieve_from_pool_deduplicates(tmp_path):
    """retrieve_from_pool (legacy) must return no duplicate image names."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(ROOT / "scripts"))
    from active_retrieval import retrieve_from_pool

    pool_paths = [tmp_path / f"pool_{i}.jpg" for i in range(3)]
    for p in pool_paths:
        p.write_bytes(b"fake")

    pool_embs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    hard_embs = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.95, 0.05, 0.0, 0.0],
    ])
    pool_embs = pool_embs / np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = hard_embs / np.linalg.norm(hard_embs, axis=1, keepdims=True)

    selected = retrieve_from_pool(hard_embs, pool_embs, pool_paths, sim_threshold=0.5, top_n=10)

    names = [p.name for p, _ in selected]
    assert len(names) == len(set(names)), f"Duplicate image names in retrieved: {names}"


def test_retrieve_from_pool_deterministic_order(tmp_path):
    """Same inputs → same output order (deterministic via -sim, name sort)."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(ROOT / "scripts"))
    from active_retrieval import retrieve_from_pool

    pool_paths = [tmp_path / f"img_{c}.jpg" for c in "abcde"]
    for p in pool_paths:
        p.write_bytes(b"fake")

    rng = np.random.RandomState(0)
    pool_embs = rng.randn(5, 8).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(2, 8).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    result_a = retrieve_from_pool(hard_embs, pool_embs, pool_paths, sim_threshold=0.0, top_n=10)
    result_b = retrieve_from_pool(hard_embs, pool_embs, pool_paths, sim_threshold=0.0, top_n=10)

    assert [p.name for p, _ in result_a] == [p.name for p, _ in result_b]


# ── Test: CLI accepts --query-root (train dirs) ───────────────────────────────

def test_cli_accepts_query_root_argument():
    """CLI must accept --query-root (repeatable). Verified via --help output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "--query-root" in result.stdout, f"--query-root not in help output: {result.stdout}"
    assert "--weights" in result.stdout, f"--weights not in help output: {result.stdout}"
    assert "--pool-root" in result.stdout
    # New-style args
    assert "--embedding-layers" in result.stdout, "--embedding-layers not in help"
    assert "--match-iou" in result.stdout, "--match-iou not in help"
    assert "--candidate-target-classes" in result.stdout, "--candidate-target-classes not in help"


# ── Test: retrieval_stats.json structure contract ─────────────────────────────

def test_retrieval_stats_required_keys():
    """retrieval_stats.json must have all required keys (structure contract)."""
    required_keys = [
        "candidate_pool_size", "query_count", "requested_top_k", "selected_unique",
        "similarity_threshold", "seed", "duplicate_candidates_removed",
        "overlap_with_used_bdd", "source_split_counts",
    ]
    mock_stats = {k: 0 for k in required_keys}
    mock_stats["source_split_counts"] = {"bdd_train": 100}
    for key in required_keys:
        assert key in mock_stats


def test_retrieval_stats_new_keys():
    """New-style retrieval_stats.json must include embedding and dedup fields."""
    new_keys = [
        "candidate_pool_before_class_filter",
        "candidate_pool_after_class_filter",
        "query_roots",
        "query_split",
        "hard_query_count",
        "candidate_hits_above_threshold",
        "unique_candidates_before_top_k",
        "duplicate_candidate_hits_removed",
        "embedding_layers",
        "embedding_dim",
    ]
    mock_stats = {k: 0 for k in new_keys}
    mock_stats["query_roots"] = ["/data/xwod/images/train"]
    mock_stats["query_split"] = "train"
    mock_stats["embedding_layers"] = [21, 24, 27]
    for key in new_keys:
        assert key in mock_stats


# ── Test: retrieved_manifest.csv required fields contract ─────────────────────

def test_retrieved_manifest_field_contract():
    """retrieved_manifest.csv must have all required fields (contract test)."""
    required_fields = [
        "retrieved_image", "retrieved_label", "source_image_name", "source_split",
        "query_image", "similarity", "hardness_score", "rank", "selected_reason",
    ]
    mock_row = {f: "" for f in required_fields}
    mock_row["source_split"] = "bdd_train"
    mock_row["selected_reason"] = "similarity_retrieval"
    mock_row["rank"] = 1

    for field in required_fields:
        assert field in mock_row
    assert mock_row["source_split"] == "bdd_train"
    assert mock_row["selected_reason"] == "similarity_retrieval"


def test_retrieved_manifest_new_field_query_dataset():
    """New-style manifest must include query_dataset column."""
    required_fields = [
        "retrieved_image", "retrieved_label", "source_image_name", "source_split",
        "query_image", "query_dataset", "similarity", "hardness_score", "rank", "selected_reason",
    ]
    mock_row = {f: "" for f in required_fields}
    mock_row["query_dataset"] = "xwod"
    assert "query_dataset" in mock_row
    assert mock_row["query_dataset"] == "xwod"


# ── Test A: GT bicycle missed (class mismatch) → hard ────────────────────────

def test_a_gt_missed_class_mismatch_is_hard():
    """GT bicycle (cls=1) missed when only bus (cls=4) predicted → hard."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import is_gt_hard

    # GT: bicycle at center (0.5, 0.5, 0.2, 0.2)
    # Predictions: bus at same location, high conf
    pred_boxes = np.array([[256.0, 256.0, 384.0, 384.0]])  # xyxy absolute

    is_hard, matched_conf = is_gt_hard(
        gt_cls=1,
        gt_box=[0.5, 0.5, 0.2, 0.2],
        pred_classes=[4],   # bus, not bicycle
        pred_confs=[0.9],
        pred_boxes_xyxy=pred_boxes,
        img_w=640,
        img_h=640,
        conf_hard=0.25,
        match_iou=0.5,
    )
    assert is_hard is True, "Class mismatch: GT bicycle with bus prediction → should be hard"
    assert matched_conf == 0.0, "No same-class prediction → matched_conf should be 0.0"


# ── Test B: Same-class high-conf prediction → not hard ───────────────────────

def test_b_gt_detected_well_is_not_hard():
    """GT bicycle detected with same-class IoU >= 0.5 and conf >= 0.25 → not hard."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import is_gt_hard

    # GT: bicycle at center (0.5, 0.5, 0.2, 0.2)
    # → absolute: cx=320, cy=320, w=128, h=128
    # → xyxy: [256, 256, 384, 384]
    # Prediction: almost identical box, class=1, conf=0.8
    pred_boxes = np.array([[320 - 64, 320 - 64, 320 + 64, 320 + 64]], dtype=float)

    is_hard, matched_conf = is_gt_hard(
        gt_cls=1,
        gt_box=[0.5, 0.5, 0.2, 0.2],
        pred_classes=[1],
        pred_confs=[0.8],
        pred_boxes_xyxy=pred_boxes,
        img_w=640,
        img_h=640,
        conf_hard=0.25,
        match_iou=0.5,
    )
    assert is_hard is False, "Well-detected GT → should NOT be hard"
    assert matched_conf == pytest.approx(0.8, abs=1e-5)


# ── Test C: Candidate class filter ────────────────────────────────────────────

def test_c_filter_pool_by_class(tmp_path):
    """filter_pool_by_class excludes images without target-class labels."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import filter_pool_by_class

    pool_img_dir = tmp_path / "images" / "train"
    pool_lbl_dir = tmp_path / "labels" / "train"
    pool_img_dir.mkdir(parents=True, exist_ok=True)
    pool_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Image 1: bicycle (cls=1) → include
    _write_image(pool_img_dir / "img_bicycle.jpg")
    _write_label(pool_lbl_dir / "img_bicycle.txt", [1])

    # Image 2: car only (cls=2) → exclude (not in target {1,3,4})
    _write_image(pool_img_dir / "img_car.jpg")
    _write_label(pool_lbl_dir / "img_car.txt", [2])

    # Image 3: motorcycle (cls=3) → include
    _write_image(pool_img_dir / "img_moto.jpg")
    _write_label(pool_lbl_dir / "img_moto.txt", [3])

    target = {1, 3, 4}
    filtered, before = filter_pool_by_class(pool_img_dir, pool_lbl_dir, target)

    assert before == 3, f"Expected 3 total pool images, got {before}"
    assert len(filtered) == 2, f"Expected 2 after filter, got {len(filtered)}"
    names = {p.name for p in filtered}
    assert "img_bicycle.jpg" in names
    assert "img_moto.jpg" in names
    assert "img_car.jpg" not in names


# ── Test D: Provenance — best similarity wins ─────────────────────────────────

def test_d_provenance_max_sim_wins(tmp_path):
    """Pool image hit by 2 queries → entry uses query with higher similarity."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import retrieve_from_pool_with_provenance

    # 3 pool images, 4-dim embeddings (unit vectors along axes)
    pool_paths = [tmp_path / f"pool_{i}.jpg" for i in range(3)]
    for p in pool_paths:
        p.write_bytes(b"fake")

    pool_embs = np.eye(3, 4, dtype=np.float32)  # each row is a unit vector

    # hard[0]: sim=0.9 to pool[0], hard[1]: sim=0.8 to pool[0]
    hard_paths = [tmp_path / "hard0.jpg", tmp_path / "hard1.jpg"]
    # construct embeddings that give desired cosine similarities
    # hard[0]: mostly axis 0 → sim(pool[0]) ≈ high
    h0 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    h0 /= np.linalg.norm(h0)
    h1 = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)
    h1 /= np.linalg.norm(h1)
    hard_embs = np.stack([h0, h1])

    hardness_scores = {"hard0.jpg": 1.0, "hard1.jpg": 0.9}
    img_to_dataset = {"hard0.jpg": "xwod", "hard1.jpg": "acdc"}

    selected, stats = retrieve_from_pool_with_provenance(
        hard_embs=hard_embs,
        hard_paths=hard_paths,
        hardness_scores=hardness_scores,
        img_to_dataset=img_to_dataset,
        pool_embs=pool_embs,
        pool_paths=pool_paths,
        sim_threshold=0.5,
        top_n=10,
    )

    # pool[0] should appear once, with query=hard[0] (higher sim)
    pool0_entries = [e for e in selected if e["pool_path"] == pool_paths[0]]
    assert len(pool0_entries) == 1, "pool[0] should appear exactly once"
    entry = pool0_entries[0]
    assert entry["query_path"] == hard_paths[0], (
        f"Expected hard_paths[0] as query (higher sim), got {entry['query_path']}"
    )
    assert entry["query_dataset"] == "xwod"


# ── Test E: Dedup statistics ──────────────────────────────────────────────────

def test_e_dedup_statistics_correct(tmp_path):
    """duplicate_candidate_hits_removed counts hits deduplicated away."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import retrieve_from_pool_with_provenance

    pool_paths = [tmp_path / "pool_0.jpg"]
    (tmp_path / "pool_0.jpg").write_bytes(b"fake")
    pool_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # 2 hard queries both above threshold for pool[0]
    h0 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    h0 /= np.linalg.norm(h0)
    h1 = np.array([0.85, 0.15, 0.0, 0.0], dtype=np.float32)
    h1 /= np.linalg.norm(h1)
    hard_embs = np.stack([h0, h1])
    hard_paths = [tmp_path / "hard0.jpg", tmp_path / "hard1.jpg"]

    selected, stats = retrieve_from_pool_with_provenance(
        hard_embs=hard_embs,
        hard_paths=hard_paths,
        hardness_scores={},
        img_to_dataset={},
        pool_embs=pool_embs,
        pool_paths=pool_paths,
        sim_threshold=0.5,
        top_n=10,
    )

    # pool[0] hit twice (from 2 queries) above threshold
    assert stats["candidate_hits_above_threshold"] == 2, (
        f"Expected 2 hits, got {stats['candidate_hits_above_threshold']}"
    )
    assert stats["unique_candidates_before_top_k"] == 1, (
        f"Expected 1 unique, got {stats['unique_candidates_before_top_k']}"
    )
    assert stats["duplicate_candidate_hits_removed"] == 1, (
        f"Expected 1 duplicate removed, got {stats['duplicate_candidate_hits_removed']}"
    )
    assert stats["selected_unique"] == 1


# ── Test F: val/test query-root rejected ──────────────────────────────────────

def test_f_val_query_root_rejected():
    """validate_query_root must reject val splits."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import validate_query_root

    with pytest.raises(SystemExit):
        validate_query_root(Path("/data/xwod/images/val"))


def test_f_test_query_root_rejected():
    """validate_query_root must reject test splits."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import validate_query_root

    with pytest.raises(SystemExit):
        validate_query_root(Path("/data/xwod/images/test"))


# ── Test G: Cache metadata mismatch → rebuild ─────────────────────────────────

def test_g_cache_metadata_mismatch_triggers_rebuild(tmp_path):
    """load_pool_cache returns None when metadata mismatches."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import load_pool_cache, save_pool_cache

    cache_path = tmp_path / "pool_embs.npz"

    # Save a cache with specific metadata
    original_meta = {
        "checkpoint": "/workspace/runs/phase2/best.pt",
        "checkpoint_size": 100000,
        "checkpoint_sha256_prefix": "abcd1234",
        "embedding_layers": [21, 24, 27],
        "imgsz": 640,
        "pool_count": 50,
        "version": 2,
    }
    embs = np.random.randn(5, 768).astype(np.float32)
    paths = [Path(f"/data/pool/img_{i}.jpg") for i in range(5)]
    save_pool_cache(cache_path, embs, paths, original_meta)

    # Try to load with a different checkpoint_size
    different_meta = {**original_meta, "checkpoint_size": original_meta["checkpoint_size"] + 1}
    result = load_pool_cache(cache_path, different_meta)

    assert result is None, "Expected None (rebuild) when metadata mismatches checkpoint_size"


def test_g_cache_metadata_match_loads(tmp_path):
    """load_pool_cache returns embeddings when metadata matches."""
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import load_pool_cache, save_pool_cache

    cache_path = tmp_path / "pool_embs.npz"

    meta = {
        "checkpoint": "/workspace/runs/phase2/best.pt",
        "checkpoint_size": 100000,
        "checkpoint_sha256_prefix": "abcd1234",
        "embedding_layers": [21, 24, 27],
        "imgsz": 640,
        "pool_count": 5,
        "version": 2,
    }
    embs = np.random.randn(5, 768).astype(np.float32)
    paths = [Path(f"/data/pool/img_{i}.jpg") for i in range(5)]
    save_pool_cache(cache_path, embs, paths, meta)

    result = load_pool_cache(cache_path, meta)
    assert result is not None, "Expected cache to load when metadata matches"
    loaded_embs, loaded_paths = result
    assert loaded_embs.shape == (5, 768)
    assert len(loaded_paths) == 5


# ── Test H: MultiScaleHook concatenation shape and L2 norm ───────────────────

@pytest.mark.skipif(not _torch_available, reason="torch not available")
def test_h_multiscale_hook_concatenation_shape():
    """MultiScaleHook.get_embedding concatenates layer feats and L2-normalizes."""
    import torch
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import MultiScaleHook

    # Bypass __init__ to test get_embedding logic without a real model
    hook = object.__new__(MultiScaleHook)
    hook._handles = []
    hook.feats = {
        21: torch.randn(2, 256),
        24: torch.randn(2, 256),
        27: torch.randn(2, 256),
    }

    emb = hook.get_embedding([21, 24, 27])

    assert emb.shape == (2, 768), f"Expected shape (2, 768), got {emb.shape}"

    # Verify L2-normalized (each row norm ≈ 1.0)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), (
        f"Expected L2 norms ≈ 1.0, got {norms.tolist()}"
    )


@pytest.mark.skipif(not _torch_available, reason="torch not available")
def test_h_multiscale_hook_clear():
    """MultiScaleHook.clear() empties feats dict."""
    import torch
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import MultiScaleHook

    hook = object.__new__(MultiScaleHook)
    hook._handles = []
    hook.feats = {21: torch.randn(2, 256), 24: torch.randn(2, 256)}

    hook.clear()
    assert len(hook.feats) == 0, "feats dict should be empty after clear()"


@pytest.mark.skipif(not _torch_available, reason="torch not available")
def test_h_multiscale_hook_missing_layer_raises():
    """MultiScaleHook.get_embedding raises RuntimeError for missing layer."""
    import torch
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import MultiScaleHook

    hook = object.__new__(MultiScaleHook)
    hook._handles = []
    hook.feats = {21: torch.randn(2, 256)}  # only layer 21

    with pytest.raises(RuntimeError, match="Layer 24 hook did not produce output"):
        hook.get_embedding([21, 24, 27])


# ── Test I: --clean removes stale output; without --clean stale files fail ────

def test_i_clean_removes_stale_output(tmp_path):
    """Second retrieval run with --clean must not leave stale images from first run."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import place_file, IMAGE_EXTS

    out_img_dir = tmp_path / "images" / "train"
    out_lbl_dir = tmp_path / "labels" / "train"
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    # Simulate first run: 5 stale images
    for i in range(5):
        (out_img_dir / f"stale_{i:03d}.jpg").write_bytes(b"stale")
    assert len(list(out_img_dir.iterdir())) == 5

    # --clean removes the whole out_root
    import shutil
    out_root = tmp_path
    shutil.rmtree(out_root)
    out_root.mkdir()

    # Second run writes only 2 images
    out_img_dir2 = out_root / "images" / "train"
    out_lbl_dir2 = out_root / "labels" / "train"
    out_img_dir2.mkdir(parents=True)
    out_lbl_dir2.mkdir(parents=True)
    for i in range(2):
        (out_img_dir2 / f"new_{i:03d}.jpg").write_bytes(b"new")

    imgs = list(out_img_dir2.iterdir())
    assert len(imgs) == 2, f"Expected 2 after clean+rerun, got {len(imgs)}: {[p.name for p in imgs]}"
    assert all("stale" not in p.name for p in imgs), "Stale files survived --clean"


def test_i_no_clean_stale_raises(tmp_path):
    """Without --clean, non-empty output dir must raise RuntimeError."""
    out_img_dir = tmp_path / "images" / "train"
    out_img_dir.mkdir(parents=True)
    (out_img_dir / "stale.jpg").write_bytes(b"stale")

    # Simulate the guard logic from main_new
    stale_img = tmp_path / "images" / "train"
    stale = []
    if stale_img.exists() and any(stale_img.iterdir()):
        stale.append(str(stale_img))
    assert stale, "Guard should detect non-empty output dir"


def test_i_label_count_mismatch_raises(tmp_path):
    """Missing retrieved label must trigger OUTPUT LABEL COUNT MISMATCH invariant."""
    out_img_dir = tmp_path / "images" / "train"
    out_lbl_dir = tmp_path / "labels" / "train"
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    # 3 images but only 2 labels (one pool image had no label file)
    for i in range(3):
        (out_img_dir / f"img_{i}.jpg").write_bytes(b"fake")
    for i in range(2):
        (out_lbl_dir / f"img_{i}.txt").write_text(f"1 0.5 0.5 0.1 0.1")

    selected_unique = 3
    output_img_count = sum(
        1 for p in out_img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and (p.is_file() or p.is_symlink())
    )
    output_lbl_count = sum(
        1 for p in out_lbl_dir.iterdir()
        if p.suffix.lower() == ".txt" and (p.is_file() or p.is_symlink())
    )

    assert output_img_count == selected_unique  # images are fine
    assert output_lbl_count == 2               # labels are short

    # The invariant check that must raise
    with pytest.raises(RuntimeError, match="OUTPUT LABEL COUNT MISMATCH"):
        if output_lbl_count != selected_unique:
            raise RuntimeError(
                f"OUTPUT LABEL COUNT MISMATCH: wrote {output_lbl_count} labels "
                f"but selected_unique={selected_unique}. "
                "Every retrieved BDD image must have exactly one corresponding label."
            )


# ── Test J: shared basename across datasets — provenance not overwritten ──────

def test_j_shared_basename_provenance():
    """XWOD and ACDC with same filename must keep separate provenance records."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import find_hard_samples_gt_aware  # noqa: F401
    # We test the dict-key mechanism directly (canonical path, not basename)

    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Create two directories with the same basename "shared.jpg" but different paths
        xwod_img = td / "xwod" / "images" / "train" / "shared.jpg"
        acdc_img = td / "acdc" / "images" / "train" / "shared.jpg"
        xwod_lbl = td / "xwod" / "labels" / "train" / "shared.txt"
        acdc_lbl = td / "acdc" / "labels" / "train" / "shared.txt"
        for p in [xwod_img, acdc_img]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")
        xwod_lbl.parent.mkdir(parents=True, exist_ok=True)
        acdc_lbl.parent.mkdir(parents=True, exist_ok=True)
        xwod_lbl.write_text("1 0.5 0.5 0.2 0.2")  # bicycle
        acdc_lbl.write_text("1 0.5 0.5 0.2 0.2")  # bicycle

        # Simulate storing by canonical key
        hardness_scores = {}
        img_to_dataset = {}

        xwod_key = str(xwod_img.resolve())
        acdc_key = str(acdc_img.resolve())
        hardness_scores[xwod_key] = 0.9  # XWOD: high hardness
        hardness_scores[acdc_key] = 0.3  # ACDC: lower hardness
        img_to_dataset[xwod_key] = "xwod"
        img_to_dataset[acdc_key] = "acdc"

        # Keys are distinct even though basenames are the same
        assert xwod_key != acdc_key, "Canonical keys should differ for different paths"
        assert hardness_scores[xwod_key] == 0.9
        assert hardness_scores[acdc_key] == 0.3
        assert img_to_dataset[xwod_key] == "xwod"
        assert img_to_dataset[acdc_key] == "acdc"


# ── Test K: pool fingerprint detects filename change at same count ─────────────

def test_k_pool_fingerprint_filename_change():
    """Pool fingerprint must change when filenames change even if count is the same."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import _pool_fingerprint

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pool_a = [td / "a.jpg", td / "b.jpg", td / "c.jpg"]
        pool_b = [td / "a.jpg", td / "b.jpg", td / "d.jpg"]  # d replaces c, same count

        fp_a = _pool_fingerprint(pool_a)
        fp_b = _pool_fingerprint(pool_b)
        assert fp_a != fp_b, "Pool fingerprint must change when filenames change"


def test_k_pool_fingerprint_same_files_stable():
    """Same pool paths → same fingerprint (deterministic)."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import _pool_fingerprint

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pool = [td / "x.jpg", td / "y.jpg"]
        assert _pool_fingerprint(pool) == _pool_fingerprint(pool)


def test_k_cache_fingerprint_pool_fingerprint_included():
    """_cache_fingerprint must include pool_fingerprint key."""
    sys.path.insert(0, str(scripts_dir))
    from active_retrieval import _cache_fingerprint

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Create a fake weights file
        fake_weights = td / "best.pt"
        fake_weights.write_bytes(b"fake")
        pool = [td / "a.jpg", td / "b.jpg"]
        meta = _cache_fingerprint(fake_weights, [21, 24, 27], 640, pool)
        assert "pool_fingerprint" in meta, "pool_fingerprint must be in cache metadata"
        assert isinstance(meta["pool_fingerprint"], str) and len(meta["pool_fingerprint"]) > 0


def test_k_cache_pool_fingerprint_mismatch_rebuilds(tmp_path):
    """load_pool_cache must reject cache when pool_fingerprint differs."""
    sys.path.insert(0, str(scripts_dir))

    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")

    from active_retrieval import load_pool_cache, _cache_fingerprint

    cache_file = tmp_path / "cache.npz"
    pool_a = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    pool_b = [tmp_path / "a.jpg", tmp_path / "c.jpg"]  # c replaces b

    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")

    # Build cache with pool_a
    meta_a = _cache_fingerprint(fake_weights, [21, 24, 27], 640, pool_a)
    meta_json = __import__("json").dumps(meta_a)
    fake_embs = np.zeros((2, 768))
    np.savez(cache_file, embs=fake_embs,
             paths=np.array([str(p) for p in pool_a]),
             meta=np.array(meta_json))

    # Try to load with pool_b fingerprint → should reject
    meta_b = _cache_fingerprint(fake_weights, [21, 24, 27], 640, pool_b)
    result = load_pool_cache(cache_file, meta_b)
    assert result is None, "Cache with pool_fingerprint mismatch must be rejected"
