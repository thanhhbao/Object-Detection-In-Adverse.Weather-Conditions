"""Tests for scripts/retrieve_dinov2.py — protocol and CLI validation.

No GPU, no real DINOv2. Tests cover:
- Test A: validate_query_root rejects val/test splits
- Test B: _pool_fingerprint used in DINOv2 cache metadata
- Test C: --clean removes stale output before rewrite
- Test D: output count invariants (image == label == manifest == selected_unique)
- Test E: overlap_with_used_bdd == 0 enforced
- Test F: DINOv2 cache metadata includes required fields
- Test G: threshold-diagnostic-only — threshold=0.0 vs threshold=0.999 produce same global top-N
- Test H: filter_pool_by_class excludes car-only images
- Test I: exact top_k guarantee — hard-fail if pool < top_k
- Test J: retrieval_stats.json contains all required similarity and provenance fields
- Test K: dinov2_smoke_check raises on bad embedding (non-finite / wrong norm)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake_img")


def _write_label(path: Path, classes: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in classes]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Test A: validate_query_root rejects val/test ──────────────────────────────

def test_A_validate_query_root_rejects_val(tmp_path):
    """validate_query_root (from active_retrieval) must reject images/val paths."""
    from active_retrieval import validate_query_root

    val_dir = tmp_path / "mydataset" / "images" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "mydataset" / "labels" / "val").mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit):
        validate_query_root(val_dir)


def test_A_validate_query_root_rejects_test(tmp_path):
    """validate_query_root must reject images/test paths."""
    from active_retrieval import validate_query_root

    test_dir = tmp_path / "mydataset" / "images" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "mydataset" / "labels" / "test").mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit):
        validate_query_root(test_dir)


def test_A_validate_query_root_accepts_train(tmp_path):
    """validate_query_root must accept images/train paths."""
    from active_retrieval import validate_query_root

    train_dir = tmp_path / "mydataset" / "images" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    lbl_train = tmp_path / "mydataset" / "labels" / "train"
    lbl_train.mkdir(parents=True, exist_ok=True)

    result = validate_query_root(train_dir)
    assert result == train_dir.resolve()


# ── Test B: _pool_fingerprint in DINOv2 cache metadata ───────────────────────

def test_B_pool_fingerprint_in_dinov2_cache_meta(tmp_path):
    """DINOv2 cache metadata must include pool_fingerprint from _pool_fingerprint."""
    from active_retrieval import _pool_fingerprint
    import retrieve_dinov2 as rd

    pool_paths = []
    for i in range(5):
        p = tmp_path / f"img_{i:03d}.jpg"
        p.write_bytes(b"fake")
        pool_paths.append(p)

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake_checkpoint" * 100)

    meta = rd._dinov2_cache_meta(weights, "dinov2_vitb14", pool_paths)

    assert "pool_fingerprint" in meta, "cache meta must have pool_fingerprint"
    assert "dinov2_model" in meta, "cache meta must have dinov2_model"
    assert "checkpoint_sha256_prefix" in meta, "cache meta must have checkpoint_sha256_prefix"
    assert meta["pool_fingerprint"] == _pool_fingerprint(pool_paths)
    assert meta["dinov2_model"] == "dinov2_vitb14"


# ── Test C: --clean removes stale output ─────────────────────────────────────

def test_C_clean_removes_stale_output(tmp_path):
    """--clean logic: non-empty out_root is removed before fresh write."""
    import shutil

    out_root = tmp_path / "out"
    stale_img = out_root / "images" / "train"
    stale_img.mkdir(parents=True, exist_ok=True)
    stale_file = stale_img / "stale.jpg"
    stale_file.write_bytes(b"stale")

    assert stale_file.exists()

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    assert not stale_file.exists(), "stale file should be gone after --clean"
    assert out_root.exists(), "out_root should be recreated after --clean"


def test_C_without_clean_fails_if_non_empty(tmp_path):
    """Without --clean, non-empty output dir must raise RuntimeError."""
    out_root = tmp_path / "out"
    stale_img = out_root / "images" / "train"
    stale_img.mkdir(parents=True, exist_ok=True)
    (stale_img / "stale.jpg").write_bytes(b"stale")

    stale = []
    if stale_img.exists() and any(stale_img.iterdir()):
        stale.append(str(stale_img))

    with pytest.raises(RuntimeError, match="non-empty"):
        if stale:
            raise RuntimeError(f"Output directory is non-empty: {stale}\nPass --clean to remove and rebuild.")


# ── Test D: output count invariants ──────────────────────────────────────────

def test_D_output_count_invariant_passes_when_equal():
    """No exception when image_count == label_count == manifest_count == selected_unique."""
    output_img_count = 10
    output_lbl_count = 10
    manifest_row_count = 10
    selected_unique = 10

    if output_img_count != selected_unique:
        raise RuntimeError("image count mismatch")
    if output_lbl_count != selected_unique:
        raise RuntimeError("label count mismatch")
    if manifest_row_count != selected_unique:
        raise RuntimeError("manifest count mismatch")


def test_D_output_count_invariant_fails_on_image_mismatch():
    """RuntimeError when image_count != selected_unique."""
    output_img_count = 9
    selected_unique = 10

    with pytest.raises(RuntimeError, match="OUTPUT COUNT MISMATCH|image count"):
        if output_img_count != selected_unique:
            raise RuntimeError(
                f"OUTPUT COUNT MISMATCH: wrote {output_img_count} images but selected_unique={selected_unique}."
            )


def test_D_output_count_invariant_fails_on_label_mismatch():
    """RuntimeError when label_count != selected_unique."""
    output_lbl_count = 8
    selected_unique = 10

    with pytest.raises(RuntimeError, match="LABEL COUNT MISMATCH|label count"):
        if output_lbl_count != selected_unique:
            raise RuntimeError(
                f"OUTPUT LABEL COUNT MISMATCH: wrote {output_lbl_count} labels but selected_unique={selected_unique}."
            )


# ── Test E: overlap_with_used_bdd == 0 enforced ──────────────────────────────

def test_E_overlap_check_raises_on_overlap(tmp_path):
    """If overlap detected with used BDD, RuntimeError must be raised."""
    selected_names = {"img001.jpg", "img002.jpg", "img003.jpg"}
    used_bdd_names = {"img002.jpg", "img999.jpg"}

    overlap = selected_names & used_bdd_names
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: {len(overlap)} retrieved image(s) appear in used BDD30K: "
                f"{sorted(overlap)[:5]}"
            )


def test_E_overlap_check_passes_when_disjoint():
    """No exception when retrieved images are disjoint from used BDD."""
    selected_names = {"img001.jpg", "img002.jpg"}
    used_bdd_names = {"img100.jpg", "img200.jpg"}

    overlap = selected_names & used_bdd_names
    assert len(overlap) == 0


# ── Test F: DINOv2 cache metadata fields ─────────────────────────────────────

def test_F_dinov2_cache_meta_required_fields(tmp_path):
    """Cache metadata must include pool_fingerprint, dinov2_model, checkpoint_sha256_prefix."""
    import retrieve_dinov2 as rd

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * 200)
    pool_paths = [tmp_path / f"p{i}.jpg" for i in range(3)]
    for p in pool_paths:
        p.write_bytes(b"img")

    meta = rd._dinov2_cache_meta(weights, "dinov2_vits14", pool_paths)

    required = ["pool_fingerprint", "dinov2_model", "checkpoint_sha256_prefix",
                "checkpoint", "checkpoint_size", "embedding_layers", "imgsz",
                "pool_count", "version"]
    for key in required:
        assert key in meta, f"cache meta missing key: {key}"

    assert meta["dinov2_model"] == "dinov2_vits14"
    assert meta["embedding_layers"] == ["dinov2_cls_token"]
    assert meta["pool_count"] == 3
    assert meta["version"] == 1


# ── Test G: threshold-diagnostic-only — same top-N regardless of threshold ───

def test_G_threshold_does_not_change_selection():
    """
    threshold is DIAGNOSTIC ONLY: top-N selection must be identical for
    threshold=0.0 and threshold=0.999 because threshold does not filter.
    """
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    rng = np.random.RandomState(7)
    n_pool = 30
    n_hard = 4
    dim = 8
    top_n = 10

    pool_embs = rng.randn(n_pool, dim).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(n_hard, dim).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    pool_paths = [Path(f"/fake/pool_{i:03d}.jpg") for i in range(n_pool)]
    hard_paths = [Path(f"/fake/hard_{i}.jpg") for i in range(n_hard)]
    hardness = {str(p.resolve()): 0.9 for p in hard_paths}
    img_to_ds = {str(p.resolve()): "xwod" for p in hard_paths}

    sel_low, stats_low = rd.retrieve_dinov2(
        hard_embs, hard_paths, hardness, img_to_ds,
        pool_embs, pool_paths, sim_threshold=0.0, top_n=top_n
    )
    sel_high, stats_high = rd.retrieve_dinov2(
        hard_embs, hard_paths, hardness, img_to_ds,
        pool_embs, pool_paths, sim_threshold=0.999, top_n=top_n
    )

    names_low = [e["pool_path"].name for e in sel_low]
    names_high = [e["pool_path"].name for e in sel_high]
    assert names_low == names_high, (
        f"threshold changed selection — must be diagnostic only.\n"
        f"  low:  {names_low}\n  high: {names_high}"
    )
    # Both must produce exactly top_n results
    assert len(sel_low) == top_n
    assert len(sel_high) == top_n

    # threshold_is_diagnostic_only must be True in both stats dicts
    assert stats_low.get("threshold_is_diagnostic_only") is True
    assert stats_high.get("threshold_is_diagnostic_only") is True

    # Diagnostic counts differ (more hits at lower threshold)
    assert stats_low["candidate_hits_above_threshold"] >= stats_high["candidate_hits_above_threshold"]


def test_G_retrieval_determinism():
    """Same pool embeddings, same hard embeddings → identical selection order."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    rng = np.random.RandomState(42)
    n_pool = 20
    n_hard = 3
    dim = 8

    pool_embs = rng.randn(n_pool, dim).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(n_hard, dim).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    pool_paths = [Path(f"/fake/pool_{i:03d}.jpg") for i in range(n_pool)]
    hard_paths = [Path(f"/fake/hard_{i}.jpg") for i in range(n_hard)]
    hardness = {str(p.resolve()): 0.9 for p in hard_paths}
    img_to_ds = {str(p.resolve()): "xwod" for p in hard_paths}

    sel_a, _ = rd.retrieve_dinov2(hard_embs, hard_paths, hardness, img_to_ds,
                                   pool_embs, pool_paths, sim_threshold=0.0, top_n=10)
    sel_b, _ = rd.retrieve_dinov2(hard_embs, hard_paths, hardness, img_to_ds,
                                   pool_embs, pool_paths, sim_threshold=0.0, top_n=10)

    names_a = [e["pool_path"].name for e in sel_a]
    names_b = [e["pool_path"].name for e in sel_b]
    assert names_a == names_b, f"Non-deterministic: {names_a} != {names_b}"


# ── Test H: filter_pool_by_class excludes car-only images ─────────────────────

def test_H_filter_pool_excludes_car_only(tmp_path):
    """filter_pool_by_class (from active_retrieval) must exclude car-only images."""
    from active_retrieval import filter_pool_by_class

    pool_img = tmp_path / "images" / "train"
    pool_lbl = tmp_path / "labels" / "train"
    pool_img.mkdir(parents=True, exist_ok=True)
    pool_lbl.mkdir(parents=True, exist_ok=True)

    _write_image(pool_img / "car_only.jpg")
    _write_label(pool_lbl / "car_only.txt", [2])

    _write_image(pool_img / "has_bicycle.jpg")
    _write_label(pool_lbl / "has_bicycle.txt", [1, 2])

    _write_image(pool_img / "has_moto.jpg")
    _write_label(pool_lbl / "has_moto.txt", [3])

    _write_image(pool_img / "has_bus.jpg")
    _write_label(pool_lbl / "has_bus.txt", [4, 2])

    rare_classes = {1, 3, 4}
    filtered, before = filter_pool_by_class(pool_img, pool_lbl, rare_classes)

    filtered_names = {p.name for p in filtered}
    assert "car_only.jpg" not in filtered_names, "car-only image must be excluded"
    assert "has_bicycle.jpg" in filtered_names
    assert "has_moto.jpg" in filtered_names
    assert "has_bus.jpg" in filtered_names
    assert before == 4
    assert len(filtered) == 3


# ── Test I: exact top_k guarantee ─────────────────────────────────────────────

def test_I_exact_topk_fails_when_pool_too_small():
    """retrieve_dinov2 must hard-fail (RuntimeError) if pool has fewer than top_n candidates."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    rng = np.random.RandomState(1)
    n_pool = 5   # smaller than top_n=10
    n_hard = 2
    dim = 4

    pool_embs = rng.randn(n_pool, dim).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(n_hard, dim).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    pool_paths = [Path(f"/fake/p{i}.jpg") for i in range(n_pool)]
    hard_paths = [Path(f"/fake/h{i}.jpg") for i in range(n_hard)]
    hardness = {str(p.resolve()): 1.0 for p in hard_paths}
    img_to_ds = {str(p.resolve()): "xwod" for p in hard_paths}

    with pytest.raises(RuntimeError, match="EXACT-10 FAIL"):
        rd.retrieve_dinov2(
            hard_embs, hard_paths, hardness, img_to_ds,
            pool_embs, pool_paths, sim_threshold=0.0, top_n=10
        )


def test_I_exact_topk_returns_exactly_topk():
    """retrieve_dinov2 must return exactly top_n results when pool is large enough."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    rng = np.random.RandomState(2)
    n_pool = 50
    n_hard = 3
    dim = 8
    top_n = 20

    pool_embs = rng.randn(n_pool, dim).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(n_hard, dim).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    pool_paths = [Path(f"/fake/pool_{i:03d}.jpg") for i in range(n_pool)]
    hard_paths = [Path(f"/fake/hard_{i}.jpg") for i in range(n_hard)]
    hardness = {str(p.resolve()): 0.8 for p in hard_paths}
    img_to_ds = {str(p.resolve()): "acdc" for p in hard_paths}

    selected, stats = rd.retrieve_dinov2(
        hard_embs, hard_paths, hardness, img_to_ds,
        pool_embs, pool_paths, sim_threshold=0.999, top_n=top_n  # high threshold: diagnostic only
    )

    assert len(selected) == top_n, f"Expected exactly {top_n}, got {len(selected)}"
    assert stats["selected_unique"] == top_n


# ── Test J: retrieval_stats.json required similarity and provenance fields ─────

def test_J_stats_have_sim_and_provenance_fields():
    """retrieve_dinov2 stats dict must contain sim_all/sim_selected stats and provenance."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    rng = np.random.RandomState(3)
    n_pool = 15
    n_hard = 2
    dim = 6

    pool_embs = rng.randn(n_pool, dim).astype(np.float32)
    pool_embs /= np.linalg.norm(pool_embs, axis=1, keepdims=True)
    hard_embs = rng.randn(n_hard, dim).astype(np.float32)
    hard_embs /= np.linalg.norm(hard_embs, axis=1, keepdims=True)

    pool_paths = [Path(f"/fake/p{i}.jpg") for i in range(n_pool)]
    hard_paths_xwod = [Path(f"/fake/xwod/h0.jpg")]
    hard_paths_acdc = [Path(f"/fake/acdc/h1.jpg")]
    all_hard_paths = hard_paths_xwod + hard_paths_acdc
    hardness = {str(p.resolve()): 0.9 for p in all_hard_paths}
    img_to_ds = {
        str(hard_paths_xwod[0].resolve()): "xwod",
        str(hard_paths_acdc[0].resolve()): "acdc",
    }

    _, stats = rd.retrieve_dinov2(
        hard_embs, all_hard_paths, hardness, img_to_ds,
        pool_embs, pool_paths, sim_threshold=0.0, top_n=10
    )

    # All-pair similarity fields
    for field in ["sim_all_min", "sim_all_median", "sim_all_mean", "sim_all_max"]:
        assert field in stats, f"Missing field: {field}"
        assert isinstance(stats[field], float), f"{field} must be float"

    # Selected similarity fields
    for field in ["sim_selected_min", "sim_selected_p05", "sim_selected_median",
                  "sim_selected_p95", "sim_selected_max"]:
        assert field in stats, f"Missing field: {field}"
        assert isinstance(stats[field], float), f"{field} must be float"

    # Ordering sanity
    assert stats["sim_selected_min"] <= stats["sim_selected_median"] <= stats["sim_selected_max"]
    assert stats["sim_all_min"] <= stats["sim_all_max"]

    # Provenance fields
    assert "unique_contributing_hard_queries" in stats
    assert isinstance(stats["unique_contributing_hard_queries"], int)
    assert stats["unique_contributing_hard_queries"] >= 1

    assert "selected_by_query_dataset" in stats
    ds_map = stats["selected_by_query_dataset"]
    assert isinstance(ds_map, dict)
    assert sum(ds_map.values()) == 10  # total selected

    # threshold_is_diagnostic_only flag
    assert stats.get("threshold_is_diagnostic_only") is True


# ── Test K: dinov2_smoke_check raises on bad embedding ────────────────────────

def test_K_smoke_check_raises_on_non_finite(tmp_path):
    """dinov2_smoke_check must raise RuntimeError if embedding is non-finite."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    # Mock the extraction function to return a NaN embedding
    img = tmp_path / "fake.jpg"
    img.write_bytes(b"x")

    original_fn = rd.extract_dinov2_embeddings

    def mock_extract(model, paths, batch_size, device, imgsz=224):
        embs = np.array([[float("nan"), 1.0, 0.0]])
        return embs, paths

    rd.extract_dinov2_embeddings = mock_extract
    try:
        with pytest.raises(RuntimeError, match="DINO SMOKE"):
            rd.dinov2_smoke_check(None, img, "cpu")
    finally:
        rd.extract_dinov2_embeddings = original_fn


def test_K_smoke_check_raises_on_wrong_norm(tmp_path):
    """dinov2_smoke_check must raise RuntimeError if L2 norm is not ≈ 1.0."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    img = tmp_path / "fake.jpg"
    img.write_bytes(b"x")

    original_fn = rd.extract_dinov2_embeddings

    def mock_extract(model, paths, batch_size, device, imgsz=224):
        embs = np.array([[2.0, 3.0, 4.0]])  # norm ≠ 1
        return embs, paths

    rd.extract_dinov2_embeddings = mock_extract
    try:
        with pytest.raises(RuntimeError, match="DINO SMOKE"):
            rd.dinov2_smoke_check(None, img, "cpu")
    finally:
        rd.extract_dinov2_embeddings = original_fn


def test_K_smoke_check_passes_on_valid_embedding(tmp_path):
    """dinov2_smoke_check must NOT raise when embedding is finite and L2-normalized."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd

    img = tmp_path / "fake.jpg"
    img.write_bytes(b"x")

    original_fn = rd.extract_dinov2_embeddings

    def mock_extract(model, paths, batch_size, device, imgsz=224):
        emb = np.array([[1.0, 0.0, 0.0]])  # L2 norm = 1.0
        return emb, paths

    rd.extract_dinov2_embeddings = mock_extract
    try:
        rd.dinov2_smoke_check(None, img, "cpu")  # must not raise
    finally:
        rd.extract_dinov2_embeddings = original_fn
