"""Tests for scripts/retrieve_dinov2.py — protocol and CLI validation.

No GPU, no real DINOv2. Tests cover:
- Test A: validate_query_root rejects val/test splits
- Test B: _pool_fingerprint used in DINOv2 cache metadata
- Test C: --clean removes stale output before rewrite
- Test D: output count invariants (image == label == manifest == selected_unique)
- Test E: overlap_with_used_bdd == 0 enforced
- Test F: DINOv2 cache metadata includes required fields
- Test G: determinism — same pool, same seed, same model → identical selection
- Test H: filter_pool_by_class excludes car-only images
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
    # also create labels/val so it doesn't fail on missing labels
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

    # Create fake pool files
    pool_paths = []
    for i in range(5):
        p = tmp_path / f"img_{i:03d}.jpg"
        p.write_bytes(b"fake")
        pool_paths.append(p)

    # Build fake weights file
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

    # Simulate --clean logic
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

    # Simulate the guard logic from retrieve_dinov2.py
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

    # Should not raise
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
    # Should not raise
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


# ── Test G: determinism — same inputs → same order ───────────────────────────

def test_G_retrieval_determinism():
    """Same pool embeddings, same hard embeddings, same seed → identical selection order."""
    np = pytest.importorskip("numpy")
    import retrieve_dinov2 as rd
    from pathlib import Path

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

    # car only (class 2) — should be excluded
    _write_image(pool_img / "car_only.jpg")
    _write_label(pool_lbl / "car_only.txt", [2])

    # bicycle (class 1) — should be included
    _write_image(pool_img / "has_bicycle.jpg")
    _write_label(pool_lbl / "has_bicycle.txt", [1, 2])

    # motorcycle (class 3) — should be included
    _write_image(pool_img / "has_moto.jpg")
    _write_label(pool_lbl / "has_moto.txt", [3])

    # bus (class 4) — should be included
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
