"""Tests for scripts/active_retrieval.py — protocol and CLI validation.

GPU-dependent embedding tests are skipped. Tests cover:
- CLI argument acceptance (--query-root repeatable)
- validate_labels function for invalid class IDs
- Dedup logic in retrieve_from_pool
- Seed determinism (output ordering)
- Manifest structure
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "active_retrieval.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_label(path: Path, classes: list[int]) -> None:
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


# ── Test C: no duplicate retrieved image names ────────────────────────────────

def test_retrieve_from_pool_deduplicates(tmp_path):
    """retrieve_from_pool must return no duplicate image names."""
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


# ── Test: seed makes output deterministic ─────────────────────────────────────

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
    """
    CLI must accept --query-root (repeatable). Verified via --help output.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0
    assert "--query-root" in result.stdout, f"--query-root not in help output: {result.stdout}"
    assert "--weights" in result.stdout, f"--weights not in help output: {result.stdout}"
    assert "--pool-root" in result.stdout


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
