"""
tests/test_adversarial_f14_cache.py
===================================
Adversarial and Empirical Stress Test Suite for Milestone 3 (Feature 14):
Portable Parquet/JSON Cache Persistence, Integrity, and Concurrency.

Evaluates:
1. Zero-pickle enforcement (monkeypatch pickle.dump/dumps, verify .pkl absence).
2. Cache corruption and recovery:
   - Malformed, truncated, or missing JSON metadata.
   - Corrupted or truncated Parquet DataFrames (testing PyO3 Rust panic resilience).
   - Missing companion files (JSON without Parquet, or vice versa).
3. Atomic write durability and mid-write error unlinking (.tmp_* cleanup).
4. Concurrency safety:
   - Multi-worker concurrent saves with identical cache key (no FileNotFoundError or collision).
   - Concurrent reads during active writes (with <= 5s timeout guards).
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.config import PipelineConfig
from src.icp_engine import (
    DiscoveredICP,
    ICPEngineResult,
    StatisticalFeatureProof,
    _compute_engine_cache_key,
    load_sha256_cached_result,
    save_sha256_cached_result,
)


def _create_synthetic_result(client_sha256: str = "stress_test_sha256_client") -> ICPEngineResult:
    """Helper to generate a fully populated synthetic ICPEngineResult for stress tests."""
    proof = StatisticalFeatureProof(
        feature_type="prefecture_name",
        feature_value="ΑΤΤΙΚΗΣ",
        feature_label_el="Περιφέρεια: ΑΤΤΙΚΗΣ",
        icp_count=100,
        icp_share=0.85,
        universe_count=12000,
        universe_share=0.49,
        raw_lift=1.73,
        eb_lift=1.65,
        p_value_raw=1.2e-8,
        p_value_fdr=2.4e-7,
    )
    profile = DiscoveredICP(
        profile_id="ICP_1",
        cluster_idx=0,
        title_el="Δοκιμαστικό Προφίλ 1",
        ai_description_el="Περιγραφή δοκιμαστικού προφίλ.",
        client_count=120,
        client_share=0.40,
        turnover_total=3000000.0,
        turnover_share=0.55,
        turnover_median=25000.0,
        turnover_mean=30000.0,
        stability_score=0.95,
        confidence_score=0.92,
        centroid_vector=np.ones(64, dtype=np.float32) / 8.0,
        spatial_medoid_latlon_deg=(37.9838, 23.7275),
        significant_features=[proof],
        eb_log_lifts={("prefecture_name", "ΑΤΤΙΚΗΣ"): 0.5007},
    )
    return ICPEngineResult(
        optimal_k=1,
        global_bootstrap_ari=0.94,
        silhouette_by_k={1: 0.0, 2: 0.35},
        profiles=[profile],
        assignments_df=pl.DataFrame({
            "company_id": list(range(500)),
            "profile_id": ["ICP_1"] * 500,
            "score": [0.85] * 500,
        }),
        lookalikes_df=pl.DataFrame({
            "company_id": list(range(500, 1000)),
            "profile_id": ["ICP_1"] * 500,
            "rank": list(range(1, 501)),
            "score": [0.75] * 500,
            "αιτιολόγηση": ["Ταύτιση"] * 500,
        }),
        all_candidates_ranked=list(range(500, 1000)),
        client_sha256=client_sha256,
        loaded_from_sha256_cache=False,
    )


# ---------------------------------------------------------------------------
# 1. Zero-Pickle Enforcement
# ---------------------------------------------------------------------------
def test_zero_pickle_enforcement_under_monkeypatch(tmp_path: Path):
    """
    Empirically verify that pickle is never called during save or load.
    Monkeypatch pickle.dump, dumps, load, loads to fail fast with RuntimeError.
    """
    def forbidden(*args, **kwargs):
        raise RuntimeError("CRITICAL: pickle invocation is strictly forbidden!")

    with patch("pickle.dump", side_effect=forbidden), \
         patch("pickle.dumps", side_effect=forbidden), \
         patch("pickle.load", side_effect=forbidden), \
         patch("pickle.loads", side_effect=forbidden):

        cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
        res = _create_synthetic_result("zero_pickle_test_key")

        # Must succeed without triggering forbidden pickle calls
        saved_path = save_sha256_cached_result(res, cfg)
        assert saved_path.exists()
        assert saved_path.suffix == ".json"

        loaded = load_sha256_cached_result("zero_pickle_test_key", cfg)
        assert loaded is not None
        assert loaded.assignments_df.height == 500

    # Verify no .pkl files were generated
    pkl_files = list(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == 0, f"Found .pkl files: {pkl_files}"


# ---------------------------------------------------------------------------
# 2. Cache Corruption & Recovery
# ---------------------------------------------------------------------------
def test_corrupted_meta_json_clean_none(tmp_path: Path):
    """Verify that malformed JSON, truncated JSON, empty files, or missing schema keys cleanly return None."""
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("test_json_corruption")
    save_sha256_cached_result(res, cfg)

    key = _compute_engine_cache_key("test_json_corruption", cfg)
    meta_path = tmp_path / f"cached_{key}_meta.json"
    assert meta_path.exists()

    # 1. Truncated JSON
    meta_path.write_text('{"format_version": 1, "diagnostics": {', encoding="utf-8")
    assert load_sha256_cached_result("test_json_corruption", cfg) is None

    # 2. Empty JSON file
    meta_path.write_text("", encoding="utf-8")
    assert load_sha256_cached_result("test_json_corruption", cfg) is None

    # 3. Missing diagnostics / profiles schema keys
    meta_path.write_text(json.dumps({"format_version": 1, "client_sha256": "foo"}), encoding="utf-8")
    assert load_sha256_cached_result("test_json_corruption", cfg) is None

    # 4. Invalid JSON syntax
    meta_path.write_text("<<<NOT JSON>>>", encoding="utf-8")
    assert load_sha256_cached_result("test_json_corruption", cfg) is None


def test_missing_companion_files_clean_none(tmp_path: Path):
    """Verify that if any of the three cache files is missing, load_sha256_cached_result returns None."""
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("test_partial_cache")
    save_sha256_cached_result(res, cfg)

    key = _compute_engine_cache_key("test_partial_cache", cfg)
    meta_path = tmp_path / f"cached_{key}_meta.json"
    ass_path = tmp_path / f"cached_{key}_assignments.parquet"
    look_path = tmp_path / f"cached_{key}_lookalikes.parquet"

    # Delete assignments only
    ass_path.unlink()
    assert load_sha256_cached_result("test_partial_cache", cfg) is None

    # Re-save, then delete lookalikes only
    save_sha256_cached_result(res, cfg)
    look_path.unlink()
    assert load_sha256_cached_result("test_partial_cache", cfg) is None

    # Re-save, then delete meta only
    save_sha256_cached_result(res, cfg)
    meta_path.unlink()
    assert load_sha256_cached_result("test_partial_cache", cfg) is None


def test_corrupted_parquet_bytes_clean_none(tmp_path: Path):
    """Verify that arbitrary corrupt bytes in parquet file cleanly return None."""
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("test_parquet_corrupt_bytes")
    save_sha256_cached_result(res, cfg)

    key = _compute_engine_cache_key("test_parquet_corrupt_bytes", cfg)
    ass_path = tmp_path / f"cached_{key}_assignments.parquet"
    assert ass_path.exists()

    ass_path.write_bytes(b"INVALID_PARQUET_HEADER_DATA_12345")
    assert load_sha256_cached_result("test_parquet_corrupt_bytes", cfg) is None


def test_corrupted_parquet_body_clean_fallback_without_panic(tmp_path: Path):
    """
    Adversarial test: corrupt Parquet body while keeping valid Parquet footer intact.
    Under Polars, reading such files triggers a Rust assertion failure:
    `range.end as u64 <= self.size`.
    `load_sha256_cached_result` must catch PyO3 panic exceptions and cleanly return None.
    """
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("test_parquet_corruption")
    save_sha256_cached_result(res, cfg)

    key = _compute_engine_cache_key("test_parquet_corruption", cfg)
    ass_path = tmp_path / f"cached_{key}_assignments.parquet"
    assert ass_path.exists()

    # Create a corrupted parquet file with valid footer pointing past truncated file length
    content = ass_path.read_bytes()
    footer_len = int.from_bytes(content[-8:-4], "little")
    footer = content[-8 - footer_len:]
    ass_path.write_bytes(content[:50] + footer)

    # Must cleanly return None without unhandled pyo3_runtime.PanicException
    loaded = load_sha256_cached_result("test_parquet_corruption", cfg)
    assert loaded is None, f"Expected clean None on corrupted parquet, got {loaded}"


# ---------------------------------------------------------------------------
# 3. Atomic Write Durability & Mid-Write Cleanup
# ---------------------------------------------------------------------------
def test_atomic_write_mid_write_failure_cleans_temporary_files(tmp_path: Path):
    """
    Verify that an error occurring mid-write unlinks all partial .tmp_* files
    and does not leave behind corrupted target cache files.
    """
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("test_atomic_failure")

    with patch.object(res.lookalikes_df, "write_parquet", side_effect=IOError("Simulated disk full mid-write")):
        with pytest.raises(IOError):
            save_sha256_cached_result(res, cfg)

    # Verify no dangling temporary files exist
    tmp_files = list(tmp_path.glob(".tmp_*"))
    assert len(tmp_files) == 0, f"Dangling temporary files detected after failure: {tmp_files}"

    # Verify no partial target files were published
    target_files = list(tmp_path.glob("cached_*"))
    assert len(target_files) == 0, f"Partial target files found after failure: {target_files}"

    # Verify load returns None
    assert load_sha256_cached_result("test_atomic_failure", cfg) is None


# ---------------------------------------------------------------------------
# 4. Concurrency Safety
# ---------------------------------------------------------------------------
def test_concurrent_multi_worker_save_no_file_not_found_error(tmp_path: Path):
    """
    Adversarial test: 8 concurrent workers invoke save_sha256_cached_result
    with the exact same cache key simultaneously.
    Workers must NOT collide on temporary file paths, race on replace(), or raise FileNotFoundError.
    Guarded with a 5-second timeout to prevent indefinite hangs.
    """
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("shared_concurrent_key")

    def _worker(worker_id: int) -> str:
        try:
            save_sha256_cached_result(res, cfg)
            return "OK"
        except Exception as e:
            return f"FAIL_{worker_id}: {type(e).__name__}: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, i) for i in range(8)]
        # Bounded wait with 5.0 second timeout
        results = [f.result(timeout=5.0) for f in futures]

    failures = [r for r in results if r != "OK"]
    assert not failures, f"Concurrent saves failed: {failures}"

    # Verify no dangling temporary files
    dangling = list(tmp_path.glob(".tmp_*"))
    assert len(dangling) == 0, f"Dangling temporary files detected: {dangling}"

    # Verify resulting cache loads cleanly
    loaded = load_sha256_cached_result("shared_concurrent_key", cfg)
    assert loaded is not None
    assert loaded.assignments_df.height == 500


def test_concurrent_reads_during_active_writes(tmp_path: Path):
    """
    Empirical test: Verify that concurrent reads during ongoing writes do not crash
    or observe partially written/corrupt states. Guarded by a 5.0s timeout.
    """
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    res = _create_synthetic_result("read_write_concurrency_key")

    # Seed initial valid cache
    save_sha256_cached_result(res, cfg)

    stop = [False]
    read_errors: list[str] = []
    successful_reads = [0]

    def _reader():
        while not stop[0]:
            try:
                loaded = load_sha256_cached_result("read_write_concurrency_key", cfg)
                if loaded is not None:
                    assert loaded.assignments_df.height == 500
                    successful_reads[0] += 1
            except Exception as e:
                read_errors.append(f"{type(e).__name__}: {e}")
            time.sleep(0.001)

    def _writer():
        for _ in range(15):
            save_sha256_cached_result(res, cfg)
            time.sleep(0.005)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_reader = executor.submit(_reader)
        f_writer = executor.submit(_writer)

        f_writer.result(timeout=5.0)
        stop[0] = True
        f_reader.result(timeout=5.0)

    assert not read_errors, f"Read errors during concurrent writes: {read_errors}"
    assert successful_reads[0] > 0, "No successful reads occurred during concurrency test"
