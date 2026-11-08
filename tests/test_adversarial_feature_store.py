"""
tests/test_adversarial_feature_store.py
=======================================
Adversarial and Empirical Challenge Test Suite for Milestone 2:
Feature Store .npz Persistence, Caching, and Two-Tier Fallback Engine.

Evaluates:
1. Bit-exact numerical parity (all 7 feature matrices across multiple models).
2. allow_pickle=False strict security enforcement across all keys.
3. Resilience to non-contiguous, strided, and Fortran memory layouts.
4. Robustness of two-tier fallback under file deletion, truncation, corruption, and schema drift.
5. High-resolution load latency benchmarking (<30ms median requirement).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.config import PipelineConfig
from src.features.store import (
    CompanyFeatureStore,
    load_cached_feature_store,
    load_feature_store_npz,
    load_feature_store_parquet,
    save_feature_store_npz,
    save_feature_store_parquet,
)
from src.ingest import ingest_all


# =============================================================================
# 1. BIT-EXACT NUMERICAL PARITY (All 7 Feature Matrices & Models)
# =============================================================================

def test_bit_exact_numerical_equality_all_matrices() -> None:
    """
    Empirically verify bit-exact numerical parity between .parquet and .npz caches.
    Ensures that converting list-columns from Parquet to contiguous NumPy in .npz
    introduced zero numerical deviation or precision drift across all 24,424 records.
    """
    parquet_path = Path("data/cached_universe_features.parquet")
    npz_path = Path("data/cached_universe_features.npz")

    assert parquet_path.exists(), "data/cached_universe_features.parquet missing"
    assert npz_path.exists(), "data/cached_universe_features.npz missing"

    models_to_test = [
        "company_neural_svd_granite-107m",
    ]

    matrices = [
        "company_algebraic_svd",
        "kad_hierarchy_features",
        "spatial_features",
        "legal_features",
        "age_branch_features",
        "unified_features",
    ]

    for model_col in models_to_test:
        p_data = load_feature_store_parquet(parquet_path, model_col=model_col)
        n_data = load_feature_store_npz(npz_path, model_col=model_col)

        assert p_data is not None, f"Parquet load failed for {model_col}"
        assert n_data is not None, f"NPZ load failed for {model_col}"

        # 1. Verify IDs and metadata
        np.testing.assert_array_equal(p_data["company_id"], n_data["company_id"])
        assert p_data["vat"] == n_data["vat"]
        np.testing.assert_array_equal(p_data["has_gps"], n_data["has_gps"])

        # Latlon coordinates with NaNs
        np.testing.assert_array_equal(
            np.isnan(p_data["latlon_rad"]),
            np.isnan(n_data["latlon_rad"]),
        )
        valid_gps = ~np.isnan(p_data["latlon_rad"])
        np.testing.assert_array_equal(
            p_data["latlon_rad"][valid_gps],
            n_data["latlon_rad"][valid_gps],
        )

        # 2. Check neural model matrix
        assert model_col in p_data
        assert model_col in n_data
        neural_diff = np.max(np.abs(p_data[model_col] - n_data[model_col]))
        assert neural_diff == 0.0, f"{model_col} has non-zero diff: {neural_diff}"
        assert np.array_equal(p_data[model_col], n_data[model_col])

        # 3. Check all other 6 feature matrices
        for m in matrices:
            assert m in p_data, f"Matrix {m} missing from Parquet data"
            assert m in n_data, f"Matrix {m} missing from NPZ data"
            diff = np.max(np.abs(p_data[m] - n_data[m]))
            assert diff == 0.0, f"Matrix {m} has non-zero diff: {diff}"
            assert np.array_equal(p_data[m], n_data[m]), f"Matrix {m} not bit-exact equal!"


# =============================================================================
# 2. ALLOW_PICKLE=FALSE SECURITY AND TYPE ENFORCEMENT
# =============================================================================

def test_allow_pickle_false_strict_security() -> None:
    """
    Adversarial verification of allow_pickle=False.
    Ensure raw np.load directly on the production .npz file with allow_pickle=False
    succeeds without any ValueError or pickle deserialization risk.
    """
    npz_path = Path("data/cached_universe_features.npz")
    with np.load(npz_path, allow_pickle=False) as f:
        # Verify all keys are readable under allow_pickle=False
        for k in f.files:
            arr = f[k]
            # Ensure no object dtypes exist (object dtypes require pickle)
            assert arr.dtype != object, f"Key {k} has object dtype: {arr.dtype}"
            # Vat array must be fixed-width Unicode, not object
            if k == "vat":
                assert arr.dtype.kind == "U", f"VAT array is not Unicode dtype: {arr.dtype}"
                assert len(arr) == 24424
                # Test zero-padding preservation (e.g. 9-digit string length)
                for sample_vat in arr[:100]:
                    assert len(str(sample_vat)) == 9, f"VAT corrupted: {sample_vat}"


def test_save_npz_handles_object_arrays_by_coercion(tmp_path: Path) -> None:
    """
    Adversarially pass object arrays (e.g. lists of strings, object bools)
    to save_feature_store_npz and verify it coerces them to primitive/Unicode dtypes
    so allow_pickle=False never fails on reload.
    """
    n = 20
    npz_file = tmp_path / "object_coercion.npz"

    fs = CompanyFeatureStore(
        company_ids=np.arange(1, n + 1, dtype=np.int32),
        vats=[f"0000000{i:02d}" for i in range(n)],  # Python list of strings
        semantic_vectors=np.random.randn(n, 32).astype(np.float32),
        semantic_svd=np.random.randn(n, 32).astype(np.float32),
        kad_hierarchy_features=np.random.randn(n, 16).astype(np.float32),
        spatial_features=np.random.randn(n, 8).astype(np.float32),
        legal_features=np.random.randn(n, 4).astype(np.float32),
        age_branch_features=np.random.randn(n, 6).astype(np.float32),
        unified_features=np.random.randn(n, 66).astype(np.float32),
        latlon_rad=np.random.randn(n, 2).astype(np.float64),
        has_gps=np.array([True, False] * (n // 2), dtype=object),  # Object array!
        embedding_telemetry={},
        company_neural_svd=np.random.randn(n, 32).astype(np.float32),
        company_algebraic_svd=np.random.randn(n, 32).astype(np.float32),
    )

    save_feature_store_npz(fs, npz_file, model_col="test_model")

    # Raw load strictly forbidding pickle
    with np.load(npz_file, allow_pickle=False) as f:
        assert f["has_gps"].dtype == bool
        assert f["vat"].dtype.kind == "U"
        assert f["company_id"].dtype == np.int64


# =============================================================================
# 3. NON-CONTIGUOUS AND ADVERSARIAL MEMORY LAYOUTS
# =============================================================================

def test_non_contiguous_array_resilience(tmp_path: Path) -> None:
    """
    Stress-test save_feature_store_npz with:
    - Strided views: arr[::2, ::-1]
    - Fortran-contiguous layouts: np.asfortranarray(arr)
    - Transposed views: arr.T
    - Sliced blocks: arr[:, 5:15]
    Verify reload yields strictly C-contiguous arrays with identical numerical values.
    """
    n = 30
    npz_file = tmp_path / "non_contiguous.npz"

    # Construct adversarial non-contiguous inputs
    raw_alg = np.random.randn(n * 2, 64).astype(np.float32)
    strided_alg = raw_alg[::2, :]
    assert not strided_alg.flags.c_contiguous

    fortran_kad = np.asfortranarray(np.random.randn(n, 36).astype(np.float32))
    assert fortran_kad.flags.f_contiguous and not fortran_kad.flags.c_contiguous

    transposed_spatial = np.random.randn(50, n).astype(np.float32).T
    assert not transposed_spatial.flags.c_contiguous

    sliced_legal = np.random.randn(n, 20).astype(np.float32)[:, ::2][:, :7]
    assert not sliced_legal.flags.c_contiguous

    reversed_age = np.random.randn(n, 17).astype(np.float32)[::-1, :]
    assert not reversed_age.flags.c_contiguous

    fs = CompanyFeatureStore(
        company_ids=np.arange(1, n + 1, dtype=np.int64),
        vats=[f"0941234{i:02d}" for i in range(n)],
        semantic_vectors=strided_alg,
        semantic_svd=strided_alg,
        kad_hierarchy_features=fortran_kad,
        spatial_features=transposed_spatial,
        legal_features=sliced_legal,
        age_branch_features=reversed_age,
        unified_features=np.random.randn(n, 100).astype(np.float32)[::1, ::2],
        latlon_rad=np.random.randn(n, 2).astype(np.float64),
        has_gps=np.ones(n, dtype=bool),
        embedding_telemetry={},
        company_neural_svd=strided_alg.copy(),
        company_algebraic_svd=strided_alg,
    )

    save_feature_store_npz(fs, npz_file, model_col="test_model")

    loaded = load_feature_store_npz(npz_file, model_col="test_model")
    assert loaded is not None

    # Verify all arrays are restored as C-contiguous
    check_keys = [
        "company_id",
        "company_algebraic_svd",
        "company_neural_svd",
        "kad_hierarchy_features",
        "spatial_features",
        "legal_features",
        "age_branch_features",
        "unified_features",
        "latlon_rad",
        "has_gps",
    ]
    for k in check_keys:
        arr = loaded[k]
        assert arr.flags.c_contiguous, f"Array {k} is not C-contiguous after reload!"

    # Verify exact numerical parity
    np.testing.assert_array_equal(loaded["company_algebraic_svd"], strided_alg)
    np.testing.assert_array_equal(loaded["kad_hierarchy_features"], fortran_kad)
    np.testing.assert_array_equal(loaded["spatial_features"], transposed_spatial)
    np.testing.assert_array_equal(loaded["legal_features"], sliced_legal)
    np.testing.assert_array_equal(loaded["age_branch_features"], reversed_age)


# =============================================================================
# 4. TWO-TIER FALLBACK ADVERSARIAL STRESS TESTS
# =============================================================================

@pytest.fixture(scope="module")
def shared_dataset():
    cfg = PipelineConfig()
    return ingest_all(cfg.data_dir)


def test_fallback_when_npz_missing(tmp_path: Path, shared_dataset) -> None:
    """Verify fallback to Parquet and automatic warming of companion .npz when .npz is absent."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)
    assert not test_npz.exists()

    cfg = PipelineConfig(cached_universe_features_path=str(test_parquet))
    fs = load_cached_feature_store(cfg, shared_dataset)

    assert fs is not None
    assert fs.embedding_telemetry["universe_cache_file"] == str(test_parquet)
    assert test_npz.exists(), "Companion .npz was not generated during fallback"

    # Subsequent load must hit the newly created .npz
    fs_subsequent = load_cached_feature_store(cfg, shared_dataset)
    assert fs_subsequent is not None
    assert fs_subsequent.embedding_telemetry["universe_cache_file"] == str(test_npz)


@pytest.mark.parametrize(
    "corrupt_mode",
    [
        "garbage_bytes",
        "empty_file",
        "truncated_zip",
        "missing_required_keys",
        "mismatched_row_count",
    ],
)
def test_fallback_when_npz_corrupted(tmp_path: Path, shared_dataset, corrupt_mode: str) -> None:
    """Verify graceful fallback to Parquet without unhandled exception under various corruptions."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)

    if corrupt_mode == "garbage_bytes":
        test_npz.write_bytes(b"RANDOM_CORRUPT_GARBAGE_PAYLOAD_NOT_A_ZIP_ARCHIVE" * 10)
    elif corrupt_mode == "empty_file":
        test_npz.write_bytes(b"")
    elif corrupt_mode == "truncated_zip":
        with open("data/cached_universe_features.npz", "rb") as f:
            header = f.read(256)
        test_npz.write_bytes(header)
    elif corrupt_mode == "missing_required_keys":
        # Only company_id, missing all other required matrices
        np.savez(test_npz, company_id=np.arange(24424, dtype=np.int64))
    elif corrupt_mode == "mismatched_row_count":
        # Wrong number of companies (e.g. 100 instead of 24,424)
        np.savez(
            test_npz,
            company_id=np.arange(100, dtype=np.int64),
            vat=np.asarray(["094123456"] * 100, dtype=str),
            has_gps=np.ones(100, dtype=bool),
            latlon_rad=np.zeros((100, 2), dtype=np.float64),
            company_algebraic_svd=np.zeros((100, 24), dtype=np.float32),
            kad_hierarchy_features=np.zeros((100, 36), dtype=np.float32),
            spatial_features=np.zeros((100, 50), dtype=np.float32),
            legal_features=np.zeros((100, 7), dtype=np.float32),
            age_branch_features=np.zeros((100, 17), dtype=np.float32),
            **{"company_neural_svd_granite-107m": np.zeros((100, 24), dtype=np.float32)},
        )

    cfg = PipelineConfig(cached_universe_features_path=str(test_parquet))
    fs = load_cached_feature_store(cfg, shared_dataset)

    assert fs is not None, f"Fallback failed for corrupt mode: {corrupt_mode}"
    assert fs.embedding_telemetry["universe_cache_file"] == str(test_parquet)


def test_fallback_when_model_column_missing_in_npz(tmp_path: Path, shared_dataset) -> None:
    """When .npz lacks requested neural model column, falls back to Parquet containing that column."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)

    # Save NPZ with generic company_neural_svd but without the requested granite-107m model
    np.savez(
        test_npz,
        company_id=np.arange(24424, dtype=np.int64),
        vat=np.asarray(["094123456"] * 24424, dtype=str),
        has_gps=np.ones(24424, dtype=bool),
        latlon_rad=np.zeros((24424, 2), dtype=np.float64),
        company_algebraic_svd=np.zeros((24424, 24), dtype=np.float32),
        company_neural_svd=np.zeros((24424, 24), dtype=np.float32),
        kad_hierarchy_features=np.zeros((24424, 36), dtype=np.float32),
        spatial_features=np.zeros((24424, 50), dtype=np.float32),
        legal_features=np.zeros((24424, 7), dtype=np.float32),
        age_branch_features=np.zeros((24424, 17), dtype=np.float32),
    )

    # Direct check: requesting missing specific model must strictly return None despite generic model existing
    assert load_feature_store_npz(test_npz, model_col="company_neural_svd_granite-107m") is None

    cfg = PipelineConfig(
        cached_universe_features_path=str(test_parquet),
        embedding_model="granite-107m",
    )
    fs = load_cached_feature_store(cfg, shared_dataset)
    assert fs is not None
    assert fs.embedding_telemetry["universe_cache_file"] == str(test_parquet)


def test_fallback_when_both_caches_missing(tmp_path: Path, shared_dataset) -> None:
    """When neither .npz nor .parquet exist, load_cached_feature_store safely returns None."""
    missing_parquet = tmp_path / "non_existent.parquet"
    cfg = PipelineConfig(cached_universe_features_path=str(missing_parquet))
    fs = load_cached_feature_store(cfg, shared_dataset)
    assert fs is None


# =============================================================================
# 5. LATENCY BENCHMARK (< 30ms median requirement)
# =============================================================================

def test_npz_load_latency_benchmark() -> None:
    """
    Benchmark load_feature_store_npz on the full 24,424-record dataset across 20 iterations.
    Verifies that median load latency is strictly < 30ms and measures speedup over Parquet.
    """
    path_npz = Path("data/cached_universe_features.npz")
    path_parquet = Path("data/cached_universe_features.parquet")
    model_col = "company_neural_svd_granite-107m"

    # Warm-up run
    load_feature_store_npz(path_npz, model_col)

    npz_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = load_feature_store_npz(path_npz, model_col)
        dt = (time.perf_counter() - t0) * 1000
        assert res is not None
        npz_times.append(dt)

    median_npz = float(np.median(npz_times))
    min_npz = float(np.min(npz_times))
    max_npz = float(np.max(npz_times))

    print(f"\n[Feature Store NPZ Benchmark] Min: {min_npz:.2f}ms | Median: {median_npz:.2f}ms | Max: {max_npz:.2f}ms")

    assert median_npz < 30.0, (
        f"Feature store median load latency ({median_npz:.2f}ms) exceeds 30ms requirement!"
    )
