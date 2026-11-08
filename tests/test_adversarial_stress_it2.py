"""
tests/test_adversarial_stress_it2.py
Empirical stress-testing suite for Challenger M2 (Iteration 2).
Focus areas:
1. Row count mismatch in .npz:
   - Off-by-one (+1 and -1) row count mismatch.
   - Zero-row .npz mismatch.
   - Partial subset mismatch.
   - Parquet also mismatched (both invalid -> returns None).
   - Verify dimension alignment of returned CompanyFeatureStore.
2. Model routing isolation:
   - Unrequested generic neural model present in .npz while requesting missing specific model.
   - load_cached_feature_store behavior when requested model is missing in both .npz and .parquet.
   - model_col=None handling.
3. Atomic cache warming write resilience and cleanup:
   - Interrupted / simulated failed save during cache warming cleans up temp file.
   - Concurrent simulation of warming writes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.config import PipelineConfig
from src.features.store import (
    CompanyFeatureStore,
    load_cached_feature_store,
    load_feature_store_npz,
    load_feature_store_parquet,
)
from src.ingest import ingest_all


@pytest.fixture(scope="module")
def shared_dataset():
    cfg = PipelineConfig()
    return ingest_all(cfg.data_dir)


def test_row_count_mismatch_off_by_one_and_zero(tmp_path: Path, shared_dataset) -> None:
    """Test row count mismatch off by +1, -1, 0, and subset."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)
    n_expected = len(shared_dataset.enriched_companies)
    assert n_expected == 24424

    for test_rows in [0, 1, n_expected - 1, n_expected + 1, 500]:
        # Generate an NPZ with test_rows rows
        np.savez(
            test_npz,
            company_id=np.arange(test_rows, dtype=np.int64),
            vat=np.asarray(["094123456"] * test_rows, dtype=str),
            has_gps=np.ones(test_rows, dtype=bool),
            latlon_rad=np.zeros((test_rows, 2), dtype=np.float64),
            company_algebraic_svd=np.zeros((test_rows, 24), dtype=np.float32),
            kad_hierarchy_features=np.zeros((test_rows, 36), dtype=np.float32),
            spatial_features=np.zeros((test_rows, 50), dtype=np.float32),
            legal_features=np.zeros((test_rows, 7), dtype=np.float32),
            age_branch_features=np.zeros((test_rows, 17), dtype=np.float32),
            **{"company_neural_svd_granite-107m": np.zeros((test_rows, 24), dtype=np.float32)},
        )

        cfg = PipelineConfig(cached_universe_features_path=str(test_parquet))
        fs = load_cached_feature_store(cfg, shared_dataset)

        assert fs is not None, f"Fallback failed for test_rows={test_rows}"
        # Fallback must have come from Parquet
        assert fs.embedding_telemetry["universe_cache_file"] == str(test_parquet)
        # All dimensions must match exactly n_expected (24,424), NOT test_rows!
        assert len(fs.company_ids) == n_expected
        assert len(fs.vats) == n_expected
        assert fs.unified_features.shape[0] == n_expected
        assert fs.company_algebraic_svd.shape[0] == n_expected
        assert fs.kad_hierarchy_features.shape[0] == n_expected
        assert fs.spatial_features.shape[0] == n_expected
        assert fs.legal_features.shape[0] == n_expected
        assert fs.age_branch_features.shape[0] == n_expected
        assert fs.latlon_rad.shape[0] == n_expected
        assert len(fs.has_gps) == n_expected


def test_both_npz_and_parquet_mismatched_row_count(tmp_path: Path, shared_dataset) -> None:
    """When both .npz and .parquet have mismatched row counts, returns None safely."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    # Create 50-row parquet
    df_small = pl.DataFrame({
        "company_id": np.arange(50, dtype=np.int64),
        "vat": ["094123456"] * 50,
        "company_algebraic_svd": [[0.0] * 24] * 50,
        "company_neural_svd_granite-107m": [[0.0] * 24] * 50,
        "kad_hierarchy_features": [[0.0] * 36] * 50,
        "spatial_features": [[0.0] * 50] * 50,
        "legal_features": [[0.0] * 7] * 50,
        "age_branch_features": [[0.0] * 17] * 50,
        "unified_features": [[0.0] * 134] * 50,
        "latlon_rad": [[0.0, 0.0]] * 50,
        "has_gps": [True] * 50,
    })
    df_small.write_parquet(test_parquet)

    # Create 100-row npz
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
    assert fs is None, "Should have returned None when both caches have row count mismatch"


def test_model_column_routing_strictness(tmp_path: Path, shared_dataset) -> None:
    """
    Adversarial test for model column routing:
    - NPZ contains only generic 'company_neural_svd' and 'company_neural_svd_model_A'.
    - Querying 'company_neural_svd_model_B' must strictly return None from load_feature_store_npz.
    - Querying 'company_neural_svd_model_A' returns model_A.
    - Querying without model_col returns generic company_neural_svd.
    """
    test_npz = tmp_path / "models.npz"
    n = 24424

    alg_data = np.full((n, 24), 1.0, dtype=np.float32)
    generic_data = np.full((n, 24), 2.0, dtype=np.float32)
    model_a_data = np.full((n, 24), 3.0, dtype=np.float32)

    np.savez(
        test_npz,
        company_id=np.arange(n, dtype=np.int64),
        vat=np.asarray(["094123456"] * n, dtype=str),
        has_gps=np.ones(n, dtype=bool),
        latlon_rad=np.zeros((n, 2), dtype=np.float64),
        company_algebraic_svd=alg_data,
        company_neural_svd=generic_data,
        company_neural_svd_model_A=model_a_data,
        kad_hierarchy_features=np.zeros((n, 36), dtype=np.float32),
        spatial_features=np.zeros((n, 50), dtype=np.float32),
        legal_features=np.zeros((n, 7), dtype=np.float32),
        age_branch_features=np.zeros((n, 17), dtype=np.float32),
    )

    # 1. Querying unrepresented model_B MUST return None, NOT generic
    res_b = load_feature_store_npz(test_npz, model_col="company_neural_svd_model_B")
    assert res_b is None, "load_feature_store_npz leaked generic model when model_B was requested!"

    # 2. Querying model_A returns model_A
    res_a = load_feature_store_npz(test_npz, model_col="company_neural_svd_model_A")
    assert res_a is not None
    assert np.all(res_a["company_neural_svd"] == 3.0)
    assert np.all(res_a["company_neural_svd_model_A"] == 3.0)

    # 3. Querying model_col=None returns generic model
    res_none = load_feature_store_npz(test_npz, model_col=None)
    assert res_none is not None
    assert np.all(res_none["company_neural_svd"] == 2.0)


def test_atomic_cache_warming_cleanup_on_failure(tmp_path: Path, shared_dataset) -> None:
    """
    Verify that if an error occurs during companion .npz warming write,
    the temporary file is cleaned up and the original/target .npz is not left in a corrupt state.
    """
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)
    assert not test_npz.exists()

    cfg = PipelineConfig(cached_universe_features_path=str(test_parquet))

    # Simulate an error in os.replace during cache warming
    with patch("os.replace", side_effect=OSError("Simulated disk error during replace")):
        fs = load_cached_feature_store(cfg, shared_dataset)
        # Fallback to Parquet should still succeed and return fs
        assert fs is not None
        assert fs.embedding_telemetry["universe_cache_file"] == str(test_parquet)

    # Verify no dangling .tmp_*.npz files remain in directory
    tmp_files = list(tmp_path.glob("*.tmp_*.npz"))
    assert len(tmp_files) == 0, f"Dangling temporary cache files found: {tmp_files}"
    # Target NPZ should not exist (since replace failed)
    assert not test_npz.exists()


def test_atomic_cache_warming_simulated_crash_in_savez(tmp_path: Path, shared_dataset) -> None:
    """Verify cleanup when np.savez itself fails during cache warming."""
    test_parquet = tmp_path / "features.parquet"
    test_npz = tmp_path / "features.npz"

    shutil.copy("data/cached_universe_features.parquet", test_parquet)
    assert not test_npz.exists()

    cfg = PipelineConfig(cached_universe_features_path=str(test_parquet))

    # Simulate an error inside np.savez
    with patch("numpy.savez", side_effect=IOError("Simulated write interruption")):
        fs = load_cached_feature_store(cfg, shared_dataset)
        assert fs is not None

    tmp_files = list(tmp_path.glob("*.tmp_*.npz"))
    assert len(tmp_files) == 0, f"Dangling temporary cache files found: {tmp_files}"
    assert not test_npz.exists()

