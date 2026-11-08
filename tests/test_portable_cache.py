"""
Milestone 3 Unit Tests: Portable Parquet/JSON Cache, Scoring Kernel & Dataclass Unification
===========================================================================================
Covers Features 11, 12, 13, 14, 15:
- Feature 11: Modular exported scoring kernel and EB log-lifts helpers
- Feature 12: Clean signature of score_companies_against_icps returning np.ndarray
- Feature 13: PipelineConfig.trait_weights externalization and normalization
- Feature 14: Portable Parquet/JSON cache serialization with zero pickle files
- Feature 15: Dataclass replace in make_icp_engine_ranker
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import DEFAULT_TRAIT_WEIGHTS, PipelineConfig
from src.icp_engine import (
    DiscoveredICP,
    ICPEngineResult,
    ICPProfile,
    StatisticalFeatureProof,
    build_vectorized_categorical_tables,
    compute_eb_log_lifts_vectorized,
    compute_scoring_kernel,
    load_sha256_cached_result,
    save_sha256_cached_result,
    score_companies_against_icps,
)


@pytest.fixture(scope="module")
def engine_service():
    """Retrieve warmed singleton EngineService."""
    from src.server import get_service
    return get_service()


@pytest.fixture(scope="module")
def universe_feature_store(engine_service):
    """Retrieve full company feature store."""
    cfg = engine_service.defaults_cfg
    if engine_service.last_fs is not None:
        return engine_service.last_fs
    return engine_service._feature_store("granite-107m", 0.50, cfg)


@pytest.fixture(scope="module")
def baseline_result(engine_service):
    """Retrieve computed baseline ICP result."""
    if engine_service.last_result is not None:
        return engine_service.last_result
    return engine_service._run_engine(engine_service.defaults_cfg)


@pytest.fixture
def mock_icp_result() -> ICPEngineResult:
    """Create a fully populated synthetic ICPEngineResult for round-trip testing."""
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
        ai_description_el="Περιγραφή δοκιμαστικού προφίλ με ελληνικούς χαρακτήρες.",
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
    profile.model_name = "test-model-v1"

    assignments_df = pl.DataFrame({
        "company_id": [101, 102, 103],
        "profile_id": ["ICP_1", "ICP_1", "NO_FIT"],
        "score": [0.88, 0.76, 0.22],
    })

    lookalikes_df = pl.DataFrame({
        "profile_id": ["ICP_1", "ICP_1"],
        "company_id": [201, 202],
        "rank": [1, 2],
        "score": [0.91, 0.89],
        "αιτιολόγηση": [
            "Ταύτιση 91.0% με ICP_1: Περιοχή υψηλής διείσδυσης (ΑΤΤΙΚΗΣ)",
            "Ταύτιση 89.0% με ICP_1: Νομική μορφή ΙΚΕ",
        ],
    })

    return ICPEngineResult(
        optimal_k=1,
        global_bootstrap_ari=0.94,
        silhouette_by_k={1: 0.0, 2: 0.35},
        profiles=[profile],
        assignments_df=assignments_df,
        lookalikes_df=lookalikes_df,
        all_candidates_ranked=[201, 202, 301, 302],
        client_sha256="test_sha256_hash_1234567890abcdef",
        loaded_from_sha256_cache=False,
    )


def test_portable_cache_roundtrip_parquet_and_json(tmp_path: Path, mock_icp_result: ICPEngineResult):
    """Verify save_sha256_cached_result and load_sha256_cached_result use Parquet and JSON with 100% fidelity."""
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)

    # 1. Save result
    meta_path = save_sha256_cached_result(mock_icp_result, cfg)
    assert meta_path.exists()
    assert meta_path.suffix == ".json"

    # Verify zero .pkl files generated
    pkl_files = list(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == 0, f"Expected 0 pickle files, found: {pkl_files}"

    # Verify Parquet and JSON files exist
    json_files = list(tmp_path.glob("*.json"))
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(json_files) >= 1
    assert len(parquet_files) >= 2  # assignments and lookalikes

    # Verify no dangling temporary files
    tmp_files = list(tmp_path.glob(".tmp_*"))
    assert len(tmp_files) == 0, f"Dangling temporary files detected: {tmp_files}"

    # 2. Load result back
    loaded = load_sha256_cached_result(mock_icp_result.client_sha256, cfg)
    assert loaded is not None
    assert loaded.loaded_from_sha256_cache is True
    assert loaded.optimal_k == mock_icp_result.optimal_k
    assert loaded.global_bootstrap_ari == mock_icp_result.global_bootstrap_ari
    assert loaded.silhouette_by_k == mock_icp_result.silhouette_by_k
    assert loaded.all_candidates_ranked == mock_icp_result.all_candidates_ranked
    assert loaded.client_sha256 == mock_icp_result.client_sha256

    # Verify DataFrames round-trip bit-exact
    assert loaded.assignments_df.equals(mock_icp_result.assignments_df)
    assert loaded.lookalikes_df.equals(mock_icp_result.lookalikes_df)

    # Verify profiles round-trip
    assert len(loaded.profiles) == len(mock_icp_result.profiles)
    p_orig = mock_icp_result.profiles[0]
    p_load = loaded.profiles[0]
    assert p_load.profile_id == p_orig.profile_id
    assert p_load.title_el == p_orig.title_el
    assert p_load.ai_description_el == p_orig.ai_description_el
    assert p_load.client_count == p_orig.client_count
    assert p_load.client_share == p_orig.client_share
    assert p_load.turnover_total == p_orig.turnover_total
    assert np.allclose(p_load.centroid_vector, p_orig.centroid_vector, atol=1e-6)
    assert p_load.spatial_medoid_latlon_deg == p_orig.spatial_medoid_latlon_deg
    assert p_load.eb_log_lifts == p_orig.eb_log_lifts
    assert getattr(p_load, "model_name", None) == "test-model-v1"

    # Verify diagnostics property
    diag = loaded.diagnostics
    assert diag["optimal_k"] == 1
    assert diag["global_bootstrap_ari"] == 0.94
    assert diag["client_sha256"] == mock_icp_result.client_sha256


def test_portable_cache_miss_and_disabled(tmp_path: Path):
    """Verify cache miss returns None and use_sha256_client_cache=False disables loading."""
    cfg = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=True)
    assert load_sha256_cached_result("unknown_hash", cfg) is None

    cfg_disabled = PipelineConfig(cache_dir=tmp_path, use_sha256_client_cache=False)
    assert load_sha256_cached_result("unknown_hash", cfg_disabled) is None


def test_trait_weights_configuration_and_normalization():
    """Verify Feature 13: PipelineConfig.trait_weights externalization and normalization."""
    cfg = PipelineConfig()
    assert len(cfg.trait_weights) == len(DEFAULT_TRAIT_WEIGHTS)
    assert cfg.trait_weights == DEFAULT_TRAIT_WEIGHTS
    assert isinstance(cfg.trait_weights, tuple)

    # Dict input normalization in __post_init__
    cfg_dict = PipelineConfig(trait_weights={"primary_kad": 2.5, "legal_type": 1.5})
    assert isinstance(cfg_dict.trait_weights, tuple)
    assert cfg_dict.trait_weights == (("primary_kad", 2.5), ("legal_type", 1.5))

    # List of tuples normalization
    cfg_list = PipelineConfig(trait_weights=[("prefecture_name", 1.8), ("branch_status", 0.5)])
    assert isinstance(cfg_list.trait_weights, tuple)
    assert cfg_list.trait_weights == (("prefecture_name", 1.8), ("branch_status", 0.5))

    # Dataclass replace
    replaced = replace(cfg, trait_weights=(("osm_tier", 0.8),))
    assert replaced.trait_weights == (("osm_tier", 0.8),)
    assert cfg.trait_weights == DEFAULT_TRAIT_WEIGHTS  # original unmodified


def test_make_icp_engine_ranker_dataclass_replace(universe_feature_store):
    """Verify Feature 15: make_icp_engine_ranker uses dataclasses.replace cleanly."""
    from evaluate_holdout import make_icp_engine_ranker

    base_cfg = PipelineConfig()
    ranker = make_icp_engine_ranker(
        fs=universe_feature_store,
        cfg=base_cfg,
        fast_cv_mode=True,
    )
    assert callable(ranker)
    # Original config unchanged
    assert base_cfg.kmeans_n_init == 200
    assert base_cfg.n_bootstrap_runs == 200
    assert base_cfg.use_sha256_client_cache is True


def test_compute_scoring_kernel_bounds_and_monotonicity():
    """Verify Feature 11: compute_scoring_kernel produces bounded, well-calibrated scores."""
    n, k = 10, 2
    cos_sim = np.array([[0.8, 0.4], [0.2, 0.9]], dtype=np.float32)
    eb_raw = np.array([[1.5, -0.5], [0.0, 2.0]], dtype=np.float32)
    sp_sim = np.array([0.7, 0.3], dtype=np.float32)

    scores = compute_scoring_kernel(
        cos_sim_mat=cos_sim,
        eb_raw_mat=eb_raw,
        spatial_sim=sp_sim,
        w_cos=0.6,
        w_eb=0.3,
        w_sp=0.1,
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2, 2)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Monotonicity with respect to cosine similarity
    scores_higher_cos = compute_scoring_kernel(
        cos_sim_mat=cos_sim + 0.1,
        eb_raw_mat=eb_raw,
        spatial_sim=sp_sim,
        w_cos=0.6,
        w_eb=0.3,
        w_sp=0.1,
    )
    assert (scores_higher_cos >= scores).all()


def test_score_companies_against_icps_signature_cleanup(engine_service, universe_feature_store, baseline_result):
    """Verify Feature 12: score_companies_against_icps returns np.ndarray directly (no tuple)."""
    fs = universe_feature_store
    profiles = baseline_result.profiles
    cfg = PipelineConfig()
    ec_sorted = engine_service._meta_full

    client_rep_mask = np.where(ec_sorted["is_client_vat"].to_numpy())[0]
    client_latlon = fs.latlon_rad[client_rep_mask]
    client_weights = np.ones(len(client_rep_mask), dtype=np.float32)

    res = score_companies_against_icps(
        ec_sorted=ec_sorted,
        fs=fs,
        profiles=profiles,
        client_latlon_rad=client_latlon,
        client_weights=client_weights,
        cfg=cfg,
    )
    # Must be np.ndarray, NOT tuple
    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape == (ec_sorted.height, len(profiles))
    assert (res >= 0.0).all() and (res <= 1.0).all()


def test_zero_pickle_files_in_cache_directory():
    """Verify Section 10 & Milestone 3 invariant: zero .pkl files exist in output/cache/."""
    cache_path = Path("output/cache")
    if cache_path.exists():
        pkl_files = list(cache_path.glob("*.pkl"))
        assert len(pkl_files) == 0, f"Found legacy .pkl files in output/cache: {pkl_files}"
