"""
Adversarial Empirical Stress Tests: Milestone 3 Mathematical Engine Unification
================================================================================
Empirical Challenger 2 verification suite covering:
1. Numerical parity (< 1e-7 floating-point tolerance) for `compute_scoring_kernel`
   and `compute_eb_log_lifts_vectorized` across `src/icp_engine.py` and `evaluate_holdout.py`.
2. Trait weights mutability, score sensitivity, cache key uniqueness, and stale cache prevention.
3. Dataclass replace resilience in `make_icp_engine_ranker` ensuring strict type preservation
   for `Path` and nested `tuple` structures.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.config import DEFAULT_TRAIT_WEIGHTS, PipelineConfig
from src.icp_engine import (
    DiscoveredICP,
    ICPEngineResult,
    StatisticalFeatureProof,
    _compute_engine_cache_key,
    build_vectorized_categorical_tables,
    compute_eb_log_lifts_vectorized,
    compute_scoring_kernel,
    load_sha256_cached_result,
    save_sha256_cached_result,
    score_companies_against_icps,
)
from evaluate_holdout import (
    _compute_eb_log_lifts_vectorized as evaluate_holdout_eb_lifts,
    compute_scoring_kernel as evaluate_holdout_scoring_kernel,
    make_icp_engine_ranker,
)


@pytest.fixture(scope="module")
def engine_service():
    """Retrieve singleton EngineService with warmed dataset."""
    from src.server import get_service
    return get_service()


@pytest.fixture(scope="module")
def feature_store(engine_service):
    """Retrieve universe feature store."""
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


# ==============================================================================
# SECTION 1: NUMERICAL PARITY & FLOATING-POINT EQUIVALENCE (< 1e-7 ERROR)
# ==============================================================================

def test_scoring_kernel_identity_and_numerical_parity():
    """Verify that compute_scoring_kernel in evaluate_holdout is identical and produces < 1e-7 error."""
    # 1. Verify object identity
    assert evaluate_holdout_scoring_kernel is compute_scoring_kernel

    # 2. Monte Carlo verification against float64 analytical ground truth
    rng = np.random.default_rng(2026)
    n, k = 500, 5

    cos_sim = rng.uniform(0.0, 1.0, size=(n, k)).astype(np.float32)
    eb_raw = rng.uniform(-10.0, 10.0, size=(n, k)).astype(np.float32)
    spatial_sim_1d = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    spatial_sim_2d = spatial_sim_1d[:, None]

    w_cos, w_eb, w_sp = 0.60, 0.30, 0.10

    # Ground truth in float64
    eb_prob_f64 = 1.0 / (1.0 + np.exp(-0.55 * eb_raw.astype(np.float64)))
    sp_vec_f64 = spatial_sim_1d.astype(np.float64)[:, None]
    w_sum_f64 = w_cos + w_eb + w_sp
    expected_f64 = (
        w_cos * cos_sim.astype(np.float64)
        + w_eb * eb_prob_f64
        + w_sp * sp_vec_f64
    ) / w_sum_f64
    expected_clipped = np.clip(expected_f64, 0.0, 1.0).astype(np.float32)

    # Output from src.icp_engine
    out_icp_1d = compute_scoring_kernel(cos_sim, eb_raw, spatial_sim_1d, w_cos, w_eb, w_sp)
    out_icp_2d = compute_scoring_kernel(cos_sim, eb_raw, spatial_sim_2d, w_cos, w_eb, w_sp)

    # Output from evaluate_holdout
    out_eh_1d = evaluate_holdout_scoring_kernel(cos_sim, eb_raw, spatial_sim_1d, w_cos, w_eb, w_sp)

    # Mathematical reference in float32
    eb_prob_f32 = 1.0 / (1.0 + np.exp(-0.55 * eb_raw))
    sp_vec_f32 = spatial_sim_1d[:, None]
    w_sum_f32 = max(w_cos + w_eb + w_sp, 1e-6)
    expected_f32 = np.clip((w_cos * cos_sim + w_eb * eb_prob_f32 + w_sp * sp_vec_f32) / w_sum_f32, 0.0, 1.0)

    # 1. Cross-module parity across src/icp_engine and evaluate_holdout must be bit-exact (0.0 error < 1e-7)
    diff_eh_icp = float(np.max(np.abs(out_icp_1d - out_eh_1d)))
    assert diff_eh_icp == 0.0, f"Cross-module discrepancy: {diff_eh_icp}"
    assert np.array_equal(out_icp_1d, out_eh_1d), "evaluate_holdout and icp_engine must be bit-exact"
    assert np.array_equal(out_icp_1d, out_icp_2d), "1D and 2D spatial inputs must yield identical scores"

    # 2. Parity against float32 mathematical reference formula is bit-exact (0.0 error < 1e-7)
    diff_math_f32 = float(np.max(np.abs(out_icp_1d - expected_f32)))
    assert diff_math_f32 == 0.0, f"Mathematical float32 reference discrepancy: {diff_math_f32}"

    # 3. Parity against double-precision float64 reference is bounded by single-precision machine epsilon (2^-23 ~ 1.19e-7)
    max_diff_f64 = float(np.max(np.abs(out_icp_1d - expected_clipped)))
    f32_eps = float(np.finfo(np.float32).eps)
    assert max_diff_f64 <= f32_eps, f"Diff against float64 exceeds float32 epsilon: {max_diff_f64:.2e} > {f32_eps:.2e}"


def test_scoring_kernel_adversarial_weights_and_extremal_inputs():
    """Stress test scoring kernel against degenerate weights, near-zero denominators, and extreme raw values."""
    n, k = 100, 3
    cos = np.full((n, k), 0.5, dtype=np.float32)
    eb = np.full((n, k), 0.0, dtype=np.float32)
    sp = np.full(n, 0.5, dtype=np.float32)

    # 1. All zero weights (denominator floor = 1e-6)
    out_zero = compute_scoring_kernel(cos, eb, sp, 0.0, 0.0, 0.0)
    assert not np.isnan(out_zero).any()
    assert not np.isinf(out_zero).any()
    assert (out_zero == 0.0).all()

    # 2. Extreme weight scale disparity
    out_huge = compute_scoring_kernel(cos, eb, sp, 1e8, 1e-8, 1e-8)
    assert np.allclose(out_huge, 0.5, atol=1e-6)

    # 3. Extremal raw EB inputs (saturation to 0.0 and 1.0)
    eb_extreme = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    cos_mid = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    sp_mid = np.array([0.5], dtype=np.float32)

    out_extreme = compute_scoring_kernel(cos_mid, eb_extreme, sp_mid, 0.0, 1.0, 0.0)
    assert np.isclose(out_extreme[0, 0], 0.0, atol=1e-5)  # sigmoid(-55) ~ 0
    assert np.isclose(out_extreme[0, 1], 0.5, atol=1e-5)  # sigmoid(0) = 0.5
    assert np.isclose(out_extreme[0, 2], 1.0, atol=1e-5)  # sigmoid(+55) ~ 1


def test_eb_log_lifts_vectorized_analytical_parity():
    """Verify compute_eb_log_lifts_vectorized matches exact analytical shrinkage (< 1e-7 error)."""
    assert evaluate_holdout_eb_lifts is compute_eb_log_lifts_vectorized

    rng = np.random.default_rng(42)
    n_univ = 2000
    n_sub = 150
    m_prior = 12.0

    # Create synthetic categorical columns with varying counts
    # Dim 0: 50 categories (some < 5)
    codes_0 = rng.integers(0, 50, size=n_univ).astype(np.int32)
    # Dim 1: 5 categories (high frequency)
    codes_1 = rng.integers(0, 5, size=n_univ).astype(np.int32)

    cat_tables = {
        "n_univ": n_univ,
        "codes_list": [codes_0, codes_1],
        "univ_counts_list": [
            np.bincount(codes_0, minlength=50).astype(np.float64),
            np.bincount(codes_1, minlength=5).astype(np.float64),
        ],
        "valid_mask_list": [
            np.bincount(codes_0, minlength=50) >= 5,
            np.bincount(codes_1, minlength=5) >= 5,
        ],
        "weights_list": [1.2, 0.8],
    }

    row_indices = rng.choice(n_univ, size=n_sub, replace=False)

    # Vectorized execution
    vec_results = compute_eb_log_lifts_vectorized(cat_tables, row_indices, m_prior)

    # Analytical pure-scalar calculation in double precision
    for d_idx, (codes, k_univ, valid_mask) in enumerate(
        zip(cat_tables["codes_list"], cat_tables["univ_counts_list"], cat_tables["valid_mask_list"])
    ):
        n_cats = len(k_univ)
        k_icp = np.zeros(n_cats, dtype=np.float64)
        for r in row_indices:
            k_icp[codes[r]] += 1.0

        ref_ll = np.zeros(n_cats, dtype=np.float32)
        for c in range(n_cats):
            if valid_mask[c]:
                p_u = k_univ[c] / float(n_univ)
                p_eb = (k_icp[c] + m_prior * p_u) / (float(n_sub) + m_prior)
                lift = p_eb / max(p_u, 1e-9)
                clipped = np.clip(lift, 0.25, 8.0)
                ref_ll[c] = np.float32(np.log(clipped))
            else:
                ref_ll[c] = np.float32(0.0)

        diff = float(np.max(np.abs(vec_results[d_idx] - ref_ll)))
        assert diff < 1e-7, f"EB log-lifts analytical diff exceeds tolerance on dim {d_idx}: {diff:.2e}"


def test_eb_log_lifts_vectorized_boundary_conditions():
    """Verify boundary conditions: whole universe selection, empty selection, and clipping bounds."""
    n_univ = 500
    m_prior = 10.0
    codes = np.repeat(np.arange(10, dtype=np.int32), 50)  # 10 categories, each 50 counts
    cat_tables = {
        "n_univ": n_univ,
        "codes_list": [codes],
        "univ_counts_list": [np.full(10, 50.0, dtype=np.float64)],
        "valid_mask_list": [np.ones(10, dtype=bool)],
        "weights_list": [1.0],
    }

    # 1. Whole universe selected: ICP == Universe -> All lifts shrink to 1.0 -> log_lift == 0.0
    all_rows = np.arange(n_univ)
    ll_univ = compute_eb_log_lifts_vectorized(cat_tables, all_rows, m_prior)[0]
    assert np.allclose(ll_univ, 0.0, atol=1e-6)

    # 2. Empty selection: row_indices = np.array([], dtype=int) -> n_sub = max(0, 1) = 1
    ll_empty = compute_eb_log_lifts_vectorized(cat_tables, np.array([], dtype=np.int32), m_prior)[0]
    assert not np.isnan(ll_empty).any()
    assert (ll_empty <= 0.0).all()  # with 0 ICP counts, lift is strictly downward

    # 3. Maximum saturation: category 0 has 100% representation in a large cluster
    c0_rows = np.where(codes == 0)[0]
    ll_c0 = compute_eb_log_lifts_vectorized(cat_tables, c0_rows, m_prior)[0]
    # Category 0 should hit or approach max clip log(8.0) ~ 2.07944
    assert np.isclose(ll_c0[0], np.log(8.0), atol=0.2)


# ==============================================================================
# SECTION 2: TRAIT WEIGHTS MUTABILITY & CACHING RESILIENCE
# ==============================================================================

def test_trait_weights_mutability_produces_divergent_scores(engine_service, feature_store, baseline_result):
    """Verify that altering cfg.trait_weights directly changes the resulting score matrix."""
    fs = feature_store
    profiles = baseline_result.profiles
    ec_sorted = engine_service._meta_full

    client_rep_mask = np.where(ec_sorted["is_client_vat"].to_numpy())[0]
    client_latlon = fs.latlon_rad[client_rep_mask]
    client_weights = np.ones(len(client_rep_mask), dtype=np.float32)

    cfg_default = PipelineConfig()
    scores_default = score_companies_against_icps(
        ec_sorted, fs, profiles, client_latlon, client_weights, cfg_default
    )

    # Weight regime B: zero out all categorical traits except primary_kad and boost primary_kad
    cfg_boost_kad = PipelineConfig(
        trait_weights=tuple(
            (k, 10.0 if k == "primary_kad" else 0.0) for k, _ in DEFAULT_TRAIT_WEIGHTS
        )
    )
    scores_boost_kad = score_companies_against_icps(
        ec_sorted, fs, profiles, client_latlon, client_weights, cfg_boost_kad
    )

    # Weight regime C: zero out primary_kad, boost municipality
    cfg_boost_muni = PipelineConfig(
        trait_weights=tuple(
            (k, 10.0 if k == "municipality_name" else 0.0) for k, _ in DEFAULT_TRAIT_WEIGHTS
        )
    )
    scores_boost_muni = score_companies_against_icps(
        ec_sorted, fs, profiles, client_latlon, client_weights, cfg_boost_muni
    )

    diff_ab = float(np.max(np.abs(scores_default - scores_boost_kad)))
    diff_bc = float(np.max(np.abs(scores_boost_kad - scores_boost_muni)))

    assert diff_ab > 0.05, f"Altering trait_weights failed to meaningfully alter scores: diff={diff_ab:.4f}"
    assert diff_bc > 0.05, f"Differing trait_weights produced identical scores: diff={diff_bc:.4f}"


def test_trait_weights_cache_key_uniqueness():
    """Verify that _compute_engine_cache_key produces distinct digests for different trait weight configurations."""
    sha = "test_portfolio_sha256_hash_12345678"
    cfg_base = PipelineConfig()
    key_base = _compute_engine_cache_key(sha, cfg_base)

    # Perturbation 1: small float change
    cfg_pert1 = PipelineConfig(
        trait_weights=tuple(
            (k, v + 0.01 if k == "primary_kad" else v) for k, v in DEFAULT_TRAIT_WEIGHTS
        )
    )
    key_pert1 = _compute_engine_cache_key(sha, cfg_pert1)
    assert key_base != key_pert1, "Cache key collision on 0.01 trait weight adjustment"

    # Perturbation 2: trait permutation
    reversed_weights = tuple(reversed(DEFAULT_TRAIT_WEIGHTS))
    cfg_pert2 = PipelineConfig(trait_weights=reversed_weights)
    key_pert2 = _compute_engine_cache_key(sha, cfg_pert2)
    assert key_base != key_pert2, "Cache key collision on trait weight order permutation"

    # Perturbation 3: single trait removed
    cfg_pert3 = PipelineConfig(trait_weights=DEFAULT_TRAIT_WEIGHTS[:-1])
    key_pert3 = _compute_engine_cache_key(sha, cfg_pert3)
    assert key_base != key_pert3, "Cache key collision on trait omission"


def test_no_stale_cache_hits_across_trait_weights(tmp_path: Path):
    """Verify load_sha256_cached_result yields None (cache miss) when trait weights differ."""
    proof = StatisticalFeatureProof(
        feature_type="primary_kad",
        feature_value="56110000",
        feature_label_el="KAD",
        icp_count=10,
        icp_share=0.5,
        universe_count=100,
        universe_share=0.01,
        raw_lift=5.0,
        eb_lift=4.5,
        p_value_raw=1e-5,
        p_value_fdr=1e-4,
    )
    profile = DiscoveredICP(
        profile_id="ICP_A",
        cluster_idx=0,
        title_el="ICP A",
        ai_description_el="Description",
        client_count=10,
        client_share=0.5,
        turnover_total=1e6,
        turnover_share=0.5,
        turnover_median=5e4,
        turnover_mean=5e4,
        stability_score=0.9,
        confidence_score=0.9,
        centroid_vector=np.zeros(64, dtype=np.float32),
        spatial_medoid_latlon_deg=(38.0, 23.7),
        significant_features=[proof],
        eb_log_lifts={("primary_kad", "56110000"): 1.5},
    )
    dummy_res = ICPEngineResult(
        optimal_k=1,
        global_bootstrap_ari=0.9,
        silhouette_by_k={1: 0.0},
        profiles=[profile],
        assignments_df=pl.DataFrame({"company_id": [1], "profile_id": ["ICP_A"], "score": [0.9]}),
        lookalikes_df=pl.DataFrame({"company_id": [2], "profile_id": ["ICP_A"], "rank": [1], "score": [0.85]}),
        all_candidates_ranked=[2, 3],
        client_sha256="test_cache_hash_stale_check",
    )

    cfg_a = PipelineConfig(
        cache_dir=tmp_path,
        trait_weights=(("primary_kad", 5.0), ("legal_type", 1.0)),
    )
    save_sha256_cached_result(dummy_res, cfg_a)

    # Attempt to load with different trait weights -> MUST MISS
    cfg_b = PipelineConfig(
        cache_dir=tmp_path,
        trait_weights=(("primary_kad", 1.0), ("legal_type", 5.0)),
    )
    assert load_sha256_cached_result(dummy_res.client_sha256, cfg_b) is None

    # Load with identical weights -> MUST HIT
    loaded_a = load_sha256_cached_result(dummy_res.client_sha256, cfg_a)
    assert loaded_a is not None
    assert loaded_a.client_sha256 == dummy_res.client_sha256

    # Load with equivalent dictionary representation -> MUST HIT
    cfg_a_dict = PipelineConfig(
        cache_dir=tmp_path,
        trait_weights={"primary_kad": 5.0, "legal_type": 1.0},
    )
    loaded_dict = load_sha256_cached_result(dummy_res.client_sha256, cfg_a_dict)
    assert loaded_dict is not None


# ==============================================================================
# SECTION 3: DATACLASS REPLACE RESILIENCE & TYPE PRESERVATION
# ==============================================================================

def test_make_icp_engine_ranker_dataclass_replace_type_preservation():
    """Verify dataclasses.replace preserves Path objects, nested tuples, and fast_cv_mode flags."""
    base_cfg = PipelineConfig()
    custom_data = Path("/opt/archimedes/data")
    custom_cache = Path("/opt/archimedes/cache")
    custom_weights = (("osm_tier", 2.5), ("concept_tier", 1.5))

    ranker = make_icp_engine_ranker(
        fs=None,
        cfg=base_cfg,
        fast_cv_mode=True,
        data_dir=custom_data,
        cache_dir=custom_cache,
        trait_weights=custom_weights,
        kmeans_n_init=25,  # explicit override over fast_cv_mode default of 10
    )

    # Extract eval_cfg from ranker closure
    captured = [
        c.cell_contents
        for c in ranker.__closure__
        if isinstance(getattr(c, "cell_contents", None), PipelineConfig)
    ]
    assert len(captured) == 1, "Expected single PipelineConfig captured in ranker closure"
    eval_cfg = captured[0]

    # Strict type checks
    assert isinstance(eval_cfg.data_dir, Path)
    assert eval_cfg.data_dir == custom_data
    assert isinstance(eval_cfg.cache_dir, Path)
    assert eval_cfg.cache_dir == custom_cache
    assert isinstance(eval_cfg.optimal_weights_path, Path)
    assert isinstance(eval_cfg.cached_universe_features_path, Path)
    assert isinstance(eval_cfg.trait_weights, tuple)
    assert eval_cfg.trait_weights == custom_weights

    # CV parameters
    assert eval_cfg.kmeans_n_init == 25, "Explicit override should take precedence"
    assert eval_cfg.n_bootstrap_runs == 0, "fast_cv_mode should set n_bootstrap_runs=0"
    assert eval_cfg.use_sha256_client_cache is False, "fast_cv_mode should disable disk caching"

    # Immutability of base_cfg
    assert base_cfg.kmeans_n_init == 200
    assert base_cfg.n_bootstrap_runs == 200
    assert base_cfg.use_sha256_client_cache is True
    assert base_cfg.data_dir == Path("data")


def test_make_icp_engine_ranker_invalid_overrides_raises_typeerror():
    """Verify that passing an invalid parameter name to make_icp_engine_ranker fails fast with TypeError."""
    base_cfg = PipelineConfig()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        make_icp_engine_ranker(
            fs=None,
            cfg=base_cfg,
            non_existent_hyperparameter_xyz=999,
        )


def test_make_icp_engine_ranker_fast_cv_false():
    """Verify fast_cv_mode=False retains full precision settings without mutation."""
    base_cfg = PipelineConfig()
    ranker = make_icp_engine_ranker(
        fs=None,
        cfg=base_cfg,
        fast_cv_mode=False,
    )
    eval_cfg = [
        c.cell_contents
        for c in ranker.__closure__
        if isinstance(getattr(c, "cell_contents", None), PipelineConfig)
    ][0]
    assert eval_cfg.kmeans_n_init == 200
    assert eval_cfg.n_bootstrap_runs == 200
    assert eval_cfg.use_sha256_client_cache is True
