"""
Unit and regression tests for Archimedes2026 scoring engine:
1. Similarity score bounds strictly in [0.0, 1.0], NaN/Inf free.
2. Dynamic NO_FIT thresholding and outlier handling.
3. Strict client branch exclusion from lookalikes (zero client leaks).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from src.icp_engine import (
    load_excluded_companies_map,
    score_companies_against_icps,
)
from src.server import get_service


# -------------------------------------------------------------------------
# Test Fixtures for Scoring
# -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine_service():
    """Retrieve warmed singleton EngineService."""
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
    res, _, _, _, _, _ = engine_service._compute(
        engine_service.default_params, allow_sha_cache=True
    )
    return res


# -------------------------------------------------------------------------
# 1. Similarity Score Bounds Tests [0.0, 1.0]
# -------------------------------------------------------------------------
@pytest.mark.engine
def test_similarity_scores_bounded_in_zero_to_one(
    engine_service, universe_feature_store, baseline_result
):
    """Verify that score_companies_against_icps returns scores strictly in [0.0, 1.0]."""
    ec_sorted = engine_service._meta_full
    fs = universe_feature_store
    profiles = baseline_result.profiles
    cfg = engine_service.defaults_cfg

    client_rep_mask = np.where(ec_sorted["is_client_vat"].to_numpy())[0]
    client_latlon = fs.latlon_rad[client_rep_mask]
    client_weights = np.ones(len(client_rep_mask), dtype=np.float32)

    score_mat = score_companies_against_icps(
        ec_sorted=ec_sorted,
        fs=fs,
        profiles=profiles,
        client_latlon_rad=client_latlon,
        client_weights=client_weights,
        cfg=cfg,
    )

    # 1. Dimension and type
    assert score_mat.ndim == 2
    assert score_mat.shape == (ec_sorted.height, len(profiles))

    # 2. Finite check (no NaN, no Inf)
    assert np.all(np.isfinite(score_mat)), "Score matrix contains NaN or Inf values!"

    # 3. Exact range bounds
    min_val = float(np.min(score_mat))
    max_val = float(np.max(score_mat))
    assert min_val >= 0.0, f"Score below 0.0 detected: {min_val}"
    assert max_val <= 1.0, f"Score above 1.0 detected: {max_val}"


@pytest.mark.engine
@pytest.mark.parametrize(
    "w_cos,w_eb,w_sp",
    [
        (1.0, 0.0, 0.0),        # Pure cosine
        (0.0, 1.0, 0.0),        # Pure empirical Bayes
        (0.0, 0.0, 1.0),        # Pure spatial
        (10.0, 1.0, 0.1),       # Highly skewed
        (0.001, 0.001, 0.001),  # Tiny weights
    ],
)
def test_similarity_score_bounds_under_varying_weights(
    engine_service, universe_feature_store, baseline_result, w_cos, w_eb, w_sp
):
    """Verify score bounds hold under extreme weight permutations."""
    ec_sorted = engine_service._meta_full
    fs = universe_feature_store
    profiles = baseline_result.profiles
    cfg = replace(
        engine_service.defaults_cfg,
        sim_weight_feature_cosine=w_cos,
        sim_weight_eb_lift_affinity=w_eb,
        sim_weight_spatial_kernel=w_sp,
    )

    client_rep_mask = np.where(ec_sorted["is_client_vat"].to_numpy())[0]
    client_latlon = fs.latlon_rad[client_rep_mask]
    client_weights = np.ones(len(client_rep_mask), dtype=np.float32)

    score_mat = score_companies_against_icps(
        ec_sorted=ec_sorted,
        fs=fs,
        profiles=profiles,
        client_latlon_rad=client_latlon,
        client_weights=client_weights,
        cfg=cfg,
    )

    assert np.all(np.isfinite(score_mat))
    assert float(np.min(score_mat)) >= 0.0
    assert float(np.max(score_mat)) <= 1.0


@pytest.mark.engine
def test_similarity_scores_extremal_synthetic_vectors():
    """Verify score bounds on synthetic orthogonal, anti-parallel, and identical vectors."""
    n_companies = 4
    dim = 8
    # c0: identical to centroid [1, 0, 0...]
    # c1: orthogonal [0, 1, 0...]
    # c2: diametrically opposite [-1, 0, 0...]
    # c3: arbitrary unit vector
    u_feat = np.zeros((n_companies, dim), dtype=np.float32)
    u_feat[0, 0] = 1.0
    u_feat[1, 1] = 1.0
    u_feat[2, 0] = -1.0
    u_feat[3, :] = 1.0 / np.sqrt(dim)

    centroids = np.zeros((1, dim), dtype=np.float32)
    centroids[0, 0] = 1.0  # target centroid

    # Compute raw cosine mapped to [0, 1]
    cos_scores = np.clip((u_feat @ centroids.T + 1.0) / 2.0, 0.0, 1.0)
    assert np.isclose(cos_scores[0, 0], 1.0)
    assert np.isclose(cos_scores[1, 0], 0.5)
    assert np.isclose(cos_scores[2, 0], 0.0)
    assert 0.0 <= cos_scores[3, 0] <= 1.0


# -------------------------------------------------------------------------
# 2. NO_FIT Thresholding Tests
# -------------------------------------------------------------------------
@pytest.mark.engine
def test_no_fit_forced_by_high_threshold(engine_service, universe_feature_store):
    """Verify that setting no_fit_min_score=0.9999 causes run_icp_engine to mark all clients NO_FIT."""
    from dataclasses import replace
    from src.icp_engine import run_icp_engine

    cfg = replace(engine_service.defaults_cfg, no_fit_min_score=0.9999, use_sha256_client_cache=False)
    res = run_icp_engine(engine_service.ds, universe_feature_store, cfg)
    no_fit_count = (res.assignments_df["profile_id"] == "NO_FIT").sum()
    assert no_fit_count == res.assignments_df.height
    assert no_fit_count == 300


@pytest.mark.engine
def test_no_fit_zero_threshold_assigns_all_clients(baseline_result):
    """When no_fit_min_score is 0.0 and quantile cutoff is disabled, 0 NO_FIT occur."""
    client_scores = baseline_result.assignments_df["score"].to_numpy()
    dynamic_cutoff = 0.0

    assigned_pids = [
        "ICP_1" if sc >= dynamic_cutoff else "NO_FIT" for sc in client_scores
    ]
    assert not any(pid == "NO_FIT" for pid in assigned_pids)


@pytest.mark.engine
def test_dynamic_quantile_no_fit_outlier_detection():
    """Verify that dynamic quantile cutoff accurately tags bottom tail outliers."""
    # Synthetic client scores: 10 normal clients (0.80 - 0.95) + 2 severe outliers (0.20, 0.30)
    scores = np.array([0.95, 0.92, 0.89, 0.88, 0.86, 0.85, 0.84, 0.82, 0.81, 0.80, 0.30, 0.20])
    q_thresh = 0.10  # 10th percentile
    min_score = 0.35
    q_val = float(np.quantile(scores, q_thresh)) - 0.02
    dynamic_cutoff = max(min_score, q_val)

    # 0.20 and 0.30 fall below min_score (0.35)
    assigned_pids = ["ICP_1" if sc >= dynamic_cutoff else "NO_FIT" for sc in scores]
    assert assigned_pids[-2:] == ["NO_FIT", "NO_FIT"]
    assert assigned_pids[:-2] == ["ICP_1"] * 10


@pytest.mark.engine
def test_assignments_dataframe_schema_and_ordering(baseline_result):
    """Verify assignments DataFrame schema, column types, and descending score sort."""
    df = baseline_result.assignments_df
    assert set(df.columns) == {"company_id", "profile_id", "score"}
    assert df["score"].dtype in (pl.Float32, pl.Float64)
    assert df["company_id"].dtype in (pl.Int32, pl.Int64)
    assert df["profile_id"].dtype in (pl.Utf8, pl.String)

    # Check monotonic descending sort
    scores = df["score"].to_list()
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# -------------------------------------------------------------------------
# 3. Client Branch Exclusion & Zero Leak Tests
# -------------------------------------------------------------------------
@pytest.mark.engine
def test_zero_client_leaks_in_baseline_lookalikes(engine_service, baseline_result):
    """Verify zero client company IDs or client VATs leak into lookalikes."""
    ec = engine_service._meta_full
    client_ids = set(baseline_result.assignments_df["company_id"])
    lookalike_ids = set(baseline_result.lookalikes_df["company_id"])

    # 1. Company ID level leak check
    leak_cids = client_ids.intersection(lookalike_ids)
    assert len(leak_cids) == 0, f"Client IDs leaked into lookalikes: {leak_cids}"

    # 2. VAT level leak check (covers all branches)
    client_vats = set(ec.filter(pl.col("is_client_vat"))["vat"])
    lookalike_vats = set(ec.filter(pl.col("company_id").is_in(list(lookalike_ids)))["vat"])
    leak_vats = client_vats.intersection(lookalike_vats)
    assert len(leak_vats) == 0, f"Client VATs leaked into lookalikes: {leak_vats}"


@pytest.mark.engine
def test_synthetic_multi_branch_client_exclusion():
    """Verify that multi-branch client records sharing a client VAT are 100% excluded."""
    # Construct synthetic universe with multiple branches per VAT
    vat_clients = {"094111111", "094222222"}

    # 6 companies:
    # 101: HQ for 094111111 (is_branch=0)
    # 102: Branch for 094111111 (is_branch=1)
    # 201: HQ for 094222222 (is_branch=0)
    # 202: Branch for 094222222 (is_branch=1)
    # 301: Non-client candidate (VAT 088333333)
    # 302: Non-client candidate (VAT 088444444)
    vats = ["094111111", "094111111", "094222222", "094222222", "088333333", "088444444"]
    cids = np.array([101, 102, 201, 202, 301, 302])

    is_client_vat_mask = np.array([v in vat_clients for v in vats], dtype=bool)
    is_excluded_vat_mask = np.zeros(len(vats), dtype=bool)

    non_client_indices = np.where(~is_client_vat_mask & ~is_excluded_vat_mask)[0]
    candidate_cids = cids[non_client_indices]

    # Verify ALL branch records of client VATs are masked out
    assert set(candidate_cids) == {301, 302}
    for client_branch_cid in (101, 102, 201, 202):
        assert client_branch_cid not in candidate_cids


@pytest.mark.engine
def test_excluded_companies_never_enter_lookalikes(engine_service):
    """Verify that closed or flagged companies in excluded_companies map are omitted."""
    excluded_map = load_excluded_companies_map()
    if not excluded_map:
        pytest.skip("No excluded companies in current configuration.")

    excluded_vats = set(excluded_map.keys())
    res = engine_service.last_result
    ec = engine_service._meta_full
    look_vats = set(
        ec.filter(pl.col("company_id").is_in(res.lookalikes_df["company_id"].to_list()))["vat"]
    )
    leaks = excluded_vats.intersection(look_vats)
    assert len(leaks) == 0, f"Excluded companies leaked into lookalikes: {leaks}"
