"""
tests/test_engine.py
====================
Unit tests for Archimedes2026 Mathematical ICP Engine:
1. Weighted K-Means determinism: identical cluster labels, centroids, and silhouettes given fixed random seed.
2. Silhouette calculation: verify cosine silhouette for K >= 2 against theoretical formula and scikit-learn.
3. K=1 edge cases: homogeneous data, small sample sizes, centroid as weighted mean, silhouette 0.0.
4. Bootstrap ARI stability bounds: verify global ARI and per-cluster stabilities are strictly in [0.0, 1.0],
   and K=1 returns (1.0, {0: 1.0}).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.config import PipelineConfig
from src.icp_engine import (
    compute_bootstrap_stability,
    select_optimal_k_and_cluster,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_multicluster_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 150 points in 16 dimensions consisting of 3 well-separated clusters
    (50 samples each) on the unit hypersphere, with non-uniform sample weights.
    """
    rng = np.random.default_rng(42)
    dim = 16
    n_per_cluster = 50

    # 3 distinct orthogonal directions
    c1 = np.zeros(dim); c1[0] = 1.0; c1[1] = 0.5
    c2 = np.zeros(dim); c2[2] = 1.0; c2[3] = 0.5
    c3 = np.zeros(dim); c3[4] = 1.0; c3[5] = 0.5

    x1 = rng.normal(c1, 0.15, (n_per_cluster, dim))
    x2 = rng.normal(c2, 0.15, (n_per_cluster, dim))
    x3 = rng.normal(c3, 0.15, (n_per_cluster, dim))

    X_raw = np.vstack([x1, x2, x3])
    X = normalize(X_raw, norm="l2", axis=1)

    # Log-normal turnover weights
    weights = np.exp(rng.normal(10.0, 1.0, 3 * n_per_cluster))
    return X, weights


@pytest.fixture
def synthetic_homogeneous_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 80 points tightly concentrated around a single centroid (low variance),
    representing a homogeneous client portfolio where K=1 is optimal.
    """
    rng = np.random.default_rng(101)
    dim = 12
    n_samples = 80
    center = np.ones(dim) / np.sqrt(dim)

    # Tight noise around single center
    x = rng.normal(center, 0.005, (n_samples, dim))
    X = normalize(x, norm="l2", axis=1)
    weights = rng.uniform(5000.0, 50000.0, n_samples)
    return X, weights


# =============================================================================
# 1. WEIGHTED K-MEANS DETERMINISM TESTS
# =============================================================================

@pytest.mark.engine
def test_weighted_kmeans_determinism_identical_runs(
    synthetic_multicluster_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Verify that running select_optimal_k_and_cluster with identical inputs
    and fixed kmeans_random_state produces strictly identical outputs across repeated runs.
    """
    X, weights = synthetic_multicluster_data
    cfg = PipelineConfig(
        min_icps=2,
        max_icps=4,
        kmeans_n_init=10,
        kmeans_random_state=42,
    )

    k1, labels1, centroids1, sil1 = select_optimal_k_and_cluster(X, weights, cfg)
    k2, labels2, centroids2, sil2 = select_optimal_k_and_cluster(X, weights, cfg)

    # 1. Optimal K equality
    assert k1 == k2 == 3, f"Expected K=3, got {k1} and {k2}"

    # 2. Exact label matching
    assert np.array_equal(labels1, labels2), "Cluster labels differed between identical runs"

    # 3. Exact centroid coordinate matching
    assert np.allclose(centroids1, centroids2, atol=1e-7), "Centroids differed between identical runs"

    # 4. Exact silhouette matching
    assert sil1 == sil2, "Silhouette score dictionaries differed between identical runs"


@pytest.mark.engine
def test_weighted_kmeans_centroid_sensitivity_to_sample_weights() -> None:
    """
    Verify that sample_weight meaningfully shifts the cluster centroid
    towards the high-weight samples (verifying weights are not ignored).
    """
    rng = np.random.default_rng(7)
    # 20 samples: 10 near +1.0, 10 near +2.0
    x_low = rng.normal(1.0, 0.05, (10, 4))
    x_high = rng.normal(2.0, 0.05, (10, 4))
    X = normalize(np.vstack([x_low, x_high]), norm="l2", axis=1)

    # Case A: Uniform weights
    w_uniform = np.ones(20)
    # Case B: Heavily weight the second group (1000x)
    w_skewed = np.array([1.0] * 10 + [1000.0] * 10)

    cfg = PipelineConfig(min_icps=1, max_icps=1)
    _, _, cent_uniform, _ = select_optimal_k_and_cluster(X, w_uniform, cfg)
    _, _, cent_skewed, _ = select_optimal_k_and_cluster(X, w_skewed, cfg)

    # Centroid with skewed weights must be closer to the high-weight points than the uniform centroid
    high_group_mean = normalize(x_high.mean(axis=0, keepdims=True), norm="l2")
    dist_uniform = 1.0 - float((cent_uniform @ high_group_mean.T).item())
    dist_skewed = 1.0 - float((cent_skewed @ high_group_mean.T).item())

    assert dist_skewed < dist_uniform, "Sample weights failed to pull centroid toward high-weight samples"


@pytest.mark.engine
def test_weighted_kmeans_k_ge_2_sensitivity_to_sample_weights() -> None:
    """Verify that for K >= 2, sample_weight affects KMeans cluster centroids compared to unweighted KMeans."""
    rng = np.random.default_rng(42)
    dim = 8
    c1 = np.zeros(dim); c1[0] = 1.0
    c2 = np.zeros(dim); c2[1] = 1.0
    x1 = rng.normal(c1, 0.05, (30, dim))
    c2_offset = np.zeros(dim); c2_offset[2] = 1.0
    x2_a = rng.normal(c2, 0.05, (15, dim))
    x2_b = rng.normal(c2_offset, 0.05, (15, dim))
    X = normalize(np.vstack([x1, x2_a, x2_b]), norm="l2", axis=1)
    weights = np.array([1.0] * 30 + [1.0] * 15 + [1000.0] * 15)

    cfg = PipelineConfig(min_icps=2, max_icps=2, kmeans_n_init=10, kmeans_random_state=42)
    _, _, centroids_weighted, _ = select_optimal_k_and_cluster(X, weights, cfg)

    weights_uniform = np.ones(60)
    _, _, centroids_unweighted, _ = select_optimal_k_and_cluster(X, weights_uniform, cfg)

    assert not np.allclose(centroids_weighted, centroids_unweighted, atol=1e-2), (
        "Weighted K-Means centroids for K >= 2 identical to unweighted centroids; sample_weight was ignored!"
    )


# =============================================================================
# 2. SILHOUETTE CALCULATION & THEORETICAL FORMULA TESTS
# =============================================================================

@pytest.mark.engine
def test_silhouette_calculation_matches_sklearn_and_theory() -> None:
    """
    Verify cosine silhouette calculation against both scikit-learn and
    a first-principles manual implementation:
    s(i) = (b(i) - a(i)) / max(a(i), b(i)) with cosine distance D = 1 - u . v.
    Uses 30 points (>= 24 required so max_k >= 2).
    """
    rng = np.random.default_rng(42)
    dim = 4
    n_per = 15  # 30 total samples >= 24 required for max_k >= 2
    c1 = np.array([1.0, 0.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0, 0.0])

    x1 = rng.normal(c1, 0.05, (n_per, dim))
    x2 = rng.normal(c2, 0.05, (n_per, dim))
    X = normalize(np.vstack([x1, x2]), norm="l2", axis=1)
    labels = np.array([0] * n_per + [1] * n_per, dtype=np.int32)
    n = len(X)

    # 1. First-principles theoretical calculation
    # Pairwise cosine distance matrix: D_ij = 1 - (u_i . u_j)
    D = 1.0 - (X @ X.T)

    s_manual = []
    for i in range(n):
        c_i = labels[i]
        # a(i): mean cosine distance to other points in same cluster
        same_cluster_indices = [j for j in range(n) if labels[j] == c_i and j != i]
        a_i = float(np.mean([D[i, j] for j in same_cluster_indices]))

        # b(i): mean cosine distance to points in other cluster
        other_cluster_indices = [j for j in range(n) if labels[j] != c_i]
        b_i = float(np.mean([D[i, j] for j in other_cluster_indices]))

        s_i = (b_i - a_i) / max(a_i, b_i)
        s_manual.append(s_i)

    theoretical_silhouette = float(np.mean(s_manual))

    # 2. Scikit-learn calculation
    sklearn_silhouette = float(silhouette_score(X, labels, metric="cosine"))

    assert theoretical_silhouette == pytest.approx(sklearn_silhouette, abs=1e-6)

    # 3. Engine silhouette round verification
    cfg = PipelineConfig(min_icps=2, max_icps=2, kmeans_n_init=10, kmeans_random_state=42)
    _, _, _, sil_dict = select_optimal_k_and_cluster(X, np.ones(n), cfg)

    assert sil_dict[2] == round(sklearn_silhouette, 4)
    assert sil_dict[1] == 0.0


@pytest.mark.engine
def test_silhouette_for_k_equals_1_convention(
    synthetic_multicluster_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify that silhouette_by_k[1] is always 0.0 by definition."""
    X, weights = synthetic_multicluster_data
    cfg = PipelineConfig(min_icps=1, max_icps=3, kmeans_n_init=10)
    _, _, _, sil_dict = select_optimal_k_and_cluster(X, weights, cfg)

    assert 1 in sil_dict
    assert sil_dict[1] == 0.0


# =============================================================================
# 3. K=1 EDGE CASES & HOMOGENEOUS DATA TESTS
# =============================================================================

@pytest.mark.engine
def test_k1_selected_when_homogeneous_data(
    synthetic_homogeneous_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Verify that when client data is homogeneous and homogeneity_silhouette_threshold is met:
    - best_k == 1 is returned.
    - labels is an array of all zeros.
    - centroid equals the L2-normalized weighted mean.
    - silhouette_by_k[1] == 0.0.
    """
    X, weights = synthetic_homogeneous_data
    cfg = PipelineConfig(
        min_icps=1,
        max_icps=4,
        kmeans_n_init=10,
        homogeneity_silhouette_threshold=0.50,  # Forces K=1 if quality < 0.50
    )

    best_k, labels, centroids, sil_dict = select_optimal_k_and_cluster(X, weights, cfg)

    assert best_k == 1
    assert labels.shape == (X.shape[0],)
    assert np.all(labels == 0)

    # Centroid must strictly match theoretical weighted mean
    expected_centroid = normalize(
        np.average(X, axis=0, weights=weights).reshape(1, -1),
        norm="l2",
        axis=1,
    )
    assert np.allclose(centroids, expected_centroid, atol=1e-6)
    assert sil_dict[1] == 0.0


@pytest.mark.engine
def test_k1_selected_when_small_sample_size() -> None:
    """
    Verify that for small sample sizes (N < 24 samples where n_samples // 12 <= 1),
    max_k collapses to 1 and K=1 is selected automatically without error.
    """
    rng = np.random.default_rng(99)
    X = normalize(rng.normal(0, 1, (15, 8)), norm="l2", axis=1)
    weights = np.ones(15)

    cfg = PipelineConfig(min_icps=1, max_icps=6)
    best_k, labels, centroids, sil_dict = select_optimal_k_and_cluster(X, weights, cfg)

    assert best_k == 1
    assert np.all(labels == 0)
    assert centroids.shape == (1, 8)
    assert sil_dict == {1: 0.0}


@pytest.mark.engine
def test_k1_not_selected_when_min_icps_is_2(
    synthetic_homogeneous_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify that when min_icps=2, K=1 is never selected even if silhouette is low."""
    X, weights = synthetic_homogeneous_data
    cfg = PipelineConfig(min_icps=2, max_icps=4, kmeans_n_init=10)

    best_k, labels, _, _ = select_optimal_k_and_cluster(X, weights, cfg)
    assert best_k >= 2
    assert len(np.unique(labels)) == best_k


# =============================================================================
# 4. BOOTSTRAP ARI STABILITY SCORE BOUNDS TESTS
# =============================================================================

@pytest.mark.engine
def test_bootstrap_stability_k1_contract(
    synthetic_multicluster_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify that when optimal_k == 1, compute_bootstrap_stability returns exactly (1.0, {0: 1.0})."""
    X, weights = synthetic_multicluster_data
    cfg = PipelineConfig(n_bootstrap_runs=20)
    base_labels = np.zeros(X.shape[0], dtype=np.int32)
    base_centroids = normalize(np.average(X, axis=0, weights=weights).reshape(1, -1), norm="l2")

    global_ari, per_cluster_stability = compute_bootstrap_stability(
        X_clients=X,
        sample_weights=weights,
        optimal_k=1,
        base_labels=base_labels,
        base_centroids=base_centroids,
        cfg=cfg,
    )

    assert global_ari == 1.0
    assert per_cluster_stability == {0: 1.0}


@pytest.mark.engine
def test_bootstrap_stability_fast_mode_contract(
    synthetic_multicluster_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify that when n_bootstrap_runs <= 0, stability defaults to 1.0 across all clusters."""
    X, weights = synthetic_multicluster_data
    cfg = PipelineConfig(n_bootstrap_runs=0)
    base_labels = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)
    base_centroids = np.zeros((3, X.shape[1]), dtype=np.float32)

    global_ari, per_cluster_stability = compute_bootstrap_stability(
        X_clients=X,
        sample_weights=weights,
        optimal_k=3,
        base_labels=base_labels,
        base_centroids=base_centroids,
        cfg=cfg,
    )

    assert global_ari == 1.0
    assert per_cluster_stability == {0: 1.0, 1: 1.0, 2: 1.0}


@pytest.mark.engine
def test_bootstrap_stability_bounds_and_structure(
    synthetic_multicluster_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Verify that compute_bootstrap_stability produces scores strictly bounded in [0.0, 1.0],
    and achieves high stability (> 0.7) on well-separated clusters.
    """
    X, weights = synthetic_multicluster_data
    cfg = PipelineConfig(
        min_icps=3,
        max_icps=3,
        kmeans_n_init=10,
        n_bootstrap_runs=25,
        kmeans_random_state=42,
    )

    best_k, labels, centroids, _ = select_optimal_k_and_cluster(X, weights, cfg)
    global_ari, per_cluster_stability = compute_bootstrap_stability(
        X_clients=X,
        sample_weights=weights,
        optimal_k=best_k,
        base_labels=labels,
        base_centroids=centroids,
        cfg=cfg,
    )

    # Global ARI bounds
    assert isinstance(global_ari, float)
    assert 0.0 <= global_ari <= 1.0, f"global_ari {global_ari} out of bounds [0, 1]"
    assert global_ari > 0.70, f"Expected high stability on distinct clusters, got {global_ari}"

    # Per-cluster stability structure and bounds
    assert len(per_cluster_stability) == best_k
    for c in range(best_k):
        stab = per_cluster_stability[c]
        assert isinstance(stab, float)
        assert 0.0 <= stab <= 1.0, f"per_cluster_stability[{c}] = {stab} out of bounds [0, 1]"
        assert stab > 0.70, f"Expected high cluster stability for cluster {c}, got {stab}"


@pytest.mark.engine
def test_bootstrap_stability_noisy_overlap_bounds() -> None:
    """
    Verify that even on noisy or heavily overlapping data, stability scores
    remain valid numbers within [0.0, 1.0] without producing NaNs or exceptions.
    """
    rng = np.random.default_rng(555)
    # Uniform spherical noise (low structure)
    X = normalize(rng.normal(0, 1, (100, 10)), norm="l2", axis=1)
    weights = np.ones(100)

    cfg = PipelineConfig(
        min_icps=2,
        max_icps=2,
        kmeans_n_init=5,
        n_bootstrap_runs=15,
        kmeans_random_state=42,
    )

    best_k, labels, centroids, _ = select_optimal_k_and_cluster(X, weights, cfg)
    global_ari, per_cluster_stability = compute_bootstrap_stability(
        X_clients=X,
        sample_weights=weights,
        optimal_k=best_k,
        base_labels=labels,
        base_centroids=centroids,
        cfg=cfg,
    )

    assert not np.isnan(global_ari)
    assert 0.0 <= global_ari <= 1.0
    for c in range(best_k):
        stab = per_cluster_stability[c]
        assert not np.isnan(stab)
        assert 0.0 <= stab <= 1.0
