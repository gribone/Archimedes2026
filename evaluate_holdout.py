r"""
Stage 2 & 4: Repeated 5-Fold Cross-Validation Hold-Out Optimizer & Benchmark (`evaluate_holdout.py`)
====================================================================================================
Implements the exact objective evaluation protocol from Section 8.1 and Section 5.3 with a
vectorized `<60s` architecture:
1. **Static Pre-Computation Outside All Loops:**
   - Loads `data/cached_universe_features.parquet` (Neural KAD matrix, SVD KAD matrix, static
     company feature blocks, integer-encoded categorical indices, and 3D unit-sphere GPS vectors)
     **once** before any cross-validation loop begins.
2. **Outer Loop over 100 Folds (`RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)`):**
   - For each of the 100 folds (240 Train Clients / 60 Hidden Validation Clients), computes
     fold-specific features **once per fold**:
     * Global 240-client Empirical Bayes categorical log-lifts across all 7 trait dimensions
     * 3D Spherical Haversine distance matrix & Gaussian spatial proximity kernels to the 240 train clients
3. **Inner Loop over Hyperparameter Grid (`alpha` & Feature/Scoring Weights):**
   - Uses pure NumPy matrix multiplication (`U @ centroids.T` and `eb_prob_mat[C_int, :]`) and fast
     deterministic K-Means (`n_init=10`, zero Bootstrap ARI inside CV) so all 20 grid candidates
     across 100 folds (= 2,000 full Universe evaluations) finish in **~15–25 seconds**.
4. **Final Verification & Export (`n_init=200`, `200` Bootstrap Resamples):**
   - Saves the winning hyperparameter configuration to `data/optimal_weights.json` and runs the
     full ICP Engine (`n_init=200`, `n_bootstrap_runs=200`) once at the very end to populate the
     SHA-256 reference cache and `output/` deliverables.

Run default 100-Fold (5x20) evaluation:
    uv run python evaluate_holdout.py

Run full 100-Fold Grid Search (<60s) & save `data/optimal_weights.json`:
    uv run python evaluate_holdout.py --grid-search
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import normalize
import threadpoolctl
from joblib import Parallel, delayed

threadpoolctl.threadpool_limits(limits=1)

from src.config import EMBEDDING_MODELS_REGISTRY, OPTIMAL_WEIGHTS_PATH, PipelineConfig
from src.features import (
    EARTH_RADIUS_KM,
    CompanyFeatureStore,
    build_feature_store,
    recombine_feature_store,
)
from src.icp_engine import (
    _compute_client_weights,
    build_vectorized_categorical_tables,
    compute_eb_log_lifts_vectorized,
    compute_scoring_kernel,
    export_deliverables,
    run_icp_engine,
    score_companies_against_icps,
)
from src.ingest import IngestedDataset, ingest_all
from src.llm_explainer import enrich_profiles_with_local_llm


RankerFn = Callable[[IngestedDataset, pl.DataFrame, int], list[int] | np.ndarray]

_BASELINE_CACHE: dict[int, dict[str, Any]] = {}


def _get_baseline_cache(ds: IngestedDataset) -> dict[str, Any]:
    """Pre-encode `primary_kad` into an integer array once outside the loop for <5ms vectorized baseline evaluation."""
    ec = ds.enriched_companies
    cache_key = id(ec)
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    cids = ec["company_id"].to_numpy()
    vats = ec["vat"].to_list()
    kads = ec["primary_kad"].fill_null("56110000").to_numpy()
    unique_kads, kad_codes = np.unique(kads, return_inverse=True)

    vat_to_rows: dict[str, list[int]] = {}
    cid_to_row: dict[int, int] = {}
    for idx, (v, cid) in enumerate(zip(vats, cids)):
        vat_to_rows.setdefault(v, []).append(idx)
        cid_to_row[int(cid)] = idx

    entry = {
        "n_rows": len(cids),
        "cids": cids,
        "kad_codes": kad_codes.astype(np.int32),
        "n_kads": len(unique_kads),
        "vat_to_rows": vat_to_rows,
        "cid_to_row": cid_to_row,
    }
    _BASELINE_CACHE[cache_key] = entry
    return entry


@dataclass
class HoldoutMetrics:
    """Stores Repeated K-Fold Cross-Validation Recall@500 and Recall@2000 metrics."""

    name: str
    recall_at_500_mean: float
    recall_at_500_std: float
    recall_at_2000_mean: float
    recall_at_2000_std: float
    fold_recalls_500: list[float]
    fold_recalls_2000: list[float]
    expected_recall_500_mean: float | None = None
    expected_recall_2000_mean: float | None = None


def baseline_primary_kad_ranker(
    ds: IngestedDataset,
    train_clients: pl.DataFrame,
    seed: int = 42,
) -> list[int]:
    """
    Official Section 8.1.4 Baseline:
    Rank non-training-client Universe companies that share a Primary KAD with training clients,
    ordered by the frequency of that Primary KAD among training clients.
    """
    bcache = _get_baseline_cache(ds)
    vat_to_rows = bcache["vat_to_rows"]
    kad_codes = bcache["kad_codes"]
    cids_all = bcache["cids"]
    n_rows = bcache["n_rows"]
    n_kads = bcache["n_kads"]

    train_vats = train_clients["vat"].to_list()
    train_row_idx = [r for v in train_vats for r in vat_to_rows.get(v, ())]

    is_train = np.zeros(n_rows, dtype=bool)
    is_train[train_row_idx] = True
    cand_mask = ~is_train

    kad_counts = np.bincount(kad_codes[train_row_idx], minlength=n_kads).astype(np.float64)
    cids = cids_all[cand_mask]
    freqs = kad_counts[kad_codes[cand_mask]]

    rng = np.random.default_rng(seed)
    tie_breaker = rng.uniform(0.0, 0.9999, size=len(cids))

    scores = np.where(freqs > 0, freqs + tie_breaker, tie_breaker - 1.0)
    order = np.argsort(-scores)
    return cids[order].tolist()


def make_icp_engine_ranker(
    fs: CompanyFeatureStore,
    cfg: PipelineConfig,
    fast_cv_mode: bool = True,
    **overrides: Any,
) -> RankerFn:
    """
    Wrap `run_icp_engine` into a pluggable `RankerFn` for Repeated K-Fold CV evaluation.
    In `fast_cv_mode=True`, uses `kmeans_n_init=10`, `n_bootstrap_runs=0`, and `use_sha256_client_cache=False`.
    Additional CV-specific configuration overrides can be applied cleanly via `dataclasses.replace`.
    """
    cv_params: dict[str, Any] = dict(overrides)
    if fast_cv_mode:
        cv_params.setdefault("kmeans_n_init", 10)
        cv_params.setdefault("n_bootstrap_runs", 0)
        cv_params.setdefault("use_sha256_client_cache", False)

    eval_cfg = replace(cfg, **cv_params) if cv_params else cfg

    def _ranker(
        ds: IngestedDataset,
        train_clients: pl.DataFrame,
        seed: int,
    ) -> list[int]:
        res = run_icp_engine(
            ds=ds,
            fs=fs,
            cfg=eval_cfg,
            custom_clients_df=train_clients,
            build_justifications=False,
        )
        return res.all_candidates_ranked

    return _ranker


def compute_analytical_baseline_expected_recall(
    ds: IngestedDataset,
    train_vats: set[str],
    holdout_cids: set[int],
    k: int,
) -> float:
    """Exact mathematical expectation of Recall@k for the Primary KAD Baseline using vectorized `np.bincount`."""
    bcache = _get_baseline_cache(ds)
    vat_to_rows = bcache["vat_to_rows"]
    cid_to_row = bcache["cid_to_row"]
    kad_codes = bcache["kad_codes"]
    n_rows = bcache["n_rows"]
    n_kads = bcache["n_kads"]

    train_row_idx = [r for v in train_vats for r in vat_to_rows.get(v, ())]
    is_train = np.zeros(n_rows, dtype=bool)
    is_train[train_row_idx] = True
    cand_mask = ~is_train

    kad_counts = np.bincount(kad_codes[train_row_idx], minlength=n_kads)
    cand_freqs = kad_counts[kad_codes[cand_mask]]
    max_freq = int(cand_freqs.max()) if len(cand_freqs) > 0 else 0
    if max_freq == 0:
        return 0.0

    tier_totals = np.bincount(cand_freqs, minlength=max_freq + 1)

    holdout_row_idx = [
        cid_to_row[cid]
        for cid in holdout_cids
        if cid in cid_to_row and not is_train[cid_to_row[cid]]
    ]
    holdout_freqs = kad_counts[kad_codes[holdout_row_idx]]
    tier_holdouts = np.bincount(holdout_freqs, minlength=max_freq + 1)

    remaining_k = k
    expected_hits = 0.0

    for freq in range(max_freq, 0, -1):
        n_tier = int(tier_totals[freq])
        if n_tier == 0:
            continue
        h_tier = int(tier_holdouts[freq])
        if remaining_k >= n_tier:
            expected_hits += h_tier
            remaining_k -= n_tier
        else:
            expected_hits += h_tier * (remaining_k / n_tier)
            remaining_k = 0
            break

    return expected_hits / max(len(holdout_cids), 1)


def evaluate_ranker_on_holdout(
    ds: IngestedDataset,
    ranker_fn: RankerFn,
    name: str = "Baseline (Primary KAD Frequency)",
    n_splits: int = 5,
    n_repeats: int = 20,
    holdout_fraction: float = 0.20,
    base_seed: int = 42,
    is_baseline: bool = False,
) -> HoldoutMetrics:
    """
    Evaluate `ranker_fn` using `RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)`
    when `n_repeats > 1` (default 5x20 = 100 hold-out evaluations), or `KFold` when `n_repeats == 1`.
    """
    clients = ds.clients
    ec = ds.enriched_companies
    all_client_vats = clients["vat"].to_numpy()

    if n_repeats > 1:
        splitter = RepeatedKFold(
            n_splits=max(2, n_splits),
            n_repeats=n_repeats,
            random_state=base_seed,
        )
    else:
        splitter = KFold(
            n_splits=max(2, n_splits),
            shuffle=True,
            random_state=base_seed,
        )

    vat_to_cids: dict[str, list[int]] = {}
    for v_val, cid_val in zip(ec["vat"].to_list(), ec["company_id"].to_list()):
        vat_to_cids.setdefault(v_val, []).append(int(cid_val))

    r500_list: list[float] = []
    r2000_list: list[float] = []
    exp500_list: list[float] = []
    exp2000_list: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(all_client_vats)):
        seed = base_seed + fold_idx
        train_vats = set(all_client_vats[train_idx].tolist())
        holdout_vats = set(all_client_vats[val_idx].tolist())

        train_clients_df = clients.filter(pl.col("vat").is_in(train_vats))

        holdout_cids = {cid for v in holdout_vats for cid in vat_to_cids.get(v, ())}
        train_cids = {cid for v in train_vats for cid in vat_to_cids.get(v, ())}

        ranked_cids = ranker_fn(ds, train_clients_df, seed)

        top_2000 = ranked_cids[:2000]
        assert len(top_2000) >= 2000, "Lookalike ranking must produce at least 2,000 rows"
        leaked = set(top_2000) & train_cids
        assert not leaked, f"CRITICAL TRAP VIOLATION: {len(leaked)} training client IDs leaked into lookalikes!"

        top_500_set = set(ranked_cids[:500])
        top_2000_set = set(top_2000)

        r500 = len(top_500_set & holdout_cids) / len(holdout_cids)
        r2000 = len(top_2000_set & holdout_cids) / len(holdout_cids)
        r500_list.append(r500)
        r2000_list.append(r2000)

        if is_baseline:
            exp500_list.append(
                compute_analytical_baseline_expected_recall(ds, train_vats, holdout_cids, 500)
            )
            exp2000_list.append(
                compute_analytical_baseline_expected_recall(ds, train_vats, holdout_cids, 2000)
            )

    return HoldoutMetrics(
        name=name,
        recall_at_500_mean=float(np.mean(r500_list)),
        recall_at_500_std=float(np.std(r500_list)),
        recall_at_2000_mean=float(np.mean(r2000_list)),
        recall_at_2000_std=float(np.std(r2000_list)),
        fold_recalls_500=[round(x, 4) for x in r500_list],
        fold_recalls_2000=[round(x, 4) for x in r2000_list],
        expected_recall_500_mean=float(np.mean(exp500_list)) if is_baseline else None,
        expected_recall_2000_mean=float(np.mean(exp2000_list)) if is_baseline else None,
    )


def print_holdout_report(
    metrics_list: list[HoldoutMetrics],
    n_splits: int = 5,
    n_repeats: int = 20,
) -> None:
    """Pretty-print Repeated K-Fold Cross-Validation Hold-Out benchmark results."""
    total_evals = n_splits * max(1, n_repeats)
    print("=" * 84)
    print(
        f"REPEATED K-FOLD HOLD-OUT BENCHMARK (n_splits={n_splits}, n_repeats={n_repeats} -> {total_evals} Evaluations)"
    )
    print("=" * 84)
    base_m = metrics_list[0] if metrics_list else None
    for idx, m in enumerate(metrics_list):
        print(f"Model [{idx+1}]: {m.name}")
        print(
            f"  - Recall@500  ({total_evals} folds) : {100*m.recall_at_500_mean:6.2f}% ± {100*m.recall_at_500_std:4.2f}% "
            f"(~{m.recall_at_500_mean*60:.1f} / 60 hidden clients)"
        )
        print(
            f"  - Recall@2000 ({total_evals} folds) : {100*m.recall_at_2000_mean:6.2f}% ± {100*m.recall_at_2000_std:4.2f}% "
            f"(~{m.recall_at_2000_mean*60:.1f} / 60 hidden clients)"
        )
        if m.expected_recall_500_mean is not None and m.expected_recall_2000_mean is not None:
            print(
                f"  - Analytical E[Recall@k]   : E[R@500] = {100*m.expected_recall_500_mean:6.2f}% | "
                f"E[R@2000] = {100*m.expected_recall_2000_mean:6.2f}%"
            )
        elif base_m is not None and idx > 0:
            denom_500 = base_m.expected_recall_500_mean or max(base_m.recall_at_500_mean, 1e-6)
            denom_2000 = base_m.expected_recall_2000_mean or max(base_m.recall_at_2000_mean, 1e-6)
            gain_500 = m.recall_at_500_mean / denom_500
            gain_2000 = m.recall_at_2000_mean / denom_2000
            diff_500 = 100 * (m.recall_at_500_mean - base_m.recall_at_500_mean)
            diff_2000 = 100 * (m.recall_at_2000_mean - base_m.recall_at_2000_mean)
            print(
                f"  - Lift vs Baseline         : Recall@500 = {gain_500:.2f}x ({diff_500:+.2f} pp) | "
                f"Recall@2000 = {gain_2000:.2f}x ({diff_2000:+.2f} pp)"
            )
        print("-" * 84)


# ==============================================================================
# VECTORIZED <60s 100-FOLD GRID SEARCH ENGINE (Folds Outer, Grid Inner)
# ==============================================================================
# Re-exported from src.icp_engine (Features 11 & 13)
_build_vectorized_categorical_tables = build_vectorized_categorical_tables
_compute_eb_log_lifts_vectorized = compute_eb_log_lifts_vectorized


def _evaluate_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_client_vats: np.ndarray,
    vat_to_rep_row: dict[str, int],
    vat_to_all_rows: dict[str, list[int]],
    vats_list: list[str],
    vat_to_turnover: dict[str, float],
    base_cfg: PipelineConfig,
    n_companies: int,
    cids_arr: np.ndarray,
    valid_comp_idx: np.ndarray,
    all_300_valid_gps: np.ndarray,
    rep_row_to_col300: dict[int, int],
    kern_by_bw: dict[float, np.ndarray],
    unique_bws: list[float],
    unique_priors: list[float],
    U_by_candidate: list[np.ndarray],
    cat_tables: dict[str, Any],
    grid_candidates: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Evaluate all grid candidates on a single RepeatedKFold fold."""
    train_vats = set(all_client_vats[train_idx].tolist())
    holdout_vats = set(all_client_vats[val_idx].tolist())

    # Canonical train representative row indices (sorted) & non-client mask
    client_rep_indices = np.array(
        sorted(vat_to_rep_row[v] for v in train_vats if v in vat_to_rep_row),
        dtype=np.int32,
    )
    train_all_rows = [r for v in train_vats for r in vat_to_all_rows.get(v, ())]
    is_train_vat_mask = np.zeros(n_companies, dtype=bool)
    is_train_vat_mask[train_all_rows] = True
    non_client_indices = np.where(~is_train_vat_mask)[0]

    holdout_cids = {
        int(cids_arr[r]) for v in holdout_vats for r in vat_to_all_rows.get(v, ())
    }
    n_holdout = max(len(holdout_cids), 1)

    client_turnovers = np.array(
        [vat_to_turnover[vats_list[idx]] for idx in client_rep_indices],
        dtype=np.float64,
    )
    client_weights = _compute_client_weights(client_turnovers, base_cfg)

    # Fold-specific Spatial Kernel vectors (computed by slicing precomputed `kern_by_bw[bw]`)
    train_col300_idx = np.array(
        [rep_row_to_col300[int(r)] for r in client_rep_indices],
        dtype=np.int32,
    )
    valid_client_mask = all_300_valid_gps[train_col300_idx]
    spatial_by_bw: dict[float, np.ndarray] = {}

    if len(valid_comp_idx) > 0 and np.any(valid_client_mask):
        valid_col300 = train_col300_idx[valid_client_mask]
        c_w = client_weights[valid_client_mask].astype(np.float32)
        c_w_sum = float(np.sum(c_w)) or 1.0
        for bw in unique_bws:
            sp_vec = np.full(n_companies, 0.5, dtype=np.float32)
            kern = (kern_by_bw[bw][:, valid_col300] @ c_w) / c_w_sum
            p95 = float(np.percentile(kern, 95)) or 1.0
            sp_vec[valid_comp_idx] = np.clip(kern / p95, 0.0, 1.0).astype(np.float32)
            spatial_by_bw[bw] = sp_vec[non_client_indices]
    else:
        for bw in unique_bws:
            spatial_by_bw[bw] = np.full(len(non_client_indices), 0.5, dtype=np.float32)

    # Pre-slice candidate matrices and categorical codes ONCE per fold
    U_cand_list = [U_mat[non_client_indices] for U_mat in U_by_candidate]
    cids_cand = cids_arr[non_client_indices]
    is_holdout_cand = np.isin(cids_cand, list(holdout_cids))
    codes_cand = [c[non_client_indices] for c in cat_tables["codes_list"]]

    # Fold-specific Global 240-client EB log-lifts & pre-summed candidate scores ONCE per (fold, m_prior)
    glob_score_cand_by_prior: dict[float, np.ndarray] = {}
    for pr in unique_priors:
        g_ll = _compute_eb_log_lifts_vectorized(cat_tables, client_rep_indices, pr)
        glob_score_cand = np.zeros(len(non_client_indices), dtype=np.float32)
        for d_idx, w_trait in enumerate(cat_tables["weights_list"]):
            glob_vec = (0.45 * float(w_trait)) * g_ll[d_idx]
            glob_score_cand += glob_vec[codes_cand[d_idx]]
        glob_score_cand_by_prior[pr] = glob_score_cand

    fold_r500: list[float] = []
    fold_r2000: list[float] = []

    # 3. INNER LOOP over the hyperparameter grid using pure NumPy + fast KMeans (n_init=3)
    for cand_idx, cand in enumerate(grid_candidates):
        wp = cand["wp"]
        U_mat = U_by_candidate[cand_idx]
        X_clients = U_mat[client_rep_indices]

        # Precompute 240x240 cosine distance matrix once per candidate for instant silhouette_score
        dist_mat = np.clip(1.0 - (X_clients @ X_clients.T), 0.0, 2.0)
        np.fill_diagonal(dist_mat, 0.0)

        # Fast deterministic K selection in K=2..4 with n_init=3
        best_k = 2
        best_qual = -1.0
        best_labels = None
        best_centroids = None

        for k in (2, 3, 4):
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=2,
                max_iter=30,
                random_state=42,
            )
            lbls = km.fit_predict(X_clients, sample_weight=client_weights)
            cents = normalize(km.cluster_centers_, norm="l2", axis=1).astype(np.float32)
            counts = np.bincount(lbls, minlength=k)
            min_sz = int(counts.min())
            sil = float(silhouette_score(dist_mat, lbls, metric="precomputed"))
            pen = 0.0 if min_sz >= 12 else 0.05 * (12 - min_sz) / 12.0
            qual = sil - pen
            if qual > best_qual:
                best_qual = qual
                best_k = k
                best_labels = lbls
                best_centroids = cents

        assert best_labels is not None and best_centroids is not None

        # Vectorized Cosine Similarity on non-client candidates: (N_cand, K)
        U_cand = U_cand_list[cand_idx]
        cos_sim_mat = np.clip((U_cand @ best_centroids.T + 1.0) * 0.5, 0.0, 1.0)

        # Vectorized Empirical Bayes Log-Lift Affinity on non-client candidates: (N_cand, K)
        m_prior = float(wp["eb_prior"])
        glob_score_cand = glob_score_cand_by_prior[m_prior]
        eb_raw_mat = np.empty((len(non_client_indices), best_k), dtype=np.float32)

        for c_idx in range(best_k):
            c_rows = client_rep_indices[best_labels == c_idx]
            clust_ll = _compute_eb_log_lifts_vectorized(cat_tables, c_rows, m_prior)
            score_c = glob_score_cand.copy()
            for d_idx, w_trait in enumerate(cat_tables["weights_list"]):
                clust_vec = (0.55 * float(w_trait)) * clust_ll[d_idx]
                score_c += clust_vec[codes_cand[d_idx]]
            eb_raw_mat[:, c_idx] = score_c

        # Per-ICP spatial affinity (blending 65% per-cluster client density with 35% global client density)
        bw_val = float(wp["bw_km"])
        sp_global_cand = spatial_by_bw[bw_val]
        if len(valid_comp_idx) > 0 and np.any(valid_client_mask):
            spatial_sim_mat = np.empty((len(non_client_indices), best_k), dtype=np.float32)
            for c_idx in range(best_k):
                mask_c = (best_labels[valid_client_mask] == c_idx)
                if np.sum(mask_c) >= 2:
                    col_c = valid_col300[mask_c]
                    w_c = c_w[mask_c]
                    w_c_sum = float(np.sum(w_c)) or 1.0
                    kern_c = (kern_by_bw[bw_val][:, col_c] @ w_c) / w_c_sum
                    p95_c = float(np.percentile(kern_c, 95)) or 1.0
                    sp_vec_c = np.full(n_companies, 0.5, dtype=np.float32)
                    sp_vec_c[valid_comp_idx] = np.clip(kern_c / p95_c, 0.0, 1.0).astype(np.float32)
                    spatial_sim_mat[:, c_idx] = 0.65 * sp_vec_c[non_client_indices] + 0.35 * sp_global_cand
                else:
                    spatial_sim_mat[:, c_idx] = sp_global_cand
        else:
            spatial_sim_mat = sp_global_cand

        final_scores = compute_scoring_kernel(
            cos_sim_mat=cos_sim_mat,
            eb_raw_mat=eb_raw_mat,
            spatial_sim=spatial_sim_mat,
            w_cos=float(wp["w_cos"]),
            w_eb=float(wp["w_eb"]),
            w_sp=float(wp["w_sp"]),
        )
        best_cand_scores = np.max(final_scores, axis=1)

        # Use argpartition for O(N) top-2000 & top-500 extraction
        top2000_local = np.argpartition(-best_cand_scores, 2000)[:2000]
        top2000_sorted = top2000_local[np.argsort(-best_cand_scores[top2000_local])]
        top500_sorted = top2000_sorted[:500]

        fold_r500.append(int(np.sum(is_holdout_cand[top500_sorted])) / n_holdout)
        fold_r2000.append(int(np.sum(is_holdout_cand[top2000_sorted])) / n_holdout)

    return fold_r500, fold_r2000


def run_repeated_kfold_grid_search(
    ds: IngestedDataset,
    embedding_model: str = "granite-107m",
    n_splits: int = 5,
    n_repeats: int = 20,
    save_path: Path = OPTIMAL_WEIGHTS_PATH,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    r"""
    Vectorized `<60s` Grid Search over `RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)` (100 folds):
      1. Pre-computes all unified feature matrices `U_grid[(alpha, bw_profile)]` and categorical tables ONCE outside all loops.
      2. Outer loop iterates over the 100 folds, computing fold-specific client distances (`dist_km`) and
         global 240-client Empirical Bayes log-lifts ONCE per fold.
      3. Inner loop iterates over the 20 hyperparameter combinations using pure NumPy matrix multiplication
         and fast `n_init=10` (zero Bootstrap ARI inside CV).
      4. Runs `n_init=200` and `n_bootstrap_runs=200` once at the very end using the winning weights!
    """
    t_start_all = time.perf_counter()
    total_evals = n_splits * n_repeats
    print("=" * 88)
    print(
        f"VECTORIZED REPEATED K-FOLD GRID SEARCH (Model={embedding_model}, {n_splits}x{n_repeats}={total_evals} folds)"
    )
    print("=" * 88)

    # 1. Pre-compute and cache static feature blocks ONCE outside all loops
    base_cfg = PipelineConfig(embedding_model=embedding_model, hybrid_alpha=0.50)
    base_fs = build_feature_store(ds, base_cfg)

    ec_base = ds.enriched_companies.sort("company_id")
    n_companies = ec_base.height
    cids_arr = ec_base["company_id"].to_numpy()
    vats_list = ec_base["vat"].to_list()
    is_branch_arr = ec_base["is_branch"].to_numpy()

    cat_tables = _build_vectorized_categorical_tables(ec_base, base_cfg.trait_weights)

    # Pre-compute 3D unit-sphere coordinates for all valid GPS companies once
    valid_comp_idx = np.where(base_fs.has_gps)[0]
    u_lat = base_fs.latlon_rad[valid_comp_idx, 0]
    u_lon = base_fs.latlon_rad[valid_comp_idx, 1]
    u_xyz = np.column_stack(
        [np.cos(u_lat) * np.cos(u_lon), np.cos(u_lat) * np.sin(u_lon), np.sin(u_lat)]
    ).astype(np.float32)

    # Map VAT -> canonical representative row index and all company_id row indices
    vat_to_all_rows: dict[str, list[int]] = {}
    vat_to_rep_row: dict[str, int] = {}
    for row_idx, (v_str, is_b) in enumerate(zip(vats_list, is_branch_arr)):
        vat_to_all_rows.setdefault(v_str, []).append(row_idx)
        if v_str not in vat_to_rep_row or is_b < is_branch_arr[vat_to_rep_row[v_str]]:
            vat_to_rep_row[v_str] = row_idx

    clients_df = ds.clients
    all_client_vats = clients_df["vat"].to_numpy()
    vat_to_turnover = dict(
        zip(clients_df["vat"].to_list(), clients_df["turnover"].to_list())
    )

    # Define hyperparameter grid
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0] if embedding_model != "char-ngram-svd" else [0.0]
    weight_profiles = [
        {
            "profile_name": "Balanced Hybrid",
            "w_cos": 0.45,
            "w_eb": 0.40,
            "w_sp": 0.15,
            "bw_sem": 1.00,
            "bw_spat": 1.15,
            "bw_leg": 0.90,
            "eb_prior": 10.0,
            "bw_km": 25.0,
        },
        {
            "profile_name": "EB-Lift + Spatial Focus",
            "w_cos": 0.35,
            "w_eb": 0.50,
            "w_sp": 0.15,
            "bw_sem": 1.00,
            "bw_spat": 1.25,
            "bw_leg": 1.00,
            "eb_prior": 8.0,
            "bw_km": 20.0,
        },
        {
            "profile_name": "High-Precision EB + Tight Spatial",
            "w_cos": 0.25,
            "w_eb": 0.55,
            "w_sp": 0.20,
            "bw_sem": 0.95,
            "bw_spat": 1.30,
            "bw_leg": 1.05,
            "eb_prior": 6.0,
            "bw_km": 15.0,
        },
        {
            "profile_name": "Semantic Centroid Dominant",
            "w_cos": 0.60,
            "w_eb": 0.30,
            "w_sp": 0.10,
            "bw_sem": 1.30,
            "bw_spat": 1.00,
            "bw_leg": 0.85,
            "eb_prior": 12.0,
            "bw_km": 25.0,
        },
    ]

    grid_candidates: list[dict[str, Any]] = []
    U_by_candidate: list[np.ndarray] = []
    unique_priors = sorted({float(wp["eb_prior"]) for wp in weight_profiles})
    unique_bws = sorted({float(wp["bw_km"]) for wp in weight_profiles})

    for alpha in alphas:
        for wp in weight_profiles:
            cfg_trial = PipelineConfig(
                embedding_model=embedding_model,
                hybrid_alpha=alpha,
                block_weight_semantic=wp["bw_sem"],
                block_weight_spatial=wp["bw_spat"],
                block_weight_legal=wp["bw_leg"],
                sim_weight_feature_cosine=wp["w_cos"],
                sim_weight_eb_lift_affinity=wp["w_eb"],
                sim_weight_spatial_kernel=wp["w_sp"],
                eb_prior_strength=wp["eb_prior"],
                spatial_kernel_bandwidth_km=wp["bw_km"],
            )
            fs_trial = recombine_feature_store(base_fs, alpha, cfg_trial)
            U_by_candidate.append(fs_trial.unified_features)
            grid_candidates.append(
                {
                    "alpha": alpha,
                    "wp": wp,
                    "r500_folds": [],
                    "r2000_folds": [],
                }
            )

    # Pre-compute the 24,424 x 300 Spatial Distance & Gaussian Kernel Matrices ONCE outside the 100-fold loop
    all_300_rep_indices = np.array(
        sorted(vat_to_rep_row[v] for v in all_client_vats if v in vat_to_rep_row),
        dtype=np.int32,
    )
    rep_row_to_col300 = {int(r): col_i for col_i, r in enumerate(all_300_rep_indices)}
    all_300_latlon = base_fs.latlon_rad[all_300_rep_indices]
    all_300_valid_gps = ~np.isnan(all_300_latlon[:, 0]) & ~np.isnan(all_300_latlon[:, 1])

    all_c_lat = np.where(all_300_valid_gps, all_300_latlon[:, 0], 0.0)
    all_c_lon = np.where(all_300_valid_gps, all_300_latlon[:, 1], 0.0)
    all_clients_xyz = np.column_stack(
        [np.cos(all_c_lat) * np.cos(all_c_lon), np.cos(all_c_lat) * np.sin(all_c_lon), np.sin(all_c_lat)]
    ).astype(np.float32)

    dist_all_300 = EARTH_RADIUS_KM * np.arccos(
        np.clip(u_xyz @ all_clients_xyz.T, -1.0, 1.0)
    )
    kern_by_bw: dict[float, np.ndarray] = {
        bw: np.exp(-0.5 * (dist_all_300 / bw) ** 2).astype(np.float32)
        for bw in unique_bws
    }

    # Evaluate Baseline across the exact same 100 folds
    base_m = evaluate_ranker_on_holdout(
        ds=ds,
        ranker_fn=baseline_primary_kad_ranker,
        name="ForceCRM Official Baseline (Primary KAD Frequency)",
        n_splits=n_splits,
        n_repeats=n_repeats,
        base_seed=42,
        is_baseline=True,
    )
    print(
        f"  [Pre-computation Done in {time.perf_counter() - t_start_all:.2f}s | Baseline R@500={100*base_m.recall_at_500_mean:.2f}%, R@2000={100*base_m.recall_at_2000_mean:.2f}%]"
    )
    print("-" * 88)

    # 2. OUTER LOOP over the RepeatedKFold splits (Parallelized across CPU cores)
    splitter = RepeatedKFold(
        n_splits=max(2, n_splits),
        n_repeats=n_repeats,
        random_state=42,
    )
    fold_splits = list(splitter.split(all_client_vats))
    fold_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_evaluate_single_fold)(
            fold_idx=f_idx,
            train_idx=tr_idx,
            val_idx=va_idx,
            all_client_vats=all_client_vats,
            vat_to_rep_row=vat_to_rep_row,
            vat_to_all_rows=vat_to_all_rows,
            vats_list=vats_list,
            vat_to_turnover=vat_to_turnover,
            base_cfg=base_cfg,
            n_companies=n_companies,
            cids_arr=cids_arr,
            valid_comp_idx=valid_comp_idx,
            all_300_valid_gps=all_300_valid_gps,
            rep_row_to_col300=rep_row_to_col300,
            kern_by_bw=kern_by_bw,
            unique_bws=unique_bws,
            unique_priors=unique_priors,
            U_by_candidate=U_by_candidate,
            cat_tables=cat_tables,
            grid_candidates=grid_candidates,
        )
        for f_idx, (tr_idx, va_idx) in enumerate(fold_splits)
    )

    for fold_r500, fold_r2000 in fold_results:
        for cand_idx, (r500, r2000) in enumerate(zip(fold_r500, fold_r2000)):
            grid_candidates[cand_idx]["r500_folds"].append(r500)
            grid_candidates[cand_idx]["r2000_folds"].append(r2000)

    # Summarize all grid candidates across the 100 folds
    results: list[dict[str, object]] = []
    for cand in grid_candidates:
        wp = cand["wp"]
        alpha = float(cand["alpha"])
        r500_arr = np.array(cand["r500_folds"], dtype=np.float64)
        r2000_arr = np.array(cand["r2000_folds"], dtype=np.float64)
        row = {
            "embedding_model": embedding_model,
            "hybrid_alpha": alpha,
            "alpha": alpha,
            "profile_name": wp["profile_name"],
            "block_weight_semantic": wp["bw_sem"],
            "block_weight_kad_hierarchy": 0.85,
            "block_weight_spatial": wp["bw_spat"],
            "block_weight_legal": wp["bw_leg"],
            "block_weight_age_branch": 0.45,
            "sim_weight_feature_cosine": wp["w_cos"],
            "sim_weight_eb_lift_affinity": wp["w_eb"],
            "sim_weight_spatial_kernel": wp["w_sp"],
            "eb_prior_strength": wp["eb_prior"],
            "spatial_kernel_bandwidth_km": wp["bw_km"],
            "kmeans_n_init": 200,
            "n_bootstrap_runs": 200,
            "cv_n_splits": n_splits,
            "cv_n_repeats": n_repeats,
            "cv_total_evaluations": total_evals,
            "r500_mean": round(float(np.mean(r500_arr)), 5),
            "r500_std": round(float(np.std(r500_arr)), 5),
            "r2000_mean": round(float(np.mean(r2000_arr)), 5),
            "r2000_std": round(float(np.std(r2000_arr)), 5),
            "baseline_r500_analytical": round(base_m.expected_recall_500_mean or base_m.recall_at_500_mean, 5),
            "baseline_r2000_analytical": round(base_m.expected_recall_2000_mean or base_m.recall_at_2000_mean, 5),
        }
        results.append(row)
        print(
            f"  [alpha={alpha:4.2f} | {wp['profile_name']:<33}] -> "
            f"100-Fold Recall@500 = {100*row['r500_mean']:5.2f}% ± {100*row['r500_std']:4.2f}% | "
            f"Recall@2000 = {100*row['r2000_mean']:5.2f}% ± {100*row['r2000_std']:4.2f}%"
        )

    results.sort(key=lambda r: (r["r500_mean"], r["r2000_mean"]), reverse=True)
    best = results[0]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    grid_elapsed = time.perf_counter() - t_start_all
    print("=" * 88)
    print(
        f"100-FOLD GRID SEARCH COMPLETED IN {grid_elapsed:.2f}s (<60s Target)!\n"
        f"SAVED WINNING HYPERPARAMETERS TO {save_path}:\n"
        f"  Profile: {best['profile_name']} | model={best['embedding_model']} | alpha={best['hybrid_alpha']:.2f}\n"
        f"  Weights: w_cos={best['sim_weight_feature_cosine']}, w_eb={best['sim_weight_eb_lift_affinity']}, "
        f"w_sp={best['sim_weight_spatial_kernel']}, bw_sem={best['block_weight_semantic']}, "
        f"bw_spat={best['block_weight_spatial']}, bw_leg={best['block_weight_legal']}\n"
        f"  100-Fold Recall@500  = {100*best['r500_mean']:.2f}% ± {100*best['r500_std']:.2f}%\n"
        f"  100-Fold Recall@2000 = {100*best['r2000_mean']:.2f}% ± {100*best['r2000_std']:.2f}%"
    )
    print("-" * 88)

    # 4. Run n_init=200 and 200-resample Bootstrap ARI ONCE at the very end using the winning weights
    t_final = time.perf_counter()
    final_cfg = PipelineConfig.from_optimal_weights(
        kmeans_n_init=200,
        n_bootstrap_runs=200,
        use_sha256_client_cache=True,
    )
    final_fs = recombine_feature_store(base_fs, float(best["hybrid_alpha"]), final_cfg)
    # Force fresh computation with n_init=200 and 200 bootstrap runs and save SHA-256 cache
    final_cfg_no_read = PipelineConfig.from_optimal_weights(
        kmeans_n_init=200,
        n_bootstrap_runs=200,
        use_sha256_client_cache=False,
    )
    final_res = run_icp_engine(ds, final_fs, final_cfg_no_read)
    enrich_profiles_with_local_llm(final_res, final_cfg, force_deterministic=True)
    export_deliverables(final_res, final_cfg.output_dir)
    print(
        f"FINAL ICP ENGINE (n_init=200, n_bootstrap_runs=200) & SHA-256 CACHE LOCKED IN {time.perf_counter() - t_final:.2f}s:\n"
        f"  Optimal K={final_res.optimal_k} | Global Bootstrap ARI={100*final_res.global_bootstrap_ari:.1f}% | "
        f"Client SHA-256={final_res.client_sha256[:16]}..."
    )
    print("=" * 88)
    return best, results


def compare_embedding_models_section_5_3(
    ds: IngestedDataset,
    models_to_compare: list[str],
    best_alpha: float = 0.50,
    n_splits: int = 5,
    n_repeats: int = 20,
) -> None:
    """Benchmark multiple Section 5.2 embedding models across the 4 mandatory axes of Section 5.3."""
    total_evals = n_splits * max(1, n_repeats)
    print("\n" + "=" * 92)
    print(f"SECTION 5.3 EMBEDDING MODEL COMPARISON TABLE ({total_evals}-FOLD CV ON CHALLENGE DATA)")
    print("=" * 92)
    print(
        f"{'Model Key':<16} | {'Alpha':<5} | {'Recall@500':<15} | {'Recall@2000':<15} | {'Time (s)':<8} | {'License':<15}"
    )
    print("-" * 92)

    base_m = evaluate_ranker_on_holdout(
        ds=ds,
        ranker_fn=baseline_primary_kad_ranker,
        name="ForceCRM Primary KAD Baseline",
        n_splits=n_splits,
        n_repeats=n_repeats,
        is_baseline=True,
    )
    print(
        f"{'kad-baseline':<16} | {'-':<5} | "
        f"{100*base_m.recall_at_500_mean:5.2f}% ± {100*base_m.recall_at_500_std:4.2f}% | "
        f"{100*base_m.recall_at_2000_mean:5.2f}% ± {100*base_m.recall_at_2000_std:4.2f}% | "
        f"{'0.01s':<8} | {'Built-in':<15}"
    )

    for m_key in models_to_compare:
        alpha_val = 0.0 if m_key == "char-ngram-svd" else best_alpha
        cfg = PipelineConfig.from_optimal_weights(embedding_model=m_key, hybrid_alpha=alpha_val)
        t0 = time.perf_counter()
        fs = build_feature_store(ds, cfg)
        elapsed = time.perf_counter() - t0
        m = evaluate_ranker_on_holdout(
            ds=ds,
            ranker_fn=make_icp_engine_ranker(fs, cfg, fast_cv_mode=True),
            name=f"{m_key} (alpha={alpha_val})",
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
        lic = EMBEDDING_MODELS_REGISTRY.get(m_key, {}).get("license", "Custom")[:15]
        print(
            f"{m_key:<16} | {alpha_val:<5.2f} | "
            f"{100*m.recall_at_500_mean:5.2f}% ± {100*m.recall_at_500_std:4.2f}% | "
            f"{100*m.recall_at_2000_mean:5.2f}% ± {100*m.recall_at_2000_std:4.2f}% | "
            f"{elapsed:6.2f}s  | {lic:<15}"
        )
    print("=" * 92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorized <60s Repeated 5-Fold (5x20=100) Cross-Validation Hold-Out Optimizer."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="granite-107m",
        help="Embedding model key ('granite-107m', 'e5-base', 'e5-small', 'char-ngram-svd')",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Hybrid mixing alpha in [0, 1]")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable neural embedding models and run purely on algebraic character n-gram + PPMI SVD (Section 5.2 Baseline)",
    )
    parser.add_argument("--n-splits", "--splits", dest="splits", type=int, default=5, help="K-Fold splits (default: 5)")
    parser.add_argument("--n-repeats", "--repeats", dest="repeats", type=int, default=20, help="K-Fold repeats (default: 20)")
    parser.add_argument("--grid-search", action="store_true", help="Run <60s RepeatedKFold Grid Search and save data/optimal_weights.json")
    parser.add_argument(
        "--compare-models",
        nargs="*",
        default=None,
        help="Compare multiple embedding models for Section 5.3",
    )
    args = parser.parse_args()

    dataset = ingest_all()

    if args.grid_search:
        best_cfg, _ = run_repeated_kfold_grid_search(
            dataset,
            embedding_model=args.embedding_model,
            n_splits=args.splits,
            n_repeats=args.repeats,
        )
        if args.compare_models is not None:
            models = args.compare_models or ["char-ngram-svd", "granite-107m", "e5-base"]
            compare_embedding_models_section_5_3(
                dataset,
                models_to_compare=models,
                best_alpha=float(best_cfg["hybrid_alpha"]),
                n_splits=args.splits,
                n_repeats=args.repeats,
            )
    elif args.compare_models is not None:
        models = args.compare_models or ["char-ngram-svd", "granite-107m", "e5-base"]
        compare_embedding_models_section_5_3(
            dataset,
            models_to_compare=models,
            best_alpha=args.alpha if args.alpha is not None else 0.50,
            n_splits=args.splits,
            n_repeats=args.repeats,
        )
    else:
        overrides = {"holdout_n_splits": args.splits, "holdout_n_repeats": args.repeats}
        if args.no_embeddings:
            overrides["use_embedding_models"] = False
            overrides["embedding_model"] = "char-ngram-svd"
            overrides["hybrid_alpha"] = 0.0
        elif args.embedding_model:
            overrides["embedding_model"] = args.embedding_model
        if args.alpha is not None:
            overrides["hybrid_alpha"] = args.alpha
        cfg = PipelineConfig.from_optimal_weights(**overrides)
        feature_store = build_feature_store(dataset, cfg)

        baseline_metrics = evaluate_ranker_on_holdout(
            ds=dataset,
            ranker_fn=baseline_primary_kad_ranker,
            name="ForceCRM Official Baseline (Primary KAD Frequency)",
            n_splits=cfg.holdout_n_splits,
            n_repeats=cfg.holdout_n_repeats,
            base_seed=42,
            is_baseline=True,
        )
        icp_metrics = evaluate_ranker_on_holdout(
            ds=dataset,
            ranker_fn=make_icp_engine_ranker(feature_store, cfg, fast_cv_mode=True),
            name=f"Archimedes2026 Hybrid ICP Engine ({cfg.embedding_model}, alpha={cfg.hybrid_alpha:.2f})",
            n_splits=cfg.holdout_n_splits,
            n_repeats=cfg.holdout_n_repeats,
            base_seed=42,
            is_baseline=False,
        )
        print_holdout_report(
            [baseline_metrics, icp_metrics],
            n_splits=cfg.holdout_n_splits,
            n_repeats=cfg.holdout_n_repeats,
        )
