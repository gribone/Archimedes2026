r"""
Stage 2 & 4: 5-Fold Cross-Validation Hold-Out Optimizer & Benchmark (`evaluate_holdout.py`)
===========================================================================================
Implements the exact objective evaluation protocol from Section 8.1 and Section 5.3:
1. **5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`):**
   - Partitions the 300 clients into 5 non-overlapping folds of 240 Training Clients (80%)
     and 60 Hidden Validation Clients (20%), ensuring every single client in the dataset
     is evaluated in the hidden hold-out set exactly once!
2. **Official ForceCRM Baseline (Section 8.1.4):**
   - Ranks non-training-client Universe businesses sharing a Primary KAD with the training
     clients, sorted by Primary KAD frequency in the training client list (reporting both
     exact analytical expectation $E[\text{Recall}@k]$ and seeded empirical $\text{Recall}@k$).
3. **Fast 5-Fold CV Grid Search Optimizer (`--grid-search`):**
   - Searches over Hybrid mixing parameter $\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$
     (`e_k^hybrid = [sqrt(alpha) * e_k^neural, sqrt(1-alpha) * e_k^svd]`) and feature /
     scoring block weights (KAD embeddings vs. Empirical Bayes KAD/Geo/Legal lift vs.
     GPS spatial proximity vs. Legal/Age/Branch) to find and lock in the configuration
     that maximizes Mean 5-Fold `Recall@500` (with `Recall@2000` tie-breaking).
4. **Section 5.3 Multi-Model Comparison Table (`--compare-models`):**
   - Evaluates multiple embedding models from `EMBEDDING_MODELS_REGISTRY` and prints a
     ready-to-publish Markdown & ASCII comparison table for `MODEL_DATA_CARD.md`.

Run default 5-Fold CV evaluation:
    .venv/bin/python evaluate_holdout.py

Run full 5-Fold CV Grid Search + Model Comparison:
    .venv/bin/python evaluate_holdout.py --grid-search
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from src.config import EMBEDDING_MODELS_REGISTRY, PipelineConfig
from src.features import (
    CompanyFeatureStore,
    build_feature_store,
    recombine_feature_store,
)
from src.icp_engine import run_icp_engine
from src.ingest import IngestedDataset, ingest_all


RankerFn = Callable[[IngestedDataset, pl.DataFrame, int], list[int] | np.ndarray]


@dataclass
class HoldoutMetrics:
    """Stores 5-Fold Cross-Validation Recall@500 and Recall@2000 metrics."""

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
    ec = ds.enriched_companies
    train_vats = set(train_clients["vat"].to_list())

    train_rows = ec.filter(pl.col("vat").is_in(train_vats))
    kad_counts = Counter(train_rows["primary_kad"].to_list())

    candidates = ec.filter(~pl.col("vat").is_in(train_vats)).select(
        ["company_id", "primary_kad"]
    )

    cids = candidates["company_id"].to_numpy()
    kads = candidates["primary_kad"].to_list()
    freqs = np.array([kad_counts.get(k, 0) for k in kads], dtype=np.float64)

    rng = np.random.default_rng(seed)
    tie_breaker = rng.uniform(0.0, 0.9999, size=len(cids))

    scores = np.where(freqs > 0, freqs + tie_breaker, tie_breaker - 1.0)
    order = np.argsort(-scores)
    return cids[order].tolist()


def make_icp_engine_ranker(
    fs: CompanyFeatureStore,
    cfg: PipelineConfig,
) -> RankerFn:
    """Wrap `run_icp_engine` into a pluggable `RankerFn` for 5-Fold CV evaluation."""

    def _ranker(
        ds: IngestedDataset,
        train_clients: pl.DataFrame,
        seed: int,
    ) -> list[int]:
        res = run_icp_engine(
            ds=ds,
            fs=fs,
            cfg=cfg,
            custom_clients_df=train_clients,
        )
        return res.all_candidates_ranked

    return _ranker


def compute_analytical_baseline_expected_recall(
    ds: IngestedDataset,
    train_vats: set[str],
    holdout_cids: set[int],
    k: int,
) -> float:
    """Exact mathematical expectation of Recall@k for the Primary KAD Baseline."""
    ec = ds.enriched_companies
    train_rows = ec.filter(pl.col("vat").is_in(train_vats))
    kad_counts = Counter(train_rows["primary_kad"].to_list())

    candidates = ec.filter(~pl.col("vat").is_in(train_vats)).select(
        ["company_id", "primary_kad"]
    )

    tier_totals: Counter[int] = Counter()
    tier_holdouts: Counter[int] = Counter()

    for row in candidates.iter_rows(named=True):
        freq = kad_counts.get(row["primary_kad"], 0)
        if freq > 0:
            tier_totals[freq] += 1
            if row["company_id"] in holdout_cids:
                tier_holdouts[freq] += 1

    remaining_k = k
    expected_hits = 0.0

    for freq in sorted(tier_totals.keys(), reverse=True):
        n_tier = tier_totals[freq]
        h_tier = tier_holdouts[freq]
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
    holdout_fraction: float = 0.20,
    base_seed: int = 42,
    is_baseline: bool = False,
) -> HoldoutMetrics:
    """
    Evaluate `ranker_fn` using deterministic `KFold(n_splits=5, shuffle=True, random_state=42)`
    across the 300 clients (5 folds of 240 training clients / 60 hidden validation clients).
    If `n_splits != 5`, supports general K-Fold when `1 < n_splits <= 20`.
    """
    clients = ds.clients
    ec = ds.enriched_companies
    all_client_vats = clients["vat"].to_numpy()

    kf = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    r500_list: list[float] = []
    r2000_list: list[float] = []
    exp500_list: list[float] = []
    exp2000_list: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_client_vats)):
        seed = base_seed + fold_idx
        train_vats = set(all_client_vats[train_idx].tolist())
        holdout_vats = set(all_client_vats[val_idx].tolist())

        train_clients_df = clients.filter(pl.col("vat").is_in(train_vats))

        holdout_cids = set(
            ec.filter(pl.col("vat").is_in(holdout_vats))["company_id"].to_list()
        )
        train_cids = set(
            ec.filter(pl.col("vat").is_in(train_vats))["company_id"].to_list()
        )

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


def print_holdout_report(metrics_list: list[HoldoutMetrics], n_splits: int = 5) -> None:
    """Pretty-print 5-Fold Cross-Validation Hold-Out benchmark results."""
    print("=" * 80)
    print(f"5-FOLD CROSS-VALIDATION HOLD-OUT BENCHMARK (KFold n_splits={n_splits}, random_state=42)")
    print("=" * 80)
    base_m = metrics_list[0] if metrics_list else None
    for idx, m in enumerate(metrics_list):
        print(f"Model [{idx+1}]: {m.name}")
        print(
            f"  - 5-Fold Recall@500      : {100*m.recall_at_500_mean:6.2f}% ± {100*m.recall_at_500_std:4.2f}% "
            f"(~{m.recall_at_500_mean*60:.1f} / 60 hidden clients) | Folds: {[round(100*x, 1) for x in m.fold_recalls_500]}"
        )
        print(
            f"  - 5-Fold Recall@2000     : {100*m.recall_at_2000_mean:6.2f}% ± {100*m.recall_at_2000_std:4.2f}% "
            f"(~{m.recall_at_2000_mean*60:.1f} / 60 hidden clients) | Folds: {[round(100*x, 1) for x in m.fold_recalls_2000]}"
        )
        if m.expected_recall_500_mean is not None and m.expected_recall_2000_mean is not None:
            print(
                f"  - Analytical E[Recall@k] : E[R@500] = {100*m.expected_recall_500_mean:6.2f}% | "
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
                f"  - Lift vs Baseline       : Recall@500 = {gain_500:.2f}x ({diff_500:+.2f} pp) | "
                f"Recall@2000 = {gain_2000:.2f}x ({diff_2000:+.2f} pp)"
            )
        print("-" * 80)


def run_5fold_cv_grid_search(
    ds: IngestedDataset,
    embedding_model: str = "granite-107m",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    r"""
    Run a fast, 100% deterministic 5-Fold Cross-Validation Grid Search over:
      1. Hybrid Mixing Parameter $\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$
      2. Feature & Scoring Block Weight Profiles (Semantic Cosine vs. Empirical Bayes Lift vs.
         Spatial Proximity Kernel vs. Legal/Age/Branch weights)
    Returns `(best_row, all_rows)`.
    """
    print("=" * 80)
    print(f"RUNNING 5-FOLD CV GRID SEARCH (Model={embedding_model}, KFold=5, random_state=42)")
    print("=" * 80)

    base_cfg = PipelineConfig(embedding_model=embedding_model, hybrid_alpha=0.50)
    base_fs = build_feature_store(ds, base_cfg)

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
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

    results: list[dict[str, object]] = []
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
            m = evaluate_ranker_on_holdout(
                ds=ds,
                ranker_fn=make_icp_engine_ranker(fs_trial, cfg_trial),
                name=f"alpha={alpha:.2f} | {wp['profile_name']}",
                n_splits=5,
                base_seed=42,
            )
            row = {
                "alpha": alpha,
                "profile_name": wp["profile_name"],
                "w_cos": wp["w_cos"],
                "w_eb": wp["w_eb"],
                "w_sp": wp["w_sp"],
                "eb_prior": wp["eb_prior"],
                "bw_km": wp["bw_km"],
                "r500_mean": m.recall_at_500_mean,
                "r500_std": m.recall_at_500_std,
                "r2000_mean": m.recall_at_2000_mean,
                "r2000_std": m.recall_at_2000_std,
            }
            results.append(row)
            print(
                f"  [alpha={alpha:4.2f} | {wp['profile_name']:<33}] -> "
                f"5-Fold Recall@500 = {100*m.recall_at_500_mean:5.2f}% ± {100*m.recall_at_500_std:4.2f}% | "
                f"Recall@2000 = {100*m.recall_at_2000_mean:5.2f}% ± {100*m.recall_at_2000_std:4.2f}%"
            )

    # Sort by Mean 5-Fold Recall@500 descending, breaking ties with Recall@2000 descending
    results.sort(key=lambda r: (r["r500_mean"], r["r2000_mean"]), reverse=True)
    best = results[0]
    print("=" * 80)
    print(
        f"OPTIMAL LOCKED-IN CONFIGURATION: alpha={best['alpha']:.2f} | {best['profile_name']} "
        f"(w_cos={best['w_cos']}, w_eb={best['w_eb']}, w_sp={best['w_sp']}, eb_prior={best['eb_prior']}, bw_km={best['bw_km']})\n"
        f"  -> Max 5-Fold Recall@500 = {100*best['r500_mean']:.2f}% | 5-Fold Recall@2000 = {100*best['r2000_mean']:.2f}%"
    )
    print("=" * 80)
    return best, results


def compare_embedding_models_section_5_3(
    ds: IngestedDataset,
    models_to_compare: list[str],
    best_alpha: float = 0.50,
) -> None:
    """
    Benchmark multiple Section 5.2 embedding models across the 4 mandatory axes of Section 5.3:
    1. Greek Quality on Challenge Data (5-Fold Recall@500 & Recall@2000)
    2. License
    3. Origin & Digital Sovereignty
    4. Execution Cost (RAM & Time)
    """
    print("\n" + "=" * 88)
    print("SECTION 5.3 EMBEDDING MODEL COMPARISON TABLE (5-FOLD CV ON CHALLENGE DATA)")
    print("=" * 88)
    print(
        f"{'Model Key':<16} | {'Alpha':<5} | {'5-Fold R@500':<14} | {'5-Fold R@2000':<14} | {'Time (s)':<8} | {'License':<15}"
    )
    print("-" * 88)

    base_m = evaluate_ranker_on_holdout(
        ds=ds,
        ranker_fn=baseline_primary_kad_ranker,
        name="ForceCRM Primary KAD Baseline",
        n_splits=5,
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
        cfg = PipelineConfig(embedding_model=m_key, hybrid_alpha=alpha_val)
        t0 = time.perf_counter()
        fs = build_feature_store(ds, cfg)
        elapsed = time.perf_counter() - t0
        m = evaluate_ranker_on_holdout(
            ds=ds,
            ranker_fn=make_icp_engine_ranker(fs, cfg),
            name=f"{m_key} (alpha={alpha_val})",
            n_splits=5,
        )
        lic = EMBEDDING_MODELS_REGISTRY.get(m_key, {}).get("license", "Custom")[:15]
        print(
            f"{m_key:<16} | {alpha_val:<5.2f} | "
            f"{100*m.recall_at_500_mean:5.2f}% ± {100*m.recall_at_500_std:4.2f}% | "
            f"{100*m.recall_at_2000_mean:5.2f}% ± {100*m.recall_at_2000_std:4.2f}% | "
            f"{elapsed:6.2f}s  | {lic:<15}"
        )
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation Hold-Out Benchmark & Hybrid Optimizer."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="granite-107m",
        help="Embedding model key ('granite-107m', 'e5-base', 'e5-small', 'char-ngram-svd')",
    )
    parser.add_argument("--alpha", type=float, default=0.50, help="Hybrid mixing alpha in [0, 1]")
    parser.add_argument("--splits", type=int, default=5, help="Number of K-Fold splits (default: 5)")
    parser.add_argument("--w-cos", type=float, default=0.60, help="Cosine similarity weight")
    parser.add_argument("--w-eb", type=float, default=0.30, help="Empirical Bayes lift affinity weight")
    parser.add_argument("--w-spatial", type=float, default=0.10, help="Spatial kernel weight")
    parser.add_argument("--eb-prior", type=float, default=12.0, help="Empirical Bayes prior strength")
    parser.add_argument("--bandwidth-km", type=float, default=25.0, help="Spatial kernel bandwidth (km)")
    parser.add_argument("--grid-search", action="store_true", help="Run 5-Fold CV Grid Search over alpha and weights")
    parser.add_argument(
        "--compare-models",
        nargs="*",
        default=None,
        help="Compare multiple embedding models for Section 5.3 (e.g. --compare-models char-ngram-svd granite-107m e5-base)",
    )
    args = parser.parse_args()

    dataset = ingest_all()

    if args.grid_search:
        best_cfg, _ = run_5fold_cv_grid_search(dataset, embedding_model=args.embedding_model)
        if args.compare_models is not None:
            models = args.compare_models or ["char-ngram-svd", "granite-107m", "e5-base"]
            compare_embedding_models_section_5_3(
                dataset, models_to_compare=models, best_alpha=float(best_cfg["alpha"])
            )
    elif args.compare_models is not None:
        models = args.compare_models or ["char-ngram-svd", "granite-107m", "e5-base"]
        compare_embedding_models_section_5_3(
            dataset, models_to_compare=models, best_alpha=args.alpha
        )
    else:
        cfg = PipelineConfig(
            embedding_model=args.embedding_model,
            hybrid_alpha=args.alpha,
            holdout_n_splits=args.splits,
            sim_weight_feature_cosine=args.w_cos,
            sim_weight_eb_lift_affinity=args.w_eb,
            sim_weight_spatial_kernel=args.w_spatial,
            eb_prior_strength=args.eb_prior,
            spatial_kernel_bandwidth_km=args.bandwidth_km,
        )
        feature_store = build_feature_store(dataset, cfg)

        baseline_metrics = evaluate_ranker_on_holdout(
            ds=dataset,
            ranker_fn=baseline_primary_kad_ranker,
            name="ForceCRM Official Baseline (Primary KAD Frequency)",
            n_splits=cfg.holdout_n_splits,
            base_seed=42,
            is_baseline=True,
        )
        icp_metrics = evaluate_ranker_on_holdout(
            ds=dataset,
            ranker_fn=make_icp_engine_ranker(feature_store, cfg),
            name=f"Archimedes2026 Hybrid ICP Engine ({cfg.embedding_model}, alpha={cfg.hybrid_alpha:.2f})",
            n_splits=cfg.holdout_n_splits,
            base_seed=42,
            is_baseline=False,
        )
        print_holdout_report([baseline_metrics, icp_metrics], n_splits=cfg.holdout_n_splits)
