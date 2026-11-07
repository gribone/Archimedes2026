"""
Stage 2 & 4: Hold-Out Evaluation Harness & Benchmark Suite (`evaluate_holdout.py`)
==================================================================================
Evaluates and compares:
1. Official ForceCRM Baseline (Section 8.1.4): Primary KAD frequency in the training client list.
2. Our Unified Multi-Modal ICP & Lookalike Engine (`src/icp_engine.py`) with configurable
   hyperparameters via CLI flags or `PipelineConfig` (`src/config.py`).

Run with default settings:
    .venv/bin/python evaluate_holdout.py

Customize any parameter directly from the command line:
    .venv/bin/python evaluate_holdout.py --embedding-model char-ngram-svd --splits 10 --w-eb 0.55 --w-cos 0.30 --w-spatial 0.15
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from src.config import PipelineConfig
from src.features import CompanyFeatureStore, build_feature_store
from src.icp_engine import run_icp_engine
from src.ingest import IngestedDataset, ingest_all


RankerFn = Callable[[IngestedDataset, pl.DataFrame, int], list[int] | np.ndarray]


@dataclass
class HoldoutMetrics:
    """Stores Recall@k metrics across multiple Monte Carlo hold-out splits."""

    name: str
    recall_at_500_mean: float
    recall_at_500_std: float
    recall_at_2000_mean: float
    recall_at_2000_std: float
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
    """Wrap `run_icp_engine` into a pluggable `RankerFn` for Hold-Out evaluation."""

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
    n_splits: int = 10,
    holdout_fraction: float = 0.20,
    base_seed: int = 2026,
    is_baseline: bool = False,
) -> HoldoutMetrics:
    """Run `n_splits` Monte Carlo 80/20 hold-out evaluations of `ranker_fn`."""
    clients = ds.clients
    ec = ds.enriched_companies
    all_client_vats = clients["vat"].to_numpy()
    n_clients = len(all_client_vats)
    n_holdout = int(round(n_clients * holdout_fraction))

    r500_list: list[float] = []
    r2000_list: list[float] = []
    exp500_list: list[float] = []
    exp2000_list: list[float] = []

    for split_idx in range(n_splits):
        seed = base_seed + split_idx
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_clients)

        holdout_vats = set(all_client_vats[perm[:n_holdout]].tolist())
        train_vats = set(all_client_vats[perm[n_holdout:]].tolist())

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
        expected_recall_500_mean=float(np.mean(exp500_list)) if is_baseline else None,
        expected_recall_2000_mean=float(np.mean(exp2000_list)) if is_baseline else None,
    )


def print_holdout_report(metrics_list: list[HoldoutMetrics], n_splits: int = 10) -> None:
    """Pretty-print Hold-Out evaluation benchmark results and improvement over baseline."""
    print("=" * 76)
    print(f"HOLD-OUT EVALUATION BENCHMARK ({n_splits}x 80/20 Splits, 60 Hidden Clients)")
    print("=" * 76)
    base_m = metrics_list[0] if metrics_list else None
    for idx, m in enumerate(metrics_list):
        print(f"Model [{idx+1}]: {m.name}")
        print(
            f"  - Empirical Recall@500   : {100*m.recall_at_500_mean:6.2f}% ± {100*m.recall_at_500_std:4.2f}% "
            f"(~{m.recall_at_500_mean*60:.1f} / 60 hidden clients)"
        )
        print(
            f"  - Empirical Recall@2000  : {100*m.recall_at_2000_mean:6.2f}% ± {100*m.recall_at_2000_std:4.2f}% "
            f"(~{m.recall_at_2000_mean*60:.1f} / 60 hidden clients)"
        )
        if m.expected_recall_500_mean is not None and m.expected_recall_2000_mean is not None:
            print(
                f"  - Analytical E[Recall@k] : E[R@500] = {100*m.expected_recall_500_mean:6.2f}% | "
                f"E[R@2000] = {100*m.expected_recall_2000_mean:6.2f}%"
            )
        elif base_m is not None and idx > 0:
            gain_500 = m.recall_at_500_mean / max(base_m.recall_at_500_mean, 1e-6)
            gain_2000 = m.recall_at_2000_mean / max(base_m.recall_at_2000_mean, 1e-6)
            diff_500 = 100 * (m.recall_at_500_mean - base_m.recall_at_500_mean)
            diff_2000 = 100 * (m.recall_at_2000_mean - base_m.recall_at_2000_mean)
            print(
                f"  - Improvement vs Baseline: Recall@500 = {gain_500:.2f}x ({diff_500:+.2f} pp) | "
                f"Recall@2000 = {gain_2000:.2f}x ({diff_2000:+.2f} pp)"
            )
        print("-" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ForceCRM Baseline vs ICP Lookalike Engine with customizable parameters."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="char-ngram-svd",
        help="Embedding model key ('char-ngram-svd', 'granite-107m', 'e5-base', 'e5-small') or HF ID",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of 80/20 Monte Carlo splits")
    parser.add_argument("--min-icps", type=int, default=1, help="Minimum ICP clusters")
    parser.add_argument("--max-icps", type=int, default=6, help="Maximum ICP clusters")
    parser.add_argument("--w-primary", type=float, default=3.5, help="Primary KAD weight")
    parser.add_argument("--w-secondary", type=float, default=0.75, help="Secondary KAD weight")
    parser.add_argument("--w-cos", type=float, default=0.45, help="Cosine similarity weight in lookalike score")
    parser.add_argument("--w-eb", type=float, default=0.40, help="Empirical Bayes lift affinity weight")
    parser.add_argument("--w-spatial", type=float, default=0.15, help="Spatial kernel weight")
    parser.add_argument("--eb-prior", type=float, default=10.0, help="Empirical Bayes shrinkage prior strength")
    parser.add_argument("--bandwidth-km", type=float, default=25.0, help="Spatial Gaussian kernel bandwidth (km)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        embedding_model=args.embedding_model,
        holdout_n_splits=args.splits,
        min_icps=args.min_icps,
        max_icps=args.max_icps,
        w_primary_kad=args.w_primary,
        w_secondary_kad=args.w_secondary,
        sim_weight_feature_cosine=args.w_cos,
        sim_weight_eb_lift_affinity=args.w_eb,
        sim_weight_spatial_kernel=args.w_spatial,
        eb_prior_strength=args.eb_prior,
        spatial_kernel_bandwidth_km=args.bandwidth_km,
    )

    dataset = ingest_all()
    feature_store = build_feature_store(dataset, cfg)

    baseline_metrics = evaluate_ranker_on_holdout(
        ds=dataset,
        ranker_fn=baseline_primary_kad_ranker,
        name="ForceCRM Official Baseline (Primary KAD Frequency)",
        n_splits=cfg.holdout_n_splits,
        holdout_fraction=cfg.holdout_fraction,
        base_seed=cfg.random_seed,
        is_baseline=True,
    )

    icp_ranker = make_icp_engine_ranker(feature_store, cfg)
    icp_metrics = evaluate_ranker_on_holdout(
        ds=dataset,
        ranker_fn=icp_ranker,
        name=f"Archimedes2026 ICP Engine ({cfg.embedding_model}, w_cos={cfg.sim_weight_feature_cosine}, w_eb={cfg.sim_weight_eb_lift_affinity}, w_sp={cfg.sim_weight_spatial_kernel})",
        n_splits=cfg.holdout_n_splits,
        holdout_fraction=cfg.holdout_fraction,
        base_seed=cfg.random_seed,
        is_baseline=False,
    )

    print_holdout_report([baseline_metrics, icp_metrics], n_splits=cfg.holdout_n_splits)
