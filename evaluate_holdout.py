"""
Stage 2: Hold-Out Evaluation Harness & Baseline (`evaluate_holdout.py`)
=======================================================================
Implements the objective evaluation protocol from Section 8.1 of the Technical Brief:
1. Splits the 300 clients into 80% Training Clients (240 VATs) and 20% Hidden Hold-Out
   Clients (60 VATs) across multiple random seeds for low-variance evaluation.
2. Implements the official ForceCRM Baseline (Section 8.1.4):
   "επιχειρήσεις με τον ίδιο κύριο ΚΑΔ με τους πελάτες, ταξινομημένες κατά συχνότητα
   του ΚΑΔ στο πελατολόγιο" (companies sharing a Primary KAD with the training clients,
   sorted by the frequency of that Primary KAD in the training client list).
3. Computes both:
   - Exact Analytical Expected Recall@500 and Recall@2000 (accounting for random tie-breaking
     within the boundary Primary KAD bucket), and
   - Seeded Tie-Broken Empirical Recall@500 and Recall@2000.
4. Provides a pluggable interface (`RankerFn`) so our upcoming ICP + Semantic Lookalike
   engine (`src/icp_engine.py`) can be benchmarked directly against the baseline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from src.ingest import IngestedDataset, ingest_all


# Type alias for any model ranking function:
# Given (dataset, train_clients_df, seed) -> returns ordered list/array of candidate company_ids
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
    ordered by the frequency of that Primary KAD among training clients (with seeded random
    tie-breaking within the same Primary KAD frequency tier).
    """
    ec = ds.enriched_companies
    train_vats = set(train_clients["vat"].to_list())

    # Identify training client rows and count primary KAD frequencies
    train_rows = ec.filter(pl.col("vat").is_in(train_vats))
    kad_counts = Counter(train_rows["primary_kad"].to_list())

    # Candidate Universe: strictly exclude ALL company_id rows belonging to training client VATs
    candidates = ec.filter(~pl.col("vat").is_in(train_vats)).select(
        ["company_id", "primary_kad"]
    )

    cids = candidates["company_id"].to_numpy()
    kads = candidates["primary_kad"].to_list()

    # Frequency of each candidate's primary KAD in the training client set (0 if not present)
    freqs = np.array([kad_counts.get(k, 0) for k in kads], dtype=np.float64)

    # Filter or rank: Section 8.1.4 states companies with the same primary KAD as clients,
    # sorted by KAD frequency in the client list.
    rng = np.random.default_rng(seed)
    tie_breaker = rng.uniform(0.0, 0.9999, size=len(cids))

    # Only candidates with freq > 0 have the same primary KAD as clients; others follow with score 0
    scores = np.where(freqs > 0, freqs + tie_breaker, tie_breaker - 1.0)
    order = np.argsort(-scores)
    return cids[order].tolist()


def compute_analytical_baseline_expected_recall(
    ds: IngestedDataset,
    train_vats: set[str],
    holdout_cids: set[int],
    k: int,
) -> float:
    """
    Compute the exact mathematical expectation of Recall@k for the Primary KAD Baseline
    under uniform tie-breaking within equal-frequency KAD tiers.
    """
    ec = ds.enriched_companies
    train_rows = ec.filter(pl.col("vat").is_in(train_vats))
    kad_counts = Counter(train_rows["primary_kad"].to_list())

    candidates = ec.filter(~pl.col("vat").is_in(train_vats)).select(
        ["company_id", "primary_kad"]
    )

    # Group candidates by their training-client KAD frequency (> 0 only, per baseline definition)
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
            # Partial tier at boundary k: uniform random selection picks remaining_k / n_tier
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
    """
    Run `n_splits` Monte Carlo 80/20 hold-out evaluations of `ranker_fn` and report
    Recall@500 and Recall@2000.
    """
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

        # Ground-truth hidden client company_ids in the Universe
        holdout_cids = set(
            ec.filter(pl.col("vat").is_in(holdout_vats))["company_id"].to_list()
        )
        train_cids = set(
            ec.filter(pl.col("vat").is_in(train_vats))["company_id"].to_list()
        )

        ranked_cids = ranker_fn(ds, train_clients_df, seed)

        # Verify no training client leaked into lookalike recommendations
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
    """Pretty-print Hold-Out evaluation benchmark results."""
    print("=" * 72)
    print(f"STAGE 2: HOLD-OUT EVALUATION BENCHMARK ({n_splits}x 80/20 Splits, 60 Hidden Clients)")
    print("=" * 72)
    for m in metrics_list:
        print(f"Model: {m.name}")
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
        print("-" * 72)


if __name__ == "__main__":
    dataset = ingest_all()
    baseline_metrics = evaluate_ranker_on_holdout(
        ds=dataset,
        ranker_fn=baseline_primary_kad_ranker,
        name="ForceCRM Official Baseline (Primary KAD Frequency)",
        n_splits=10,
        holdout_fraction=0.20,
        base_seed=2026,
        is_baseline=True,
    )
    print_holdout_report([baseline_metrics], n_splits=10)

