"""
CLI Entry Point for `python -m src.features`
============================================
"""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.features.store import build_feature_store, print_feature_store_diagnostics
from src.ingest import ingest_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Stage 3 Deterministic Hybrid Feature Engineering & Pre-Caching."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="granite-107m",
        help="Model key from EMBEDDING_MODELS_REGISTRY ('granite-107m', 'e5-base', 'e5-small', 'char-ngram-svd')",
    )
    parser.add_argument("--alpha", type=float, default=0.50, help="Hybrid mixing alpha in [0, 1]")
    parser.add_argument("--w-primary", type=float, default=3.5, help="Primary KAD weight")
    parser.add_argument("--w-secondary", type=float, default=0.75, help="Secondary KAD weight")
    parser.add_argument("--idf-power", type=float, default=1.25, help="IDF exponent")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild caches from scratch")
    args = parser.parse_args()

    config = PipelineConfig(
        embedding_model=args.embedding_model,
        hybrid_alpha=args.alpha,
        w_primary_kad=args.w_primary,
        w_secondary_kad=args.w_secondary,
        idf_power=args.idf_power,
        use_disk_cache=not args.no_cache,
    )

    dataset = ingest_all()
    feature_store = build_feature_store(dataset, config)
    print_feature_store_diagnostics(dataset, feature_store, config)
