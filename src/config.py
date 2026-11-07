"""
Central Configuration (`src/config.py`)
=======================================
Single source of truth for all hyperparameters, Hybrid KAD representation mixing (`alpha`),
feature block weights, static Parquet caching paths, deterministic clustering settings,
and 5-Fold Cross-Validation optimization parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Registry of local embedding models for Section 5.2 & 5.3 benchmarking
EMBEDDING_MODELS_REGISTRY: dict[str, dict[str, str]] = {
    "granite-107m": {
        "hf_id": "ibm-granite/granite-embedding-107m-multilingual",
        "origin": "USA (IBM Research)",
        "license": "Apache-2.0 (Commercial use permitted)",
        "params": "107M",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "granite-r2": {
        "hf_id": "ibm-granite/granite-embedding-278m-multilingual",
        "origin": "USA (IBM Research)",
        "license": "Apache-2.0 (Commercial use permitted)",
        "params": "278M",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-base": {
        "hf_id": "intfloat/multilingual-e5-base",
        "origin": "USA/China (Microsoft Research)",
        "license": "MIT (Commercial use permitted)",
        "params": "278M",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-small": {
        "hf_id": "intfloat/multilingual-e5-small",
        "origin": "USA/China (Microsoft Research)",
        "license": "MIT (Commercial use permitted)",
        "params": "118M",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "origin": "China (BAAI)",
        "license": "MIT (Commercial use permitted)",
        "params": "568M",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "char-ngram-svd": {
        "hf_id": "local/greek-char-word-ppmi-arpack-svd",
        "origin": "Local Deterministic Algebraic + PPMI ARPACK SVD (Signal B Pure)",
        "license": "MIT / Built-in",
        "params": "64d Exact ARPACK SVD",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


@dataclass
class PipelineConfig:
    """
    Master configuration for the ForceCRM B2B Profile Analysis & Lookalike Pipeline.
    Locked-in defaults reflect the 5-Fold Cross-Validation Grid Search optimum.
    """

    # -------------------------------------------------------------------------
    # 1. Paths & Static Parquet Pre-Caching
    # -------------------------------------------------------------------------
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    cached_kad_vectors_path: Path = Path("data/cached_kad_vectors.parquet")
    cached_universe_features_path: Path = Path("data/cached_universe_features.parquet")
    use_disk_cache: bool = True

    # -------------------------------------------------------------------------
    # 2. Hybrid Semantic KAD Representation (Signal A Neural + Signal B Algebraic/PPMI)
    # -------------------------------------------------------------------------
    # Choose key from EMBEDDING_MODELS_REGISTRY ("granite-107m", "e5-base", "e5-small", "char-ngram-svd")
    embedding_model: str = "granite-107m"
    embedding_batch_size: int = 128
    embedding_device: str = "cpu"  # Enforces torch.use_deterministic_algorithms(True) on CPU

    # Hybrid mixing parameter alpha in [0, 1]:
    # e_k^hybrid = [sqrt(alpha) * e_k^neural, sqrt(1 - alpha) * e_k^svd]
    hybrid_alpha: float = 0.50

    # Dimensionality of exact ARPACK SVD for Signal B (TF-IDF + NACE hierarchy + PPMI co-occurrence)
    svd_components: int = 64

    # Relative weights by activity_type (Primary > Secondary > Auxiliary)
    w_primary_kad: float = 3.5
    w_secondary_kad: float = 0.75
    w_auxiliary_kad: float = 0.25

    # Smooth IDF weighting to suppress ubiquitous/noisy secondary KADs
    use_idf_weighting: bool = True
    idf_power: float = 1.25

    # Sublinear scaling for companies with many secondary KADs (prevents 200-KAD noise domination)
    secondary_count_damping: float = 0.35

    # Dimensionality of company-level ARPACK SVD semantic summary used in clustering
    semantic_svd_dim: int = 24

    # -------------------------------------------------------------------------
    # 3. Spatial (`cKDTree`) & Structured Feature Engineering (Requirement A1 & X1)
    # -------------------------------------------------------------------------
    spatial_radii_km: tuple[float, ...] = (1.0, 5.0, 20.0)

    # Feature block weights when forming the unified L2-normalized company representation
    block_weight_semantic: float = 1.30
    block_weight_kad_hierarchy: float = 0.85
    block_weight_spatial: float = 1.00
    block_weight_legal: float = 0.85
    block_weight_age_branch: float = 0.45

    # -------------------------------------------------------------------------
    # 4. Deterministic ICP Discovery, Turnover Weighting & Stability (A2, A3, A4, A5, A7)
    # -------------------------------------------------------------------------
    min_icps: int = 1
    max_icps: int = 6
    homogeneity_silhouette_threshold: float = 0.08

    # Deterministic K-Means settings (eliminates downstream clustering randomness)
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 50
    kmeans_random_state: int = 42

    turnover_weighting_mode: str = "log_turnover"
    turnover_weight_alpha: float = 0.50

    # Bootstrap stability settings (Requirement A5)
    n_bootstrap_runs: int = 30
    bootstrap_sample_fraction: float = 0.85

    # Empirical Bayes Shrinkage & Hypergeometric test settings (Requirement A4)
    eb_prior_strength: float = 12.0
    fdr_alpha: float = 0.05

    # Lookalike & Assignment scoring parameters (Requirements A3 & A7)
    no_fit_quantile_threshold: float = 0.05
    no_fit_min_score: float = 0.35
    total_lookalikes_to_export: int = 2000

    # Relative weights in the unified ICP Similarity / Commercial Fit metric
    # (Optimized via 5-Fold Cross-Validation on Recall@500 & Recall@2000)
    sim_weight_feature_cosine: float = 0.60
    sim_weight_eb_lift_affinity: float = 0.30
    sim_weight_spatial_kernel: float = 0.10
    spatial_kernel_bandwidth_km: float = 25.0

    # -------------------------------------------------------------------------
    # 5. Local LLM Explainer Settings (Requirement A6)
    # -------------------------------------------------------------------------
    ollama_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.0
    llm_seed: int = 2026
    llm_max_tokens: int = 450

    # -------------------------------------------------------------------------
    # 6. Evaluation Settings (Section 8.1 — 5-Fold Cross-Validation)
    # -------------------------------------------------------------------------
    holdout_n_splits: int = 5
    holdout_fraction: float = 0.20
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary (converting Paths to strings)."""
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, Path):
                raw[k] = str(v)
        return raw
