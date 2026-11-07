"""
Central Configuration (`src/config.py`)
=======================================
Single source of truth for all hyperparameters, feature weights, embedding models,
clustering settings, statistical thresholds, and evaluation options.

You can tweak defaults directly in `PipelineConfig` below, pass overrides in code
(`PipelineConfig(w_primary_kad=4.0)`), or override them via CLI flags when running
`python -m src.features` or `python evaluate_holdout.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Registry of local embedding models for Section 5.2 & 5.3 benchmarking
EMBEDDING_MODELS_REGISTRY: dict[str, dict[str, str]] = {
    "granite-107m": {
        "hf_id": "ibm-granite/granite-embedding-107m-multilingual",
        "origin": "USA (IBM Research)",
        "license": "Apache-2.0 (Commercial use permitted)",
        "params": "~107M",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "granite-278m": {
        "hf_id": "ibm-granite/granite-embedding-278m-multilingual",
        "origin": "USA (IBM Research)",
        "license": "Apache-2.0 (Commercial use permitted)",
        "params": "~278M",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-base": {
        "hf_id": "intfloat/multilingual-e5-base",
        "origin": "USA/China (Microsoft Research)",
        "license": "MIT (Commercial use permitted)",
        "params": "~278M",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-small": {
        "hf_id": "intfloat/multilingual-e5-small",
        "origin": "USA/China (Microsoft Research)",
        "license": "MIT (Commercial use permitted)",
        "params": "~118M",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "char-ngram-svd": {
        "hf_id": "local/greek-char-word-tfidf-svd",
        "origin": "Local Deterministic Greek Morphology SVD (Zero-Download Baseline)",
        "license": "MIT / Built-in",
        "params": "Deterministic SVD (128d)",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


@dataclass
class PipelineConfig:
    """
    Master configuration for the ForceCRM B2B Profile Analysis & Lookalike Pipeline.
    All parameters are exposed here for rapid experimentation and ablation studies.
    """

    # -------------------------------------------------------------------------
    # 1. Paths & Caching
    # -------------------------------------------------------------------------
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    use_disk_cache: bool = True

    # -------------------------------------------------------------------------
    # 2. Embedding Model & KAD Activity Weighting (Requirements A1 & A3)
    # -------------------------------------------------------------------------
    # Choose key from EMBEDDING_MODELS_REGISTRY ("granite-107m", "e5-base", "e5-small", "char-ngram-svd")
    # or pass any HuggingFace model ID directly.
    embedding_model: str = "granite-107m"
    embedding_batch_size: int = 128
    embedding_device: str = "cpu"  # Safe default for Ryzen 2600X + RX580

    # Relative weights by activity_type (Primary > Secondary > Auxiliary)
    w_primary_kad: float = 3.5
    w_secondary_kad: float = 0.75
    w_auxiliary_kad: float = 0.25

    # Smooth IDF weighting to suppress ubiquitous/noisy secondary KADs:
    # idf(kad) = (log((1 + N) / (1 + df)) + 1.0) ** idf_power
    use_idf_weighting: bool = True
    idf_power: float = 1.25

    # Sublinear scaling for companies with many secondary KADs (prevents 200-KAD noise domination)
    # Secondary weight is divided by (num_secondary_kads ** secondary_count_damping)
    secondary_count_damping: float = 0.35

    # Dimensionality of SVD-compressed semantic activity embedding used in clustering
    semantic_svd_dim: int = 24

    # -------------------------------------------------------------------------
    # 3. Spatial (`cKDTree`) & Structured Feature Engineering (Requirement A1 & X1)
    # -------------------------------------------------------------------------
    # Radii in kilometers for local hospitality density via 3D Earth-chord cKDTree
    spatial_radii_km: tuple[float, ...] = (1.0, 5.0, 20.0)

    # Feature block weights when forming the unified L2-normalized company representation
    block_weight_semantic: float = 1.00
    block_weight_kad_hierarchy: float = 0.85
    block_weight_spatial: float = 1.15
    block_weight_legal: float = 0.90
    block_weight_age_branch: float = 0.45

    # -------------------------------------------------------------------------
    # 4. ICP Discovery, Turnover Weighting & Stability (Requirements A2, A3, A4, A5, A7)
    # -------------------------------------------------------------------------
    min_icps: int = 1
    max_icps: int = 6
    # Minimum silhouette / BIC improvement required to prefer K > 1 over K = 1 ("homogeneous")
    homogeneity_silhouette_threshold: float = 0.08

    # How client turnover weights ICP centroids & lift:
    # "log_turnover" -> log1p(turnover), "sqrt_turnover" -> sqrt(turnover),
    # "linear_turnover" -> turnover, "uniform" -> 1.0
    turnover_weighting_mode: str = "log_turnover"
    turnover_weight_alpha: float = 0.50  # Blend between uniform (0.0) and pure turnover weight (1.0)

    # Bootstrap stability settings (Requirement A5)
    n_bootstrap_runs: int = 30
    bootstrap_sample_fraction: float = 0.85

    # Empirical Bayes Shrinkage & Hypergeometric test settings (Requirement A4)
    # Smoothed lift = (k_cluster + eb_prior_strength * p_univ * n_cluster) / (n_cluster + eb_prior_strength) / p_univ
    eb_prior_strength: float = 10.0
    fdr_alpha: float = 0.05  # Benjamini-Hochberg multiple-testing p-value threshold

    # Lookalike & Assignment scoring parameters (Requirements A3 & A7)
    no_fit_quantile_threshold: float = 0.05  # Bottom quantile or absolute similarity cutoff for "No Fit"
    no_fit_min_score: float = 0.35
    total_lookalikes_to_export: int = 2000

    # Relative weights in the unified ICP Similarity / Commercial Fit metric
    sim_weight_feature_cosine: float = 0.45
    sim_weight_eb_lift_affinity: float = 0.40
    sim_weight_spatial_kernel: float = 0.15
    spatial_kernel_bandwidth_km: float = 25.0

    # -------------------------------------------------------------------------
    # 5. Local LLM Explainer Settings (Requirement A6)
    # -------------------------------------------------------------------------
    ollama_model: str = "qwen2.5:7b-instruct"  # Fallback to deterministic template if Ollama isn't running
    llm_temperature: float = 0.0
    llm_seed: int = 2026
    llm_max_tokens: int = 450

    # -------------------------------------------------------------------------
    # 6. Evaluation Settings (Section 8.1)
    # -------------------------------------------------------------------------
    holdout_n_splits: int = 10
    holdout_fraction: float = 0.20
    random_seed: int = 2026

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary (converting Paths to strings)."""
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, Path):
                raw[k] = str(v)
        return raw

