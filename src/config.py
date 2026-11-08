"""
Central Configuration (`src/config.py`)
=======================================
Single source of truth for all hyperparameters, Hybrid KAD representation mixing (`alpha`),
feature block weights, static Parquet caching paths, deterministic clustering settings
(`kmeans_n_init=200`, `n_bootstrap_runs=200`), and RepeatedKFold (5x20=100) optimization
loaded automatically from `data/optimal_weights.json`.
"""

from __future__ import annotations

import json
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


OPTIMAL_WEIGHTS_PATH = Path("data/optimal_weights.json")

DEFAULT_TRAIT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("prefecture_name", 1.35),
    ("municipality_name", 0.90),
    ("geo_regime", 1.15),
    ("legal_type", 1.25),
    ("primary_kad", 1.20),
    ("kad_4digit", 0.70),
    ("age_bracket", 0.45),
    ("branch_status", 0.35),
    ("post_launch_status", 0.50),
    ("seasonal_launch", 0.40),
    ("brand_distinctiveness", 0.55),
    ("concept_tier", 0.60),
    ("muni_typology", 0.65),
    ("osm_tier", 0.50),
    ("transit_catchment", 0.45),
)


def load_optimal_weights_dict(path: str | Path = OPTIMAL_WEIGHTS_PATH) -> dict[str, Any]:
    """Load `data/optimal_weights.json` if it exists on disk, otherwise return empty dict."""
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@dataclass
class PipelineConfig:
    """
    Master configuration for the ForceCRM B2B Profile Analysis & Lookalike Pipeline.
    Automatically loads any locked-in hyperparameters from `data/optimal_weights.json`
    when initialized with default arguments, while preserving explicit overrides.
    """

    # -------------------------------------------------------------------------
    # 1. Paths & Static Parquet / SHA-256 Pre-Caching
    # -------------------------------------------------------------------------
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("output/cache")
    cached_kad_vectors_path: Path = Path("data/cached_kad_vectors.parquet")
    cached_universe_features_path: Path = Path("data/cached_universe_features.parquet")
    optimal_weights_path: Path = OPTIMAL_WEIGHTS_PATH
    use_disk_cache: bool = True
    use_sha256_client_cache: bool = True

    # -------------------------------------------------------------------------
    # 2. Hybrid Semantic KAD Representation (Signal A Neural + Signal B Algebraic/PPMI)
    # -------------------------------------------------------------------------
    use_embedding_models: bool = True
    embedding_model: str = "granite-107m"
    embedding_batch_size: int = 128
    embedding_device: str = "cpu"

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

    # Deterministic K-Means settings (200 initializations for global optimum WCSS)
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 200
    kmeans_random_state: int = 42

    turnover_weighting_mode: str = "log_turnover"
    turnover_weight_alpha: float = 0.50

    # Bootstrap stability settings (Requirement A5 — 200 resamples)
    n_bootstrap_runs: int = 200
    bootstrap_sample_fraction: float = 0.85

    # Empirical Bayes Shrinkage & Hypergeometric test settings (Requirement A4)
    eb_prior_strength: float = 12.0
    fdr_alpha: float = 0.05

    # Lookalike & Assignment scoring parameters (Requirements A3 & A7)
    no_fit_quantile_threshold: float = 0.05
    no_fit_min_score: float = 0.35
    total_lookalikes_to_export: int = 2000

    # Relative weights in the unified ICP Similarity / Commercial Fit metric
    sim_weight_feature_cosine: float = 0.60
    sim_weight_eb_lift_affinity: float = 0.30
    sim_weight_spatial_kernel: float = 0.10
    spatial_kernel_bandwidth_km: float = 25.0
    trait_weights: tuple[tuple[str, float], ...] = DEFAULT_TRAIT_WEIGHTS

    # -------------------------------------------------------------------------
    # 5. Local LLM Explainer Settings (Requirement A6)
    # -------------------------------------------------------------------------
    ollama_model: str = "gemma3:4b"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_max_tokens: int = 280

    # -------------------------------------------------------------------------
    # 6. Evaluation Settings (Section 8.1 — RepeatedKFold 5x20 = 100 Evaluations)
    # -------------------------------------------------------------------------
    holdout_n_splits: int = 5
    holdout_n_repeats: int = 20
    holdout_fraction: float = 0.20
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.use_embedding_models:
            self.embedding_model = "char-ngram-svd"
            self.hybrid_alpha = 0.0
        if self.trait_weights is not None:
            if isinstance(self.trait_weights, dict):
                self.trait_weights = tuple(
                    (str(k), float(v)) for k, v in self.trait_weights.items()
                )
            elif isinstance(self.trait_weights, (list, tuple)):
                self.trait_weights = tuple(
                    (str(item[0]), float(item[1])) for item in self.trait_weights
                )

    @classmethod
    def from_optimal_weights(cls, **overrides: Any) -> "PipelineConfig":
        """
        Construct a `PipelineConfig` populated from `data/optimal_weights.json` (if present),
        then apply any explicit keyword `overrides`.
        """
        opt = load_optimal_weights_dict()
        mapped: dict[str, Any] = {}
        key_map = {
            "embedding_model": "embedding_model",
            "hybrid_alpha": "hybrid_alpha",
            "alpha": "hybrid_alpha",
            "block_weight_semantic": "block_weight_semantic",
            "bw_sem": "block_weight_semantic",
            "block_weight_kad_hierarchy": "block_weight_kad_hierarchy",
            "bw_kad_hier": "block_weight_kad_hierarchy",
            "block_weight_spatial": "block_weight_spatial",
            "bw_spat": "block_weight_spatial",
            "block_weight_legal": "block_weight_legal",
            "bw_leg": "block_weight_legal",
            "block_weight_age_branch": "block_weight_age_branch",
            "bw_age_branch": "block_weight_age_branch",
            "sim_weight_feature_cosine": "sim_weight_feature_cosine",
            "w_cos": "sim_weight_feature_cosine",
            "sim_weight_eb_lift_affinity": "sim_weight_eb_lift_affinity",
            "w_eb": "sim_weight_eb_lift_affinity",
            "sim_weight_spatial_kernel": "sim_weight_spatial_kernel",
            "w_sp": "sim_weight_spatial_kernel",
            "eb_prior_strength": "eb_prior_strength",
            "eb_prior": "eb_prior_strength",
            "spatial_kernel_bandwidth_km": "spatial_kernel_bandwidth_km",
            "bw_km": "spatial_kernel_bandwidth_km",
            "kmeans_n_init": "kmeans_n_init",
            "n_bootstrap_runs": "n_bootstrap_runs",
            "trait_weights": "trait_weights",
        }
        for src_k, dst_k in key_map.items():
            if src_k in opt and dst_k not in mapped:
                mapped[dst_k] = opt[src_k]
        mapped.update(overrides)
        return cls(**mapped)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary (converting Paths to strings)."""
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, Path):
                raw[k] = str(v)
        return raw
