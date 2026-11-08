"""
Feature Engineering & Feature Store Package (`src/features`)
============================================================
Canonical package entrypoint aggregating and re-exporting all submodules:
- `src.features.store`: Fast contiguous 2D NumPy persistence & feature store assembly
- `src.features.spatial`: 3D cKDTree spatial density & HoReCa micro-clustering
- `src.features.transit`: Greek transit hubs data loader & coordinates
- `src.features.engineering`: Relocated predictive feature extractors
- `src.features.embeddings`: Hybrid Neural + Algebraic PPMI SVD embeddings
- `src.features.structured`: Firmographic blocks (KAD hierarchy, legal forms, branch)
- `src.features.text`: Greek text normalization and transliteration
- `src.features.enrichment`: OSM 3-tier matching and demographic indices
"""

from __future__ import annotations

from src.features.embeddings import (
    _canonicalize_svd_signs,
    _project_company_vectors_arpack,
    _resolve_model_info,
    build_sparse_company_kad_weight_matrix,
    compute_signal_a_neural_embeddings,
    compute_signal_b_algebraic_ppmi_svd,
    embed_kad_descriptions,
    fuse_hybrid_kad_vectors,
)
from src.features.engineering import (
    _compute_brand_features,
    _compute_geo_regime,
    _compute_municipality_typology,
    _compute_post_launch_expansion,
    _compute_seasonality_features,
    compute_brand_features,
    compute_geo_regime,
    compute_municipality_typology,
    compute_post_launch_expansion,
    compute_seasonality_features,
)
from src.features.enrichment import build_external_enrichment
from src.features.spatial import (
    EARTH_RADIUS_KM,
    build_spatial_ckdtree_features,
)
from src.features.store import (
    CompanyFeatureStore,
    build_feature_store,
    load_cached_feature_store,
    load_feature_store_npz,
    load_feature_store_parquet,
    print_feature_store_diagnostics,
    recombine_feature_store,
    save_feature_store_npz,
    save_feature_store_parquet,
)
from src.features.structured import build_structured_company_blocks
from src.features.text import (
    GREEK_TO_LATIN_TABLE,
    normalize_greek_text,
    transliterate_greek,
)
from src.features.transit import (
    FALLBACK_GREEK_TRANSIT_HUBS,
    GREEK_TRANSIT_FOOT_TRAFFIC_HUBS,
    load_greek_transit_hubs,
)

__all__ = [
    # Store & Dataclass
    "CompanyFeatureStore",
    "build_feature_store",
    "load_cached_feature_store",
    "load_feature_store_npz",
    "load_feature_store_parquet",
    "print_feature_store_diagnostics",
    "recombine_feature_store",
    "save_feature_store_npz",
    "save_feature_store_parquet",
    # Spatial
    "EARTH_RADIUS_KM",
    "build_spatial_ckdtree_features",
    # Transit
    "FALLBACK_GREEK_TRANSIT_HUBS",
    "GREEK_TRANSIT_FOOT_TRAFFIC_HUBS",
    "load_greek_transit_hubs",
    # Text
    "GREEK_TO_LATIN_TABLE",
    "normalize_greek_text",
    "transliterate_greek",
    # Engineering
    "compute_post_launch_expansion",
    "compute_seasonality_features",
    "compute_brand_features",
    "compute_municipality_typology",
    "compute_geo_regime",
    "_compute_post_launch_expansion",
    "_compute_seasonality_features",
    "_compute_brand_features",
    "_compute_municipality_typology",
    "_compute_geo_regime",
    # Structured
    "build_structured_company_blocks",
    # Embeddings
    "_canonicalize_svd_signs",
    "_project_company_vectors_arpack",
    "_resolve_model_info",
    "build_sparse_company_kad_weight_matrix",
    "compute_signal_a_neural_embeddings",
    "compute_signal_b_algebraic_ppmi_svd",
    "embed_kad_descriptions",
    "fuse_hybrid_kad_vectors",
    # Enrichment
    "build_external_enrichment",
]
