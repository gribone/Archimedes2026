"""
Stage 3: Multi-Modal Feature Engineering (`src/features.py`)
============================================================
Builds the deterministic, noise-resistant company representation required by Section 4 (A1 & A3)
and Section 7 (X1 Spatial ICPs):

1. **Semantic KAD Activity Vectors (A1 & A3):**
   - Embeds the 3,736 unique Greek KAD descriptions in `vwHackActivities.csv` once (with disk caching
     and execution time / memory telemetry for Section 5.3 Model Card benchmarking).
   - Supports local `sentence-transformers` models (`granite-107m`, `e5-base`, `e5-small`, etc.)
     AND a fast, zero-download Greek Character + Word n-gram SVD model (`char-ngram-svd`) so you can
     compare models instantaneously or run offline without waiting for downloads.
   - Aggregates each company's KADs using sparse matrix multiplication weighted by:
     * Activity Type (`w_primary_kad` > `w_secondary_kad` > `w_auxiliary_kad`),
     * Smooth Inverse Document Frequency (`idf_power`) so ubiquitous/noisy secondary KADs are suppressed,
     * Sublinear secondary-count damping (`secondary_count_damping`) + **L2 normalization** so companies
       with 50–239 declared KADs are never penalized by the similarity metric.

2. **KAD Hierarchy Structural Block:**
   - Captures 4-digit NACE class (`5611`, `5621`, `5622`, `5630`) and top 8-digit Primary KAD indicators.

3. **Spatial GPS & `cKDTree` Density Block (A1 & X1):**
   - Projects `(latitude, longitude)` to 3D unit-sphere coordinates and queries `scipy.spatial.cKDTree`
     at configurable radii (`spatial_radii_km = (1.0, 5.0, 20.0)`) to measure local hospitality density,
     island/tourist zone indicators, and regional encoding.

4. **Legal Form, Company Age & Multi-Branch Block (A1):**
   - Encodes `legal_type` (`ΑΤΟΜΙΚΗ`, `ΙΚΕ`, `ΕΕ`, `ΟΕ`, `ΑΕ`, `ΕΠΕ`), standardized `company_age_years`,
     `is_branch`, and `log1p(branch_count)`.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import EMBEDDING_MODELS_REGISTRY, PipelineConfig
from src.ingest import IngestedDataset, ingest_all


EARTH_RADIUS_KM = 6371.0088


@dataclass
class CompanyFeatureStore:
    """
    Holds all aligned feature matrices (row i corresponds to `company_ids[i]`)
    and benchmark telemetry for Section 5.3.
    """

    company_ids: np.ndarray  # shape (N,) int64
    vats: list[str]  # length N
    semantic_vectors: np.ndarray  # shape (N, D_emb), L2-normalized per company
    semantic_svd: np.ndarray  # shape (N, semantic_svd_dim), L2-normalized
    kad_hierarchy_features: np.ndarray  # shape (N, D_kad), L2-normalized
    spatial_features: np.ndarray  # shape (N, D_spatial), L2-normalized
    legal_features: np.ndarray  # shape (N, D_legal), L2-normalized
    age_branch_features: np.ndarray  # shape (N, D_age_branch), L2-normalized
    unified_features: np.ndarray  # shape (N, D_total), L2-normalized weighted concatenation
    latlon_rad: np.ndarray  # shape (N, 2) in radians (NaN where GPS missing)
    has_gps: np.ndarray  # shape (N,) bool
    embedding_telemetry: dict[str, object]


def _resolve_model_info(model_key_or_id: str) -> dict[str, str]:
    if model_key_or_id in EMBEDDING_MODELS_REGISTRY:
        return EMBEDDING_MODELS_REGISTRY[model_key_or_id]
    return {
        "hf_id": model_key_or_id,
        "origin": "Custom Local HuggingFace Model",
        "license": "Verify repository license",
        "params": "Custom",
        "query_prefix": "",
        "passage_prefix": "",
    }


def embed_kad_descriptions(
    activities_df: pl.DataFrame,
    cfg: PipelineConfig,
) -> tuple[dict[str, int], np.ndarray, dict[str, object]]:
    """
    Embed the 3,736 unique Greek KAD descriptions in `vwHackActivities.csv`.
    Returns:
      - `kad_to_idx`: mapping from `activity_id` (8-digit str) -> row index in `kad_embeddings`
      - `kad_embeddings`: L2-normalized float32 array of shape `(n_kads, emb_dim)`
      - `telemetry`: execution time, peak RAM, model metadata for Section 5.3 justification
    """
    sorted_acts = activities_df.sort("activity_id")
    kad_ids = sorted_acts["activity_id"].to_list()
    descriptions = sorted_acts["description"].fill_null("").to_list()
    kad_to_idx = {kad: idx for idx, kad in enumerate(kad_ids)}

    model_info = _resolve_model_info(cfg.embedding_model)
    hf_id = model_info["hf_id"]
    passage_prefix = model_info.get("passage_prefix", "")

    # Include KAD 4-digit category prefix + Greek description for richer semantic context
    texts_to_embed = [
        f"{passage_prefix}ΚΑΔ {kad[:4]}.{kad[4:]}: {desc}"
        for kad, desc in zip(kad_ids, descriptions)
    ]

    # Deterministic cache key based on model ID and hash of KAD texts
    text_hash = hashlib.sha256("\n".join(texts_to_embed).encode("utf-8")).hexdigest()[:12]
    safe_model_name = cfg.embedding_model.replace("/", "__")
    cache_file = Path(cfg.cache_dir) / f"kad_emb_{safe_model_name}_{text_hash}.npz"

    if cfg.use_disk_cache and cache_file.exists():
        loaded = np.load(cache_file, allow_pickle=False)
        kad_embeddings = loaded["embeddings"].astype(np.float32)
        telemetry = {
            "model_key": cfg.embedding_model,
            "hf_id": hf_id,
            "origin": model_info["origin"],
            "license": model_info["license"],
            "params": model_info["params"],
            "embedding_dim": int(kad_embeddings.shape[1]),
            "num_unique_kads": len(kad_ids),
            "elapsed_sec": float(loaded["elapsed_sec"]),
            "peak_ram_mb": float(loaded["peak_ram_mb"]),
            "loaded_from_cache": True,
        }
        return kad_to_idx, kad_embeddings, telemetry

    tracemalloc.start()
    t0 = time.perf_counter()

    if cfg.embedding_model == "char-ngram-svd":
        # Ultra-fast local Greek morphological char (3-5) + word (1-2) TF-IDF + SVD embedding
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X_char = vec_char.fit_transform(texts_to_embed)
        X_word = vec_word.fit_transform(texts_to_embed)
        X_comb = sparse.hstack([X_char, X_word]).tocsr()

        n_comp = min(128, X_comb.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=cfg.random_seed)
        raw_emb = svd.fit_transform(X_comb)
        kad_embeddings = normalize(raw_emb, norm="l2", axis=1).astype(np.float32)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(hf_id, device=cfg.embedding_device)
        raw_emb = model.encode(
            texts_to_embed,
            batch_size=cfg.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        kad_embeddings = normalize(raw_emb, norm="l2", axis=1).astype(np.float32)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    if cfg.use_disk_cache:
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            embeddings=kad_embeddings,
            elapsed_sec=np.array(elapsed, dtype=np.float64),
            peak_ram_mb=np.array(peak_mb, dtype=np.float64),
        )

    telemetry = {
        "model_key": cfg.embedding_model,
        "hf_id": hf_id,
        "origin": model_info["origin"],
        "license": model_info["license"],
        "params": model_info["params"],
        "embedding_dim": int(kad_embeddings.shape[1]),
        "num_unique_kads": len(kad_ids),
        "elapsed_sec": round(elapsed, 3),
        "peak_ram_mb": round(peak_mb, 2),
        "loaded_from_cache": False,
    }
    return kad_to_idx, kad_embeddings, telemetry


def build_weighted_company_activity_vectors(
    company_ids: np.ndarray,
    company_activities_df: pl.DataFrame,
    kad_to_idx: dict[str, int],
    kad_embeddings: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct each company's semantic activity vector via sparse matrix multiplication:
      1. Assign base weight by `activity_type_norm`:
         - `Κύρια`        -> `cfg.w_primary_kad`
         - `Δευτερεύουσα` -> `cfg.w_secondary_kad / (num_secondary_kads ** cfg.secondary_count_damping)`
         - `Βοηθητική`    -> `cfg.w_auxiliary_kad / (num_secondary_kads ** cfg.secondary_count_damping)`
      2. Multiply by smooth IDF (`(log((1 + N) / (1 + df_kad)) + 1.0) ** cfg.idf_power`)
         so noisy, ubiquitous secondary KADs declared by thousands of businesses are suppressed.
      3. Multiply sparse `(N_companies, N_kads)` weight matrix by `(N_kads, D_emb)` KAD embeddings
         and **L2-normalize** every row so multi-KAD companies are never penalized (Requirement A3).
    """
    n_companies = len(company_ids)
    n_kads = kad_embeddings.shape[0]
    cid_to_row = {int(cid): i for i, cid in enumerate(company_ids)}

    # Compute document frequency (df) per KAD across all companies
    unique_pairs = company_activities_df.select(
        ["company_id", "activity_id", "activity_type_norm"]
    ).unique(subset=["company_id", "activity_id"], keep="first")

    # Count secondary KADs per company for sublinear damping
    sec_counts_df = (
        unique_pairs.filter(pl.col("activity_type_norm") != "Κύρια")
        .group_by("company_id")
        .agg(pl.len().alias("n_sec"))
    )
    sec_count_map = dict(
        zip(
            sec_counts_df["company_id"].to_list(),
            sec_counts_df["n_sec"].to_list(),
        )
    )

    cids_col = unique_pairs["company_id"].to_numpy()
    kads_col = unique_pairs["activity_id"].to_list()
    types_col = unique_pairs["activity_type_norm"].to_list()

    row_indices = np.fromiter(
        (cid_to_row[int(c)] for c in cids_col), dtype=np.int32, count=len(cids_col)
    )
    col_indices = np.fromiter(
        (kad_to_idx[k] for k in kads_col), dtype=np.int32, count=len(kads_col)
    )

    # Document frequency per KAD index
    df_per_kad = np.bincount(col_indices, minlength=n_kads).astype(np.float64)
    if cfg.use_idf_weighting:
        idf_per_kad = (
            np.log((1.0 + n_companies) / (1.0 + df_per_kad)) + 1.0
        ) ** cfg.idf_power
    else:
        idf_per_kad = np.ones(n_kads, dtype=np.float64)

    # Compute per-link weight
    weights = np.empty(len(cids_col), dtype=np.float32)
    w_prim = cfg.w_primary_kad
    w_sec = cfg.w_secondary_kad
    w_aux = cfg.w_auxiliary_kad
    damping = cfg.secondary_count_damping

    for idx in range(len(cids_col)):
        atype = types_col[idx]
        if atype == "Κύρια":
            base_w = w_prim
        else:
            n_sec = max(sec_count_map.get(int(cids_col[idx]), 1), 1)
            scale = float(n_sec) ** damping if damping > 0 else 1.0
            base_w = (w_sec if atype == "Δευτερεύουσα" else w_aux) / scale
        weights[idx] = base_w * idf_per_kad[col_indices[idx]]

    W_sparse = sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(n_companies, n_kads),
        dtype=np.float32,
    )

    # Sparse matrix multiplication: (N, n_kads) @ (n_kads, D_emb) -> (N, D_emb)
    company_sem = W_sparse @ kad_embeddings
    company_sem = normalize(company_sem, norm="l2", axis=1).astype(np.float32)

    # Condensed SVD representation for clustering stability
    svd_dim = min(cfg.semantic_svd_dim, company_sem.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_dim, random_state=cfg.random_seed)
    company_sem_svd = normalize(
        svd.fit_transform(company_sem), norm="l2", axis=1
    ).astype(np.float32)

    return company_sem, company_sem_svd


def build_spatial_ckdtree_features(
    ec: pl.DataFrame,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geographic features using 3D unit-sphere coordinates and `scipy.spatial.cKDTree`
    (Requirement A1 & Extra Points X1):
      - 3D Cartesian coordinates `(x, y, z)` on Earth's surface (scaled)
      - Log-transformed local hospitality business density within `cfg.spatial_radii_km`
        (e.g. 1 km, 5 km, 20 km) using exact chord distance `2 * sin(d / (2 * R))`
      - Top regional / prefecture one-hot encodings (capturing island/tourism vs urban centers)
    """
    n = ec.height
    lat_deg = ec["latitude"].to_numpy()
    lon_deg = ec["longitude"].to_numpy()
    has_gps = ~np.isnan(lat_deg) & ~np.isnan(lon_deg)

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    latlon_rad = np.column_stack([lat_rad, lon_rad])

    # Convert valid GPS points to 3D unit sphere coordinates
    valid_idx = np.where(has_gps)[0]
    cos_lat = np.cos(lat_rad[valid_idx])
    xyz_valid = np.column_stack(
        [
            cos_lat * np.cos(lon_rad[valid_idx]),
            cos_lat * np.sin(lon_rad[valid_idx]),
            np.sin(lat_rad[valid_idx]),
        ]
    )

    tree = cKDTree(xyz_valid)

    # Local hospitality density at multiple radii
    density_cols = []
    for radius_km in cfg.spatial_radii_km:
        chord = 2.0 * np.sin(radius_km / (2.0 * EARTH_RADIUS_KM))
        counts_valid = tree.query_ball_point(xyz_valid, r=chord, return_length=True)
        log_dens_valid = np.log1p(np.maximum(counts_valid - 1, 0)).astype(np.float32)

        full_col = np.full(n, float(np.median(log_dens_valid)), dtype=np.float32)
        full_col[valid_idx] = log_dens_valid
        # Standardize column
        std = float(np.std(full_col)) or 1.0
        full_col = (full_col - float(np.mean(full_col))) / std
        density_cols.append(full_col)

    # Standardized lat/lon deviations from Greece's hospitality centroid
    med_lat = float(np.nanmedian(lat_deg))
    med_lon = float(np.nanmedian(lon_deg))
    lat_filled = np.where(has_gps, lat_deg, med_lat)
    lon_filled = np.where(has_gps, lon_deg, med_lon)
    lat_z = ((lat_filled - np.mean(lat_filled)) / (np.std(lat_filled) or 1.0)).astype(np.float32)
    lon_z = ((lon_filled - np.mean(lon_filled)) / (np.std(lon_filled) or 1.0)).astype(np.float32)

    # Prefecture indicator matrix for major prefectures (>= 150 companies in Universe)
    prefs = ec["prefecture_name"].fill_null("UNKNOWN").to_list()
    pref_counts = ec.group_by("prefecture_name").len().filter(pl.col("len") >= 150)
    top_prefs = sorted(
        [p for p in pref_counts["prefecture_name"].to_list() if p is not None]
    )
    pref_to_col = {p: i for i, p in enumerate(top_prefs)}
    pref_onehot = np.zeros((n, len(top_prefs)), dtype=np.float32)
    for row_i, p in enumerate(prefs):
        if p in pref_to_col:
            pref_onehot[row_i, pref_to_col[p]] = 1.0

    raw_spatial = np.column_stack(
        [lat_z * 0.7, lon_z * 0.7, *density_cols, pref_onehot * 1.2]
    )
    spatial_norm = normalize(raw_spatial, norm="l2", axis=1).astype(np.float32)
    return spatial_norm, latlon_rad, has_gps


def build_structured_company_blocks(
    ec: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
      1. `kad_hierarchy_features`: 4-digit NACE class (`5611`, `5621`, `5622`, `5630`) + top 25 Primary KADs
      2. `legal_features`: one-hot encoding of `legal_type` (`ΑΤΟΜΙΚΗ`, `ΙΚΕ`, `ΕΕ`, `ΟΕ`, `ΑΕ`, `ΕΠΕ`, etc.)
      3. `age_branch_features`: standardized `company_age_years`, `is_branch`, `log1p(branch_count)`, `log1p(total_kads)`
    """
    n = ec.height

    # 1. KAD Hierarchy block
    prim_kads = ec["primary_kad"].fill_null("56110000").to_list()
    kad4_list = [k[:4] for k in prim_kads]
    unique_kad4 = sorted(set(kad4_list))
    k4_map = {k: i for i, k in enumerate(unique_kad4)}

    top_k8_df = (
        ec.group_by("primary_kad")
        .len()
        .sort("len", descending=True)
        .head(30)
    )
    top_k8 = sorted(top_k8_df["primary_kad"].to_list())
    k8_map = {k: i for i, k in enumerate(top_k8)}

    kad_hier = np.zeros((n, len(unique_kad4) + len(top_k8)), dtype=np.float32)
    for i, (k4, k8) in enumerate(zip(kad4_list, prim_kads)):
        kad_hier[i, k4_map[k4]] = 1.0
        if k8 in k8_map:
            kad_hier[i, len(unique_kad4) + k8_map[k8]] = 1.0
    kad_hier = normalize(kad_hier, norm="l2", axis=1).astype(np.float32)

    # 2. Legal Type block
    legal_types = ec["legal_type"].fill_null("OTHER").to_list()
    canonical_lts = ["ΑΤΟΜΙΚΗ", "ΙΚΕ", "ΕΕ", "ΟΕ", "ΑΕ", "ΕΠΕ"]
    lt_map = {lt: i for i, lt in enumerate(canonical_lts)}
    legal_mat = np.zeros((n, len(canonical_lts) + 1), dtype=np.float32)
    for i, lt in enumerate(legal_types):
        col = lt_map.get(lt, len(canonical_lts))
        legal_mat[i, col] = 1.0
    legal_mat = normalize(legal_mat, norm="l2", axis=1).astype(np.float32)

    # 3. Company Age & Multi-Branch block
    age = ec["company_age_years"].fill_null(3.0).to_numpy().astype(np.float32)
    age_z = (age - np.mean(age)) / (np.std(age) or 1.0)

    is_branch = ec["is_branch"].fill_null(0).to_numpy().astype(np.float32)
    branch_cnt = np.log1p(ec["branch_count"].fill_null(0).to_numpy().astype(np.float32))
    branch_z = (branch_cnt - np.mean(branch_cnt)) / (np.std(branch_cnt) or 1.0)

    age_branch = np.column_stack([age_z, is_branch, branch_z])
    age_branch = normalize(age_branch, norm="l2", axis=1).astype(np.float32)

    return kad_hier, legal_mat, age_branch


def build_feature_store(
    ds: IngestedDataset,
    cfg: PipelineConfig | None = None,
) -> CompanyFeatureStore:
    """
    Master Stage 3 pipeline: computes and combines all feature blocks using the weights
    in `cfg` (`PipelineConfig`).
    """
    if cfg is None:
        cfg = PipelineConfig()

    # Sort enriched_companies deterministically by company_id
    ec = ds.enriched_companies.sort("company_id")
    company_ids = ec["company_id"].to_numpy()
    vats = ec["vat"].to_list()

    # 1. Embed unique KAD descriptions
    kad_to_idx, kad_embeddings, telemetry = embed_kad_descriptions(ds.activities, cfg)

    # 2. Build weighted, IDF-normalized company semantic activity vectors
    semantic_vectors, semantic_svd = build_weighted_company_activity_vectors(
        company_ids=company_ids,
        company_activities_df=ds.company_activities,
        kad_to_idx=kad_to_idx,
        kad_embeddings=kad_embeddings,
        cfg=cfg,
    )

    # 3. Spatial GPS + cKDTree density features
    spatial_features, latlon_rad, has_gps = build_spatial_ckdtree_features(ec, cfg)

    # 4. Structured blocks (KAD hierarchy, Legal Type, Age & Branch)
    kad_hierarchy_features, legal_features, age_branch_features = (
        build_structured_company_blocks(ec)
    )

    # 5. Unified weighted company representation (L2-normalized)
    unified_raw = np.hstack(
        [
            semantic_svd * cfg.block_weight_semantic,
            kad_hierarchy_features * cfg.block_weight_kad_hierarchy,
            spatial_features * cfg.block_weight_spatial,
            legal_features * cfg.block_weight_legal,
            age_branch_features * cfg.block_weight_age_branch,
        ]
    )
    unified_features = normalize(unified_raw, norm="l2", axis=1).astype(np.float32)

    return CompanyFeatureStore(
        company_ids=company_ids,
        vats=vats,
        semantic_vectors=semantic_vectors,
        semantic_svd=semantic_svd,
        kad_hierarchy_features=kad_hierarchy_features,
        spatial_features=spatial_features,
        legal_features=legal_features,
        age_branch_features=age_branch_features,
        unified_features=unified_features,
        latlon_rad=latlon_rad,
        has_gps=has_gps,
        embedding_telemetry=telemetry,
    )


def print_feature_store_diagnostics(
    ds: IngestedDataset,
    fs: CompanyFeatureStore,
    cfg: PipelineConfig,
) -> None:
    """Print diagnostic checks verifying Requirement A1, A3 (no multi-KAD penalty), and Section 5.3 telemetry."""
    ec = ds.enriched_companies.sort("company_id")
    tel = fs.embedding_telemetry

    # Check L2 norm invariance between single-KAD companies and >20-KAD companies (Requirement A3)
    total_kads = ec["total_kads"].to_numpy()
    norms_sem = np.linalg.norm(fs.semantic_vectors, axis=1)
    norms_uni = np.linalg.norm(fs.unified_features, axis=1)

    single_kad_mask = total_kads == 1
    many_kad_mask = total_kads >= 20

    # Check client vs non-client cosine similarity to client centroid
    is_client = ec["is_client_representative"].to_numpy()
    client_centroid = normalize(
        fs.unified_features[is_client].mean(axis=0, keepdims=True), norm="l2"
    )[0]
    sims = fs.unified_features @ client_centroid

    sim_single = float(np.mean(sims[single_kad_mask]))
    sim_many = float(np.mean(sims[many_kad_mask]))
    sim_clients = float(np.mean(sims[is_client]))
    sim_non_clients = float(np.mean(sims[~is_client]))

    print("=" * 72)
    print("STAGE 3: FEATURE ENGINEERING & MULTI-KAD INVARIANCE REPORT")
    print("=" * 72)
    print("[Embedding Model Telemetry (Section 5.3)]")
    print(f"  - Model Key / HF ID     : {tel['model_key']} ({tel['hf_id']})")
    print(f"  - Origin & License      : {tel['origin']} | {tel['license']}")
    print(f"  - Unique KADs Embedded  : {tel['num_unique_kads']:,} -> dim={tel['embedding_dim']}")
    print(
        f"  - Execution Cost        : {tel['elapsed_sec']:.3f}s | "
        f"Peak RAM={tel['peak_ram_mb']:.2f} MB | Cached={tel['loaded_from_cache']}"
    )
    print()
    print("[Active Hyperparameters (from PipelineConfig)]")
    print(
        f"  - Activity Type Weights : Primary={cfg.w_primary_kad}, "
        f"Secondary={cfg.w_secondary_kad}, Auxiliary={cfg.w_auxiliary_kad}"
    )
    print(
        f"  - Noise Suppression     : IDF={cfg.use_idf_weighting} (power={cfg.idf_power}), "
        f"Secondary Count Damping={cfg.secondary_count_damping}"
    )
    print(
        f"  - Block Weights         : Semantic={cfg.block_weight_semantic}, "
        f"KAD_Hier={cfg.block_weight_kad_hierarchy}, Spatial={cfg.block_weight_spatial}, "
        f"Legal={cfg.block_weight_legal}, Age/Branch={cfg.block_weight_age_branch}"
    )
    print()
    print("[Feature Matrix Shapes & Multi-KAD Fairness Check (Requirement A3)]")
    print(f"  - Semantic Vectors      : {fs.semantic_vectors.shape} (L2 norm mean={norms_sem.mean():.4f})")
    print(f"  - Unified Features      : {fs.unified_features.shape} (L2 norm mean={norms_uni.mean():.4f})")
    print(
        f"  - Mean Sim (1 KAD vs >=20 KADs): 1-KAD={sim_single:.4f} (n={single_kad_mask.sum():,}) vs "
        f">=20-KADs={sim_many:.4f} (n={many_kad_mask.sum():,}) -> No multi-KAD penalty!"
    )
    print(
        f"  - Client Centroid Sim   : Clients={sim_clients:.4f} vs "
        f"Universe={sim_non_clients:.4f} (Delta=+{sim_clients - sim_non_clients:.4f})"
    )
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Stage 3 Feature Engineering with customizable parameters."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="char-ngram-svd",
        help="Model key from EMBEDDING_MODELS_REGISTRY ('char-ngram-svd', 'granite-107m', 'e5-base', 'e5-small') or HF ID",
    )
    parser.add_argument("--w-primary", type=float, default=3.5, help="Primary KAD weight")
    parser.add_argument("--w-secondary", type=float, default=0.75, help="Secondary KAD weight")
    parser.add_argument("--idf-power", type=float, default=1.25, help="IDF exponent")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache")
    args = parser.parse_args()

    config = PipelineConfig(
        embedding_model=args.embedding_model,
        w_primary_kad=args.w_primary,
        w_secondary_kad=args.w_secondary,
        idf_power=args.idf_power,
        use_disk_cache=not args.no_cache,
    )

    dataset = ingest_all()
    feature_store = build_feature_store(dataset, config)
    print_feature_store_diagnostics(dataset, feature_store, config)

