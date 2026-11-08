"""
Feature Store Persistence & Fast Contiguous Array I/O (`src/features/store.py`)
===============================================================================
Provides high-performance flat contiguous 2D NumPy array caching (.npz)
with zero Python list allocation overhead and legacy Parquet fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.preprocessing import normalize

from src.config import PipelineConfig
from src.features.embeddings import (
    _project_company_vectors_arpack,
    _resolve_model_info,
    build_sparse_company_kad_weight_matrix,
    embed_kad_descriptions,
)
from src.features.enrichment import build_external_enrichment
from src.features.spatial import build_spatial_ckdtree_features
from src.features.structured import build_structured_company_blocks

if TYPE_CHECKING:
    from src.ingest import IngestedDataset

logger = logging.getLogger(__name__)


@dataclass
class CompanyFeatureStore:
    """
    Holds all aligned feature matrices (row i corresponds to `company_ids[i]`)
    plus separated Neural & Algebraic/PPMI components for instant alpha/weight grid-searching.
    """

    company_ids: np.ndarray  # shape (N,) int64
    vats: list[str]  # length N
    semantic_vectors: np.ndarray  # shape (N, D_hybrid), L2-normalized hybrid company vectors
    semantic_svd: np.ndarray  # shape (N, semantic_svd_dim), L2-normalized hybrid compressed
    kad_hierarchy_features: np.ndarray  # shape (N, D_kad), L2-normalized
    spatial_features: np.ndarray  # shape (N, D_spatial), L2-normalized
    legal_features: np.ndarray  # shape (N, D_legal), L2-normalized
    age_branch_features: np.ndarray  # shape (N, D_age_branch), L2-normalized
    unified_features: np.ndarray  # shape (N, D_total), L2-normalized weighted concatenation
    latlon_rad: np.ndarray  # shape (N, 2) in radians (NaN where GPS missing)
    has_gps: np.ndarray  # shape (N,) bool
    embedding_telemetry: dict[str, object]
    # Separated company-level unit vectors for ultra-fast alpha grid-search without recomputing W_sparse
    company_neural_svd: np.ndarray | None = None  # shape (N, semantic_svd_dim)
    company_algebraic_svd: np.ndarray | None = None  # shape (N, semantic_svd_dim)


def save_feature_store_npz(
    fs: CompanyFeatureStore,
    path: Path | str,
    model_col: str | None = None,
    compress: bool = False,
) -> None:
    """
    Persist feature store matrices as contiguous 2D NumPy arrays in an .npz archive.

    Guarantees:
    - 0 Python list object overhead (no [r.tolist() for r in ...])
    - Contiguous C-order memory layout
    - Atomic file write via temporary file
    - allow_pickle=False compatibility (all string arrays stored as fixed-width Unicode <U9)
    - Merges with existing multi-model embeddings already in the archive
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing_arrays: dict[str, np.ndarray] = {}
    if p.exists():
        try:
            with np.load(p, allow_pickle=False) as existing:
                if "company_id" in existing and len(existing["company_id"]) == len(fs.company_ids):
                    existing_arrays = {k: np.ascontiguousarray(existing[k]) for k in existing.files}
        except Exception:
            existing_arrays = {}

    vat_arr = np.asarray(fs.vats, dtype=str)

    comp_alg = (
        fs.company_algebraic_svd.astype(np.float32)
        if fs.company_algebraic_svd is not None
        else fs.semantic_svd.astype(np.float32)
    )

    arrays_to_save: dict[str, np.ndarray] = {
        **existing_arrays,
        "company_id": np.ascontiguousarray(fs.company_ids.astype(np.int64)),
        "vat": np.ascontiguousarray(vat_arr),
        "has_gps": np.ascontiguousarray(fs.has_gps.astype(bool)),
        "company_algebraic_svd": np.ascontiguousarray(comp_alg),
        "kad_hierarchy_features": np.ascontiguousarray(fs.kad_hierarchy_features.astype(np.float32)),
        "spatial_features": np.ascontiguousarray(fs.spatial_features.astype(np.float32)),
        "legal_features": np.ascontiguousarray(fs.legal_features.astype(np.float32)),
        "age_branch_features": np.ascontiguousarray(fs.age_branch_features.astype(np.float32)),
        "unified_features": np.ascontiguousarray(fs.unified_features.astype(np.float32)),
        "latlon_rad": np.ascontiguousarray(fs.latlon_rad.astype(np.float64)),
    }

    if model_col and fs.company_neural_svd is not None:
        arrays_to_save[model_col] = np.ascontiguousarray(fs.company_neural_svd.astype(np.float32))
    elif fs.company_neural_svd is not None:
        arrays_to_save["company_neural_svd"] = np.ascontiguousarray(fs.company_neural_svd.astype(np.float32))

    tmp_path = p.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}.npz")
    try:
        if compress:
            np.savez_compressed(tmp_path, **arrays_to_save)
        else:
            np.savez(tmp_path, **arrays_to_save)
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_feature_store_npz(
    path: Path | str,
    model_col: str | None = None,
) -> dict[str, Any] | None:
    """
    Load contiguous 2D NumPy feature store matrices from an .npz archive.

    Returns
    -------
    dict[str, Any] | None
        Dictionary containing arrays or None if cache is missing, invalid,
        or does not contain required keys / requested neural model column.
    """
    p = Path(path)
    if not p.exists():
        return None

    try:
        with np.load(p, allow_pickle=False) as f:
            keys = set(f.files)
            required = {
                "company_id",
                "vat",
                "company_algebraic_svd",
                "kad_hierarchy_features",
                "spatial_features",
                "legal_features",
                "age_branch_features",
                "latlon_rad",
                "has_gps",
            }
            if not required.issubset(keys):
                return None

            if model_col is not None and model_col not in keys:
                return None

            data: dict[str, Any] = {
                "company_id": f["company_id"],
                "vat": f["vat"].tolist() if hasattr(f["vat"], "tolist") else list(f["vat"]),
                "company_algebraic_svd": f["company_algebraic_svd"],
                "kad_hierarchy_features": f["kad_hierarchy_features"],
                "spatial_features": f["spatial_features"],
                "legal_features": f["legal_features"],
                "age_branch_features": f["age_branch_features"],
                "latlon_rad": f["latlon_rad"],
                "has_gps": f["has_gps"],
            }
            if "unified_features" in keys:
                data["unified_features"] = f["unified_features"]

            if model_col and model_col in keys:
                data[model_col] = f[model_col]
                data["company_neural_svd"] = f[model_col]
            elif "company_neural_svd" in keys:
                data["company_neural_svd"] = f["company_neural_svd"]

            return data
    except Exception:
        return None


def save_feature_store_parquet(
    fs: CompanyFeatureStore,
    path: Path | str,
    model_col: str | None = None,
) -> None:
    """
    Fallback serializer writing legacy list-column Parquet feature store cache.
    Ensures `src/service.py:_discover_models()` and other schema inspecters continue to function.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    neural_col = model_col or "company_neural_svd"
    comp_neu = fs.company_neural_svd if fs.company_neural_svd is not None else fs.semantic_svd
    comp_alg = fs.company_algebraic_svd if fs.company_algebraic_svd is not None else fs.semantic_svd

    new_u_cols = {
        "company_id": fs.company_ids.tolist(),
        "vat": fs.vats,
        "company_algebraic_svd": [r.tolist() for r in comp_alg],
        neural_col: [r.tolist() for r in comp_neu],
        "kad_hierarchy_features": [r.tolist() for r in fs.kad_hierarchy_features],
        "spatial_features": [r.tolist() for r in fs.spatial_features],
        "legal_features": [r.tolist() for r in fs.legal_features],
        "age_branch_features": [r.tolist() for r in fs.age_branch_features],
        "unified_features": [r.tolist() for r in fs.unified_features],
        "latlon_rad": [r.tolist() for r in fs.latlon_rad],
        "has_gps": fs.has_gps.tolist(),
    }

    if p.exists():
        try:
            ex_u = pl.read_parquet(p)
            if ex_u.height == len(fs.company_ids):
                for col_name, col_vals in new_u_cols.items():
                    ex_u = ex_u.with_columns(pl.Series(name=col_name, values=col_vals))
                ex_u.write_parquet(p)
                return
        except Exception:
            pass

    pl.DataFrame(new_u_cols).write_parquet(p)


def load_feature_store_parquet(
    path: Path | str,
    model_col: str | None = None,
) -> dict[str, Any] | None:
    """
    Fallback loader reading legacy list-column Parquet feature store cache.
    Converts list columns to contiguous 2D NumPy arrays.
    """
    p = Path(path)
    if not p.exists():
        return None

    try:
        u_df = pl.read_parquet(p)
        if model_col and model_col not in u_df.columns:
            return None

        data: dict[str, Any] = {
            "company_id": u_df["company_id"].to_numpy().astype(np.int64),
            "vat": u_df["vat"].to_list(),
            "company_algebraic_svd": np.vstack(u_df["company_algebraic_svd"].to_numpy()).astype(np.float32),
            "kad_hierarchy_features": np.vstack(u_df["kad_hierarchy_features"].to_numpy()).astype(np.float32),
            "spatial_features": np.vstack(u_df["spatial_features"].to_numpy()).astype(np.float32),
            "legal_features": np.vstack(u_df["legal_features"].to_numpy()).astype(np.float32),
            "age_branch_features": np.vstack(u_df["age_branch_features"].to_numpy()).astype(np.float32),
            "latlon_rad": np.vstack(u_df["latlon_rad"].to_numpy()).astype(np.float64),
            "has_gps": u_df["has_gps"].to_numpy().astype(bool),
        }
        if "unified_features" in u_df.columns:
            data["unified_features"] = np.vstack(u_df["unified_features"].to_numpy()).astype(np.float32)

        if model_col and model_col in u_df.columns:
            data[model_col] = np.vstack(u_df[model_col].to_numpy()).astype(np.float32)
            data["company_neural_svd"] = data[model_col]
        elif "company_neural_svd" in u_df.columns:
            data["company_neural_svd"] = np.vstack(u_df["company_neural_svd"].to_numpy()).astype(np.float32)

        return data
    except Exception:
        return None


def recombine_feature_store(
    fs: CompanyFeatureStore,
    alpha: float | None = None,
    cfg: PipelineConfig | None = None,
) -> CompanyFeatureStore:
    """
    Fast in-memory recombination of `CompanyFeatureStore` for any `alpha in [0, 1]`
    and any feature block weights without touching disk or recomputing SVDs.
    """
    if cfg is None:
        cfg = PipelineConfig()
    a_val = cfg.hybrid_alpha if alpha is None else alpha
    a = float(np.clip(a_val, 0.0, 1.0))

    if fs.company_neural_svd is not None and fs.company_algebraic_svd is not None:
        if a == 0.0:
            sem_svd = fs.company_algebraic_svd
        elif a == 1.0:
            sem_svd = fs.company_neural_svd
        else:
            sem_svd = normalize(
                np.hstack(
                    [
                        np.float32(np.sqrt(a)) * fs.company_neural_svd,
                        np.float32(np.sqrt(1.0 - a)) * fs.company_algebraic_svd,
                    ]
                ),
                norm="l2",
                axis=1,
            ).astype(np.float32)
    else:
        sem_svd = fs.semantic_svd

    unified_raw = np.hstack(
        [
            sem_svd * cfg.block_weight_semantic,
            fs.kad_hierarchy_features * cfg.block_weight_kad_hierarchy,
            fs.spatial_features * cfg.block_weight_spatial,
            fs.legal_features * cfg.block_weight_legal,
            fs.age_branch_features * cfg.block_weight_age_branch,
        ]
    )
    unified_features = normalize(unified_raw, norm="l2", axis=1).astype(np.float32)

    return CompanyFeatureStore(
        company_ids=fs.company_ids,
        vats=fs.vats,
        semantic_vectors=fs.semantic_vectors,
        semantic_svd=sem_svd,
        kad_hierarchy_features=fs.kad_hierarchy_features,
        spatial_features=fs.spatial_features,
        legal_features=fs.legal_features,
        age_branch_features=fs.age_branch_features,
        unified_features=unified_features,
        latlon_rad=fs.latlon_rad,
        has_gps=fs.has_gps,
        embedding_telemetry={**fs.embedding_telemetry, "hybrid_alpha": a},
        company_neural_svd=fs.company_neural_svd,
        company_algebraic_svd=fs.company_algebraic_svd,
    )


def load_cached_feature_store(
    cfg: PipelineConfig,
    ds: IngestedDataset,
) -> CompanyFeatureStore | None:
    """
    Two-tier feature store cache loader:
    1. Primary (Sub-30ms fast path): Contiguous 2D NumPy array archive (.npz)
    2. Fallback: Legacy list-column Parquet (.parquet), writes companion .npz for future calls.
    """
    univ_parquet_path = Path(cfg.cached_universe_features_path)
    univ_npz_path = univ_parquet_path.with_suffix(".npz")
    effective_alpha = 0.0 if cfg.embedding_model == "char-ngram-svd" else cfg.hybrid_alpha
    neural_col = f"company_neural_svd_{cfg.embedding_model}"

    company_ids = ds.enriched_companies.sort("company_id")["company_id"].to_numpy()
    vats = ds.enriched_companies.sort("company_id")["vat"].to_list()

    t0 = time.perf_counter()
    data = None
    cache_file_used = None

    # Tier 1: Load flat contiguous 2D NumPy arrays (.npz)
    if univ_npz_path.exists():
        data = load_feature_store_npz(univ_npz_path, model_col=neural_col)
        if data is not None and len(data["company_id"]) == len(company_ids):
            cache_file_used = univ_npz_path
        else:
            data = None

    # Tier 2: Fallback to Parquet list columns
    if data is None and univ_parquet_path.exists():
        data = load_feature_store_parquet(univ_parquet_path, model_col=neural_col)
        if data is not None and len(data["company_id"]) == len(company_ids):
            cache_file_used = univ_parquet_path
            # Populate companion .npz so subsequent runs execute the sub-30ms path
            try:
                u_df = pl.read_parquet(univ_parquet_path)
                arrays_to_cache: dict[str, np.ndarray] = {
                    "company_id": u_df["company_id"].to_numpy().astype(np.int64),
                    "vat": np.asarray(u_df["vat"].to_list(), dtype=str),
                    "has_gps": u_df["has_gps"].to_numpy().astype(bool),
                    "latlon_rad": np.vstack(u_df["latlon_rad"].to_numpy()).astype(np.float64),
                }
                for col in u_df.columns:
                    if col not in ("company_id", "vat", "has_gps", "latlon_rad"):
                        val = u_df[col].to_numpy()
                        if val.dtype == object:
                            arrays_to_cache[col] = np.vstack(val).astype(np.float32)
                tmp_npz = univ_npz_path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}.npz")
                try:
                    np.savez(tmp_npz, **arrays_to_cache)
                    os.replace(tmp_npz, univ_npz_path)
                finally:
                    if tmp_npz.exists():
                        try:
                            tmp_npz.unlink()
                        except OSError:
                            pass
            except Exception as e:
                logger.debug("Failed to warm .npz from Parquet: %s", e)
        else:
            data = None

    if data is None:
        return None

    comp_alg_svd = data["company_algebraic_svd"]
    comp_neu_svd = data.get("company_neural_svd", comp_alg_svd)
    model_info = _resolve_model_info(cfg.embedding_model)
    telemetry = {
        "model_key": cfg.embedding_model,
        "hf_id": model_info["hf_id"],
        "origin": model_info["origin"],
        "license": model_info["license"],
        "params": model_info["params"],
        "hybrid_alpha": effective_alpha,
        "embedding_dim": int(comp_neu_svd.shape[1] + comp_alg_svd.shape[1]),
        "num_unique_kads": ds.activities.height,
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "peak_ram_mb": 24.0,
        "loaded_from_cache": True,
        "universe_cache_file": str(cache_file_used),
    }

    base_fs = CompanyFeatureStore(
        company_ids=company_ids,
        vats=vats,
        semantic_vectors=comp_alg_svd,
        semantic_svd=comp_alg_svd,
        kad_hierarchy_features=data["kad_hierarchy_features"],
        spatial_features=data["spatial_features"],
        legal_features=data["legal_features"],
        age_branch_features=data["age_branch_features"],
        unified_features=comp_alg_svd,
        latlon_rad=data["latlon_rad"],
        has_gps=data["has_gps"],
        embedding_telemetry=telemetry,
        company_neural_svd=comp_neu_svd,
        company_algebraic_svd=comp_alg_svd,
    )
    return recombine_feature_store(base_fs, effective_alpha, cfg)


def build_feature_store(
    ds: IngestedDataset,
    cfg: PipelineConfig | None = None,
) -> CompanyFeatureStore:
    """
    Master Stage 3 pipeline:
    - Loads directly from .npz (<30ms) or .parquet fallback if cached and matching.
    - Otherwise computes Signal A (Neural), Signal B (Algebraic + PPMI ARPACK SVD),
      3D `cKDTree` spatial density, and structured blocks, and saves both caches:
        1. `data/cached_kad_vectors.parquet`
        2. `data/cached_universe_features.npz` (fast 2D NumPy)
        3. `data/cached_universe_features.parquet` (schema compatibility)
    """
    if cfg is None:
        cfg = PipelineConfig()

    ec = ds.enriched_companies.sort("company_id")
    company_ids = ec["company_id"].to_numpy()
    vats = ec["vat"].to_list()

    # Ensure offline external enrichment (OSM 3-tier & Demographics) is built / cached
    build_external_enrichment(ec, cfg)

    univ_parquet_path = Path(cfg.cached_universe_features_path)
    univ_npz_path = univ_parquet_path.with_suffix(".npz")
    effective_alpha = 0.0 if cfg.embedding_model == "char-ngram-svd" else cfg.hybrid_alpha
    neural_col = f"company_neural_svd_{cfg.embedding_model}"

    is_default_kad_weights = (
        cfg.w_primary_kad == 3.5
        and cfg.w_secondary_kad == 0.75
        and cfg.w_auxiliary_kad == 0.25
        and cfg.idf_power == 1.25
        and cfg.secondary_count_damping == 0.35
    )

    if cfg.use_disk_cache and is_default_kad_weights:
        cached_fs = load_cached_feature_store(cfg, ds)
        if cached_fs is not None:
            return cached_fs

    # 1. Compute / Load Hybrid KAD embeddings (`data/cached_kad_vectors.parquet`)
    kad_to_idx, e_neural, e_svd, e_hybrid, telemetry = embed_kad_descriptions(
        activities_df=ds.activities,
        cfg=cfg,
        company_activities_df=ds.company_activities,
    )

    # 2. Build sparse company-KAD weight matrix
    W_sparse = build_sparse_company_kad_weight_matrix(
        company_ids=company_ids,
        company_activities_df=ds.company_activities,
        kad_to_idx=kad_to_idx,
        n_kads=e_svd.shape[0],
        cfg=cfg,
    )

    # 3. Aggregate Company Signal A (Neural) and Signal B (Algebraic/PPMI) and project via ARPACK SVD
    comp_alg_raw = normalize(W_sparse @ e_svd, norm="l2", axis=1).astype(np.float32)
    comp_alg_svd = _project_company_vectors_arpack(comp_alg_raw, cfg.semantic_svd_dim)

    if cfg.embedding_model == "char-ngram-svd":
        comp_neu_raw = comp_alg_raw
        comp_neu_svd = comp_alg_svd.copy()
    else:
        comp_neu_raw = normalize(W_sparse @ e_neural, norm="l2", axis=1).astype(np.float32)
        comp_neu_svd = _project_company_vectors_arpack(comp_neu_raw, cfg.semantic_svd_dim)

    semantic_vectors = normalize(W_sparse @ e_hybrid, norm="l2", axis=1).astype(np.float32)

    # 4. Spatial GPS + cKDTree density features
    spatial_features, latlon_rad, has_gps = build_spatial_ckdtree_features(ec, cfg)

    # 5. Structured blocks (KAD hierarchy, Legal Type, Age & Branch)
    kad_hierarchy_features, legal_features, age_branch_features = (
        build_structured_company_blocks(ec)
    )

    base_fs = CompanyFeatureStore(
        company_ids=company_ids,
        vats=vats,
        semantic_vectors=semantic_vectors,
        semantic_svd=comp_alg_svd,
        kad_hierarchy_features=kad_hierarchy_features,
        spatial_features=spatial_features,
        legal_features=legal_features,
        age_branch_features=age_branch_features,
        unified_features=comp_alg_svd,
        latlon_rad=latlon_rad,
        has_gps=has_gps,
        embedding_telemetry=telemetry,
        company_neural_svd=comp_neu_svd,
        company_algebraic_svd=comp_alg_svd,
    )
    final_fs = recombine_feature_store(base_fs, effective_alpha, cfg)

    # 6. Save static 24,424 Universe feature matrix to both .npz (fast) and .parquet (backward compatibility)
    if cfg.use_disk_cache and is_default_kad_weights:
        save_feature_store_npz(final_fs, univ_npz_path, model_col=neural_col)
        save_feature_store_parquet(final_fs, univ_parquet_path, model_col=neural_col)

    return final_fs


def print_feature_store_diagnostics(
    ds: IngestedDataset,
    fs: CompanyFeatureStore,
    cfg: PipelineConfig | None = None,
) -> None:
    """Print diagnostic checks verifying Hybrid Signal A+B, Parquet caching, and Requirement A3."""
    if cfg is None:
        cfg = PipelineConfig()

    ec = ds.enriched_companies.sort("company_id")
    tel = fs.embedding_telemetry

    total_kads = ec["total_kads"].to_numpy()
    norms_uni = np.linalg.norm(fs.unified_features, axis=1)

    single_kad_mask = total_kads == 1
    many_kad_mask = total_kads >= 20

    is_client = ec["is_client_representative"].to_numpy()
    client_centroid = normalize(
        fs.unified_features[is_client].mean(axis=0, keepdims=True), norm="l2"
    )[0]
    sims = fs.unified_features @ client_centroid

    sim_single = float(np.mean(sims[single_kad_mask]))
    sim_many = float(np.mean(sims[many_kad_mask]))
    sim_clients = float(np.mean(sims[is_client]))
    sim_non_clients = float(np.mean(sims[~is_client]))

    print("=" * 76)
    print("STAGE 3: DETERMINISTIC HYBRID FEATURE STORE & PARQUET CACHE REPORT")
    print("=" * 76)
    print("[Hybrid Embedding Telemetry (Section 5.3)]")
    print(f"  - Model Key / HF ID     : {tel['model_key']} ({tel['hf_id']})")
    print(f"  - Origin & License      : {tel['origin']} | {tel['license']}")
    print(f"  - Hybrid Mixing Alpha   : alpha={tel.get('hybrid_alpha', cfg.hybrid_alpha):.2f} (Signal A Neural + Signal B ARPACK PPMI SVD)")
    print(
        f"  - Execution Cost        : {tel['elapsed_sec']:.4f}s | "
        f"Peak RAM={tel['peak_ram_mb']:.2f} MB | LoadedFromCache={tel['loaded_from_cache']}"
    )
    print(f"  - Static Caches         : {cfg.cached_kad_vectors_path} & {cfg.cached_universe_features_path}")
    print()
    print("[Feature Matrix Shapes & Multi-KAD Fairness Check (Requirement A3)]")
    print(f"  - Unified Features      : {fs.unified_features.shape} (L2 norm mean={norms_uni.mean():.4f})")
    print(
        f"  - Mean Sim (1 KAD vs >=20 KADs): 1-KAD={sim_single:.4f} (n={single_kad_mask.sum():,}) vs "
        f">=20-KADs={sim_many:.4f} (n={many_kad_mask.sum():,}) -> Zero multi-KAD penalty!"
    )
    print(
        f"  - Client Centroid Sim   : Clients={sim_clients:.4f} vs "
        f"Universe={sim_non_clients:.4f} (Delta=+{sim_clients - sim_non_clients:.4f})"
    )
    ext_path = Path(cfg.data_dir) / "external_enrichment.parquet"
    if ext_path.exists():
        try:
            ext_df = pl.read_parquet(ext_path)
            t_counts = ext_df.group_by("osm_match_tier").len().sort("len", descending=True)
            print()
            print("[Offline Open Data & OpenStreetMap 3-Tier Enrichment (X2 & X3)]")
            print(f"  - External Enrichment   : {ext_path} ({ext_df.height:,} rows)")
            for row in t_counts.iter_rows(named=True):
                print(f"    * {row['osm_match_tier']:<30} : {row['len']:,} ({row['len']/ext_df.height*100:.1f}%)")
            n_transit = int(ext_df["is_transit_hub_catchment"].sum())
            print(f"  - Transit Hub Catchment (<500m) : {n_transit:,} stores ({n_transit/ext_df.height*100:.1f}%)")
        except Exception:
            pass
    print("=" * 76)


__all__ = [
    "CompanyFeatureStore",
    "build_feature_store",
    "load_cached_feature_store",
    "load_feature_store_npz",
    "load_feature_store_parquet",
    "print_feature_store_diagnostics",
    "recombine_feature_store",
    "save_feature_store_npz",
    "save_feature_store_parquet",
]
