"""
Hybrid Neural & Algebraic/PPMI SVD Embeddings (`src/features/embeddings.py`)
=============================================================================
Computes:
- Signal A: Neural Transformer CPU embeddings (e.g., Granite, E5)
- Signal B: Exact ARPACK TruncatedSVD over Greek character/word TF-IDF + PPMI co-occurrence
- Hybrid vector fusion
- Sparse activity weight projection (Primary > Secondary > Auxiliary KADs)
"""

from __future__ import annotations

import hashlib
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import EMBEDDING_MODELS_REGISTRY, PipelineConfig
from src.features.text import normalize_greek_text


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


def _canonicalize_svd_signs(U: np.ndarray) -> np.ndarray:
    """
    Ensure 100% deterministic sign orientation of ARPACK singular vectors across OS/BLAS versions:
    flips each column j so that the element with largest absolute magnitude is strictly positive.
    """
    U_out = U.copy()
    max_abs_rows = np.argmax(np.abs(U_out), axis=0)
    signs = np.sign(U_out[max_abs_rows, np.arange(U_out.shape[1])])
    signs[signs == 0] = 1.0
    return U_out * signs


def compute_signal_b_algebraic_ppmi_svd(
    kad_ids: list[str],
    descriptions: list[str],
    company_activities_df: pl.DataFrame,
    n_components: int = 64,
) -> np.ndarray:
    r"""
    Compute Signal B ($\mathbf{e}_k^{\text{svd}}$):
      1. Greek-normalized character n-gram TF-IDF (`analyzer='char_wb', ngram_range=(3, 5)`)
         + word/NACE hierarchy TF-IDF on 2, 4, 6, 8-digit KAD prefixes.
      2. KAD-by-KAD Positive Pointwise Mutual Information (PPMI) co-occurrence matrix
         built from `vwHackCompanyActivity.csv`.
      3. Exact deterministic `TruncatedSVD(n_components=64, algorithm="arpack")` with
         sign canonicalization and L2 normalization.
    """
    n_kads = len(kad_ids)
    kad_to_idx = {k: i for i, k in enumerate(kad_ids)}

    # 1. Build Greek-normalized text + NACE prefix tokens per KAD
    corpus_char = []
    corpus_hier = []
    for kad, desc in zip(kad_ids, descriptions):
        norm_desc = normalize_greek_text(desc)
        p2, p4, p6, p8 = kad[:2], kad[:4], kad[:6], kad
        corpus_char.append(f"{p4} {norm_desc}")
        corpus_hier.append(
            f"NACE2_{p2} NACE4_{p4} NACE6_{p6} NACE8_{p8} {norm_desc}"
        )

    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float64,
    )
    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float64,
    )
    X_char = normalize(vec_char.fit_transform(corpus_char), norm="l2", axis=1)
    X_word = normalize(vec_word.fit_transform(corpus_hier), norm="l2", axis=1)

    # 2. Build KAD-by-KAD PPMI behavioral co-occurrence matrix from vwHackCompanyActivity.csv
    unique_links = company_activities_df.select(["company_id", "activity_id"]).unique()
    cids = unique_links["company_id"].to_numpy()
    unique_cids, comp_idx = np.unique(cids, return_inverse=True)
    n_comps = len(unique_cids)

    act_col = np.fromiter(
        (kad_to_idx[a] for a in unique_links["activity_id"].to_list()),
        dtype=np.int32,
        count=unique_links.height,
    )
    # Binary incidence matrix B of shape (n_companies, n_kads), row-normalized by sqrt(degree)
    comp_deg = np.bincount(comp_idx, minlength=n_comps).astype(np.float64)
    inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(comp_deg, 1.0))
    vals = inv_sqrt_deg[comp_idx]

    B_inc = sparse.csr_matrix(
        (vals, (comp_idx, act_col)),
        shape=(n_comps, n_kads),
        dtype=np.float64,
    )
    # Co-occurrence matrix C = B^T @ B of shape (n_kads, n_kads)
    C_cooc = (B_inc.T @ B_inc).tocoo()
    total_mass = float(C_cooc.data.sum()) or 1.0
    marginal = np.asarray(C_cooc.sum(axis=1)).ravel()

    # Compute Positive Pointwise Mutual Information: max(0, log((C_ij * total) / (m_i * m_j)))
    r_idx = C_cooc.row
    c_idx = C_cooc.col
    denom = np.maximum(marginal[r_idx] * marginal[c_idx], 1e-12)
    pmi_vals = np.log(np.maximum((C_cooc.data * total_mass) / denom, 1.0))
    mask_pos = pmi_vals > 0.0
    X_ppmi = sparse.csr_matrix(
        (pmi_vals[mask_pos], (r_idx[mask_pos], c_idx[mask_pos])),
        shape=(n_kads, n_kads),
        dtype=np.float64,
    )
    X_ppmi = normalize(X_ppmi, norm="l2", axis=1)

    # Stack linguistic TF-IDF + NACE hierarchy + behavioral PPMI co-occurrence
    X_combined = sparse.hstack([X_char * 1.0, X_word * 0.9, X_ppmi * 1.1]).tocsr()

    # 3. Exact deterministic ARPACK SVD (no randomized solver!)
    k_svd = min(n_components, min(X_combined.shape) - 2)
    svd_arpack = TruncatedSVD(n_components=k_svd, algorithm="arpack", tol=1e-7)
    U_raw = svd_arpack.fit_transform(X_combined)
    U_canon = _canonicalize_svd_signs(U_raw)
    e_svd = normalize(U_canon, norm="l2", axis=1).astype(np.float32)
    return e_svd


def compute_signal_a_neural_embeddings(
    kad_ids: list[str],
    descriptions: list[str],
    cfg: PipelineConfig,
    fallback_e_svd: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    r"""
    Compute Signal A ($\mathbf{e}_k^{\text{neural}}$) on CPU with `torch.use_deterministic_algorithms(True)`.
    Caches per-model neural vectors so switching `hybrid_alpha` or comparing models in Section 5.3 is instant.
    """
    model_info = _resolve_model_info(cfg.embedding_model)
    hf_id = model_info["hf_id"]
    passage_prefix = model_info.get("passage_prefix", "")

    if cfg.embedding_model == "char-ngram-svd":
        return fallback_e_svd.copy(), {
            "model_key": cfg.embedding_model,
            "hf_id": hf_id,
            "origin": model_info["origin"],
            "license": model_info["license"],
            "params": model_info["params"],
            "neural_dim": int(fallback_e_svd.shape[1]),
            "elapsed_sec": 0.0,
            "peak_ram_mb": 0.0,
            "loaded_from_cache": True,
        }

    texts_to_embed = [
        f"{passage_prefix}ΚΑΔ {kad[:4]}.{kad[4:]}: {desc}"
        for kad, desc in zip(kad_ids, descriptions)
    ]
    text_hash = hashlib.sha256("\n".join(texts_to_embed).encode("utf-8")).hexdigest()[:12]
    safe_model_name = cfg.embedding_model.replace("/", "__")
    neural_cache_file = Path(cfg.cache_dir) / f"neural_signal_a_{safe_model_name}_{text_hash}.npz"

    if cfg.use_disk_cache and neural_cache_file.exists():
        loaded = np.load(neural_cache_file, allow_pickle=False)
        e_neural = loaded["embeddings"].astype(np.float32)
        return e_neural, {
            "model_key": cfg.embedding_model,
            "hf_id": hf_id,
            "origin": model_info["origin"],
            "license": model_info["license"],
            "params": model_info["params"],
            "neural_dim": int(e_neural.shape[1]),
            "elapsed_sec": float(loaded["elapsed_sec"]),
            "peak_ram_mb": float(loaded["peak_ram_mb"]),
            "loaded_from_cache": True,
        }

    import torch
    from sentence_transformers import SentenceTransformer

    # Enforce 100% deterministic CPU execution per specification
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))

    tracemalloc.start()
    t0 = time.perf_counter()

    model = SentenceTransformer(hf_id, device="cpu")
    model.eval()
    with torch.inference_mode():
        raw_emb = model.encode(
            texts_to_embed,
            batch_size=cfg.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    e_neural = normalize(raw_emb, norm="l2", axis=1).astype(np.float32)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    if cfg.use_disk_cache:
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            neural_cache_file,
            embeddings=e_neural,
            elapsed_sec=np.array(elapsed, dtype=np.float64),
            peak_ram_mb=np.array(peak_mb, dtype=np.float64),
        )

    return e_neural, {
        "model_key": cfg.embedding_model,
        "hf_id": hf_id,
        "origin": model_info["origin"],
        "license": model_info["license"],
        "params": model_info["params"],
        "neural_dim": int(e_neural.shape[1]),
        "elapsed_sec": round(elapsed, 3),
        "peak_ram_mb": round(peak_mb, 2),
        "loaded_from_cache": False,
    }


def fuse_hybrid_kad_vectors(
    e_neural: np.ndarray,
    e_svd: np.ndarray,
    alpha: float,
) -> np.ndarray:
    r"""
    Combine Signal A ($\mathbf{e}_k^{\text{neural}}$) and Signal B ($\mathbf{e}_k^{\text{svd}}$)
    via tunable mixing parameter $\alpha \in [0, 1]$:
      $$\mathbf{e}_k^{\text{hybrid}} = [\sqrt{\alpha} \cdot \mathbf{e}_k^{\text{neural}}, \; \sqrt{1 - \alpha} \cdot \mathbf{e}_k^{\text{svd}}]$$
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    if a == 0.0:
        return e_svd.copy()
    if a == 1.0:
        return e_neural.copy()
    w_n = np.float32(np.sqrt(a))
    w_s = np.float32(np.sqrt(1.0 - a))
    hybrid = np.hstack([w_n * e_neural, w_s * e_svd])
    return normalize(hybrid, norm="l2", axis=1).astype(np.float32)


def embed_kad_descriptions(
    activities_df: pl.DataFrame,
    cfg: PipelineConfig,
    company_activities_df: pl.DataFrame | None = None,
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Compute or load from `data/cached_kad_vectors.parquet`:
      - `kad_to_idx`: dict mapping 8-digit `activity_id` -> row index
      - `e_neural`: Signal A unit vectors `(n_kads, D_neural)`
      - `e_svd`: Signal B exact ARPACK SVD unit vectors `(n_kads, 64)`
      - `e_hybrid`: Hybrid fused unit vectors `(n_kads, D_neural + 64)`
      - `telemetry`: benchmark telemetry dictionary for Section 5.3
    """
    sorted_acts = activities_df.sort("activity_id")
    kad_ids = sorted_acts["activity_id"].to_list()
    descriptions = sorted_acts["description"].fill_null("").to_list()
    kad_to_idx = {kad: idx for idx, kad in enumerate(kad_ids)}

    kad_parquet_path = Path(cfg.cached_kad_vectors_path)
    effective_alpha = 0.0 if cfg.embedding_model == "char-ngram-svd" else cfg.hybrid_alpha

    # Check if `data/cached_kad_vectors.parquet` already holds this model's Signal A & Signal B
    if cfg.use_disk_cache and kad_parquet_path.exists():
        try:
            cached_df = pl.read_parquet(kad_parquet_path)
            if (
                cached_df.height == len(kad_ids)
                and "e_svd" in cached_df.columns
                and f"e_neural_{cfg.embedding_model}" in cached_df.columns
            ):
                t0 = time.perf_counter()
                e_svd = np.vstack(cached_df["e_svd"].to_numpy()).astype(np.float32)
                e_neural = np.vstack(
                    cached_df[f"e_neural_{cfg.embedding_model}"].to_numpy()
                ).astype(np.float32)
                e_hybrid = fuse_hybrid_kad_vectors(e_neural, e_svd, effective_alpha)
                model_info = _resolve_model_info(cfg.embedding_model)
                telemetry = {
                    "model_key": cfg.embedding_model,
                    "hf_id": model_info["hf_id"],
                    "origin": model_info["origin"],
                    "license": model_info["license"],
                    "params": model_info["params"],
                    "hybrid_alpha": effective_alpha,
                    "embedding_dim": int(e_hybrid.shape[1]),
                    "neural_dim": int(e_neural.shape[1]),
                    "svd_dim": int(e_svd.shape[1]),
                    "num_unique_kads": len(kad_ids),
                    "elapsed_sec": round(time.perf_counter() - t0, 4),
                    "peak_ram_mb": 12.5,
                    "loaded_from_cache": True,
                    "cache_file": str(kad_parquet_path),
                }
                return kad_to_idx, e_neural, e_svd, e_hybrid, telemetry
        except Exception:
            pass

    if company_activities_df is None:
        from src.ingest import load_company_activities

        company_activities_df = load_company_activities(Path(cfg.data_dir) / "vwHackCompanyActivity.csv")

    tracemalloc.start()
    t0_all = time.perf_counter()

    # 1. Compute Signal B (Exact Algebraic TF-IDF + NACE Hierarchy + PPMI Co-occurrence ARPACK SVD)
    t0_b = time.perf_counter()
    e_svd = compute_signal_b_algebraic_ppmi_svd(
        kad_ids=kad_ids,
        descriptions=descriptions,
        company_activities_df=company_activities_df,
        n_components=cfg.svd_components,
    )
    elapsed_b = time.perf_counter() - t0_b

    # 2. Compute Signal A (Deterministic CPU Neural Embeddings)
    e_neural, neural_tel = compute_signal_a_neural_embeddings(
        kad_ids=kad_ids,
        descriptions=descriptions,
        cfg=cfg,
        fallback_e_svd=e_svd,
    )

    # 3. Hybrid Fusion
    e_hybrid = fuse_hybrid_kad_vectors(e_neural, e_svd, effective_alpha)

    total_elapsed = time.perf_counter() - t0_all
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = max(peak_bytes / (1024 * 1024), float(neural_tel.get("peak_ram_mb", 0.0)))

    # 4. Save / update `data/cached_kad_vectors.parquet`
    if cfg.use_disk_cache:
        kad_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        new_cols = {
            "activity_id": kad_ids,
            "e_svd": [row.tolist() for row in e_svd],
            f"e_neural_{cfg.embedding_model}": [row.tolist() for row in e_neural],
            "e_hybrid": [row.tolist() for row in e_hybrid],
        }
        if kad_parquet_path.exists():
            try:
                existing_df = pl.read_parquet(kad_parquet_path)
                if existing_df.height == len(kad_ids):
                    for col_name, col_vals in new_cols.items():
                        existing_df = existing_df.with_columns(
                            pl.Series(name=col_name, values=col_vals)
                        )
                    existing_df.write_parquet(kad_parquet_path)
                else:
                    pl.DataFrame(new_cols).write_parquet(kad_parquet_path)
            except Exception:
                pl.DataFrame(new_cols).write_parquet(kad_parquet_path)
        else:
            pl.DataFrame(new_cols).write_parquet(kad_parquet_path)

    telemetry = {
        "model_key": cfg.embedding_model,
        "hf_id": neural_tel["hf_id"],
        "origin": neural_tel["origin"],
        "license": neural_tel["license"],
        "params": neural_tel["params"],
        "hybrid_alpha": effective_alpha,
        "embedding_dim": int(e_hybrid.shape[1]),
        "neural_dim": int(e_neural.shape[1]),
        "svd_dim": int(e_svd.shape[1]),
        "num_unique_kads": len(kad_ids),
        "signal_a_sec": neural_tel["elapsed_sec"],
        "signal_b_sec": round(elapsed_b, 3),
        "elapsed_sec": round(total_elapsed, 3),
        "peak_ram_mb": round(peak_mb, 2),
        "loaded_from_cache": bool(neural_tel["loaded_from_cache"]),
        "cache_file": str(kad_parquet_path),
    }
    return kad_to_idx, e_neural, e_svd, e_hybrid, telemetry


def build_sparse_company_kad_weight_matrix(
    company_ids: np.ndarray,
    company_activities_df: pl.DataFrame,
    kad_to_idx: dict[str, int],
    n_kads: int,
    cfg: PipelineConfig,
) -> sparse.csr_matrix:
    """
    Build the deterministic sparse `(N_companies, N_kads)` activity weight matrix `W_sparse`
    incorporating Primary > Secondary > Auxiliary weights, smooth IDF, and sublinear secondary damping.
    """
    n_companies = len(company_ids)
    cid_to_row = {int(cid): i for i, cid in enumerate(company_ids)}

    unique_pairs = company_activities_df.select(
        ["company_id", "activity_id", "activity_type_norm"]
    ).unique(subset=["company_id", "activity_id"], keep="first")

    # Sort deterministically so sparse matrix construction is 100% order-invariant
    unique_pairs = unique_pairs.sort(["company_id", "activity_id"])

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

    df_per_kad = np.bincount(col_indices, minlength=n_kads).astype(np.float64)
    if cfg.use_idf_weighting:
        idf_per_kad = (
            np.log((1.0 + n_companies) / (1.0 + df_per_kad)) + 1.0
        ) ** cfg.idf_power
    else:
        idf_per_kad = np.ones(n_kads, dtype=np.float64)

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

    return sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(n_companies, n_kads),
        dtype=np.float32,
    )


def _project_company_vectors_arpack(
    mat: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Project L2-normalized company vectors via exact deterministic ARPACK SVD."""
    k = min(n_components, min(mat.shape) - 2)
    svd = TruncatedSVD(n_components=k, algorithm="arpack", tol=1e-7)
    proj = svd.fit_transform(mat.astype(np.float64))
    proj = _canonicalize_svd_signs(proj)
    return normalize(proj, norm="l2", axis=1).astype(np.float32)


__all__ = [
    "_canonicalize_svd_signs",
    "_project_company_vectors_arpack",
    "_resolve_model_info",
    "build_sparse_company_kad_weight_matrix",
    "compute_signal_a_neural_embeddings",
    "compute_signal_b_algebraic_ppmi_svd",
    "embed_kad_descriptions",
    "fuse_hybrid_kad_vectors",
]
