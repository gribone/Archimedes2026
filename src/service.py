"""
Interactive Engine Service (`src/service.py`)
=============================================
Stateful, thread-safe wrapper around the existing pipeline modules for the JavaScript web UI.

Speed strategy (what makes slider changes feel instant):
1. Ingestion, the Parquet feature store and per-alpha recombinations are loaded once and kept in RAM.
2. The expensive clustering stage (K selection with `n_init=200` + 200 bootstrap resamples) is memoized
   inside `src/icp_engine.py`; changing only scoring weights, the EB prior, the spatial bandwidth or the
   NO_FIT threshold re-uses the cached clusters and only re-scores the 24,424 companies.
3. Changes that *do* require re-clustering (alpha, K range, turnover weighting, client file) run in
   "fast" precision (`n_init=20`, 30 bootstrap resamples) and can be upgraded to full precision on demand.
   Exports always run at full precision.
4. Requests carry a generation number; stale requests queued behind a running computation are dropped.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src import icp_engine
from src.config import EMBEDDING_MODELS_REGISTRY, PipelineConfig, load_optimal_weights_dict
from src.features import CompanyFeatureStore, build_feature_store, recombine_feature_store
from src.icp_engine import ICPEngineResult, compute_clients_sha256, export_deliverables, run_icp_engine
from src.ingest import IngestedDataset, ingest_all, load_clients_excel
from src.verifier import get_verifier
from src.llm_explainer import (
    AI_DISCLOSURE_PREFIX_EL,
    EU_AI_ACT_FOOTER_EL,
    build_grounded_prompt_el,
    check_local_ollama_status,
    enrich_profiles_with_local_llm,
    validate_grounded_llm_output,
)

FAST_N_INIT = 20
FAST_N_BOOT = 30
EXPORT_NAMES = ("profiles.json", "assignments.csv", "lookalikes.csv")

TURNOVER_MODES = [
    {"value": "log_turnover", "label": "Λογαριθμική (ln κύκλου εργασιών)", "label_en": "Logarithmic (ln revenue)"},
    {"value": "sqrt_turnover", "label": "Τετραγωνική ρίζα κύκλου εργασιών", "label_en": "Square Root of Revenue"},
    {"value": "linear_turnover", "label": "Γραμμική", "label_en": "Linear"},
    {"value": "uniform", "label": "Ισόβαρη (χωρίς κύκλο εργασιών)", "label_en": "Uniform (Unweighted)"},
]

# Specs drive both server-side validation and the analyst panel in the web UI.
# effect = "instant": re-scoring only (cached clusters) | "recluster": requires re-running K-Means.
PARAM_SPECS: list[dict[str, Any]] = [
    {"key": "use_embeddings", "group": "model", "type": "toggle", "effect": "recluster",
     "label": "Χρήση Μοντέλων Embeddings", "label_en": "Enable Neural Embedding Models",
     "help": "Ενεργοποίηση νευρωνικών μοντέλων embeddings (Granite, E5). Όταν απενεργοποιηθεί, χρησιμοποιείται αποκλειστικά το αλγεβρικό σήμα (TF-IDF + PPMI SVD).",
     "help_en": "Enable neural embedding models (Granite, E5). When disabled, only the algebraic signal (TF-IDF + PPMI SVD) is used."},
    {"key": "embedding_model", "group": "model", "type": "select", "effect": "recluster",
     "label": "Μοντέλο embeddings ΚΑΔ", "label_en": "KAD Embedding Model",
     "help": "Τοπικό μοντέλο αναπαράστασης των περιγραφών ΚΑΔ.",
     "help_en": "Local embedding model for KAD activity descriptions."},
    {"key": "hybrid_alpha", "group": "model", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05,
     "effect": "recluster", "label": "Hybrid α", "label_en": "Hybrid α Mixing Weight",
     "help": "0 = αλγεβρικό σήμα (TF-IDF + PPMI), 1 = νευρωνικό σήμα.",
     "help_en": "0 = algebraic signal (TF-IDF + PPMI), 1 = neural embedding signal."},
    {"key": "min_icps", "group": "cluster", "type": "range", "min": 1, "max": 6, "step": 1,
     "effect": "recluster", "label": "Ελάχιστος αριθμός προφίλ (K)", "label_en": "Minimum ICP Clusters (K)",
     "help": "Κάτω όριο για τον αριθμό των ICP.", "help_en": "Lower bound for the number of discovered ICP clusters."},
    {"key": "max_icps", "group": "cluster", "type": "range", "min": 1, "max": 6, "step": 1,
     "effect": "recluster", "label": "Μέγιστος αριθμός προφίλ (K)", "label_en": "Maximum ICP Clusters (K)",
     "help": "Άνω όριο για τον αριθμό των ICP.", "help_en": "Upper bound for the number of discovered ICP clusters."},
    {"key": "turnover_weighting_mode", "group": "cluster", "type": "select", "effect": "recluster",
     "label": "Στάθμιση Κύκλου Εργασιών", "label_en": "Revenue Weighting Mode",
     "help": "Πώς επηρεάζει ο κύκλος εργασιών κάθε πελάτη τη διαμόρφωση των προφίλ.",
     "help_en": "How each client's revenue weights ICP cluster formation.",
     "options": TURNOVER_MODES},
    {"key": "turnover_weight_alpha", "group": "cluster", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05,
     "effect": "recluster", "label": "Ένταση Στάθμισης Κύκλου Εργασιών", "label_en": "Revenue Weighting Intensity",
     "help": "0 = όλοι οι πελάτες ισοβαρείς.", "help_en": "0 = all clients weighted equally."},
    {"key": "sim_weight_feature_cosine", "group": "score", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05,
     "effect": "instant", "label": "Βάρος Ομοιότητας Προφίλ", "label_en": "Profile Similarity Weight",
     "help": "Συνημιτονική ομοιότητα με το κέντρο του ICP.", "help_en": "Cosine similarity to the ICP centroid."},
    {"key": "sim_weight_eb_lift_affinity", "group": "score", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05,
     "effect": "instant", "label": "Βάρος Στατιστικών Χαρακτηριστικών", "label_en": "Statistical Trait Lift Weight",
     "help": "Empirical Bayes lift σε Νομό, Δήμο, ΚΑΔ, νομική μορφή.", "help_en": "Empirical Bayes lift across Prefecture, Municipality, KAD, and Legal Form."},
    {"key": "sim_weight_spatial_kernel", "group": "score", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05,
     "effect": "instant", "label": "Βάρος Γεωγραφικής Εγγύτητας", "label_en": "Spatial Proximity Weight",
     "help": "Πυκνότητα πελατών γύρω από κάθε επιχείρηση.", "help_en": "Client density around each candidate business."},
    {"key": "spatial_kernel_bandwidth_km", "group": "score", "type": "range", "min": 5.0, "max": 100.0, "step": 5.0,
     "effect": "instant", "label": "Ακτίνα Εγγύτητας (km)", "label_en": "Proximity Bandwidth (km)",
     "help": "Εύρος του γεωγραφικού πυρήνα Gauss.", "help_en": "Gaussian spatial kernel bandwidth in kilometers."},
    {"key": "eb_prior_strength", "group": "score", "type": "range", "min": 1.0, "max": 30.0, "step": 1.0,
     "effect": "instant", "label": "Empirical Bayes Prior", "label_en": "Empirical Bayes Prior Strength",
     "help": "Μεγαλύτερη τιμή = πιο συντηρητικά lifts για σπάνια χαρακτηριστικά.", "help_en": "Higher values apply stronger shrinkage to rare traits."},
    {"key": "no_fit_min_score", "group": "score", "type": "range", "min": 0.15, "max": 0.65, "step": 0.01,
     "effect": "instant", "label": "Κατώφλι NO_FIT", "label_en": "NO_FIT Score Threshold",
     "help": "Πελάτες κάτω από αυτή την ταύτιση χαρακτηρίζονται NO_FIT.", "help_en": "Clients below this similarity score are flagged as NO_FIT."},
]
SPEC_BY_KEY = {s["key"]: s for s in PARAM_SPECS}


@dataclass(frozen=True)
class RunParams:
    client_set: str
    use_embeddings: bool
    embedding_model: str
    hybrid_alpha: float
    min_icps: int
    max_icps: int
    turnover_weighting_mode: str
    turnover_weight_alpha: float
    sim_weight_feature_cosine: float
    sim_weight_eb_lift_affinity: float
    sim_weight_spatial_kernel: float
    spatial_kernel_bandwidth_km: float
    eb_prior_strength: float
    no_fit_min_score: float
    precision: str = "fast"


CFG_KEYS = tuple(f.name for f in fields(RunParams) if f.name not in ("client_set", "precision"))


class Superseded(Exception):
    """Raised when a newer run request arrived while this one was queued."""


def _clean(o: Any) -> Any:
    """Make an object strictly JSON-serializable (numpy scalars, NaN -> None, dates -> ISO)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return o


class EngineService:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.ds: IngestedDataset = ingest_all()
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._generation = 0

        self.defaults_cfg = PipelineConfig.from_optimal_weights()
        self.full_n_init = max(int(self.defaults_cfg.kmeans_n_init), FAST_N_INIT)
        self.full_n_boot = max(int(self.defaults_cfg.n_bootstrap_runs), FAST_N_BOOT)
        self.available_models = self._discover_models()

        ec_sorted = self.ds.enriched_companies.sort("company_id")
        ext_path = Path(self.defaults_cfg.data_dir) / "external_enrichment.parquet"
        if ext_path.exists():
            try:
                ext_df = pl.read_parquet(ext_path).select([
                    "company_id", "osm_match_tier", "osm_venue_name", "osm_amenity",
                    "osm_cuisine", "osm_distance_m", "osm_name_similarity",
                    "purchasing_power_index", "dist_to_transit_hub_km", "is_transit_hub_catchment"
                ])
                ec_sorted = ec_sorted.join(ext_df, on="company_id", how="left")
            except Exception:
                pass

        # Precompute 250m HoReCa micro-cluster density & nearest client distance (km) via 3D cKDTree
        n_comp = ec_sorted.height
        lat_deg = ec_sorted["latitude"].to_numpy()
        lon_deg = ec_sorted["longitude"].to_numpy()
        has_gps = ~np.isnan(lat_deg) & ~np.isnan(lon_deg)
        valid_idx = np.where(has_gps)[0]

        horeca_250m_arr = np.zeros(n_comp, dtype=np.int32)
        nearest_client_km_arr = np.full(n_comp, np.nan, dtype=np.float32)

        if len(valid_idx) > 0:
            from scipy.spatial import cKDTree
            from src.features.spatial import EARTH_RADIUS_KM

            lat_rad = np.deg2rad(lat_deg[valid_idx])
            lon_rad = np.deg2rad(lon_deg[valid_idx])
            cos_lat = np.cos(lat_rad)
            xyz_valid = np.column_stack([
                cos_lat * np.cos(lon_rad),
                cos_lat * np.sin(lon_rad),
                np.sin(lat_rad),
            ])
            u_tree = cKDTree(xyz_valid)
            chord_250m = 2.0 * np.sin(0.25 / (2.0 * EARTH_RADIUS_KM))
            counts_250m = np.maximum(u_tree.query_ball_point(xyz_valid, r=chord_250m, return_length=True) - 1, 0)
            horeca_250m_arr[valid_idx] = counts_250m.astype(np.int32)

            is_client_rep = ec_sorted["is_client_representative"].to_numpy()[valid_idx]
            client_valid_idx = np.where(is_client_rep)[0]
            if len(client_valid_idx) > 0:
                c_tree = cKDTree(xyz_valid[client_valid_idx])
                dists_chord, _ = c_tree.query(xyz_valid, k=1)
                dists_km = 2.0 * np.arcsin(np.clip(dists_chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM
                nearest_client_km_arr[valid_idx] = np.round(dists_km, 2).astype(np.float32)

        ec_sorted = ec_sorted.with_columns(
            pl.Series("horeca_density_250m", horeca_250m_arr),
            pl.Series("nearest_client_km_default", nearest_client_km_arr),
            pl.when(
                pl.col("legal_type").is_in(["ΑΕ", "ΕΠΕ"])
                | (pl.col("branch_count").fill_null(0) >= 2)
            )
            .then(pl.lit("Tier A — Enterprise & Multi-Branch Hospitality"))
            .when(
                (pl.col("legal_type") == "ΙΚΕ")
                | (pl.col("branch_count").fill_null(0) == 1)
                | (pl.col("has_post_launch_expansion").fill_null(False))
            )
            .then(pl.lit("Tier B — Structured Growth B2B Account"))
            .otherwise(pl.lit("Tier C — Independent HoReCa Venue"))
            .alias("commercial_segment_tier"),
        )

        self._universe_vats = set(ec_sorted["vat"].to_list())
        self._meta_full = ec_sorted
        self._meta_slim = ec_sorted.select(
            [
                pl.col("company_id"),
                pl.col("vat"),
                pl.col("company_name_el").alias("name"),
                pl.col("legal_type"),
                pl.col("prefecture_name").alias("prefecture"),
                pl.col("municipality_name").alias("municipality"),
                pl.col("municipality_typology").alias("muni_typology"),
                pl.col("has_distinctive_brand"),
                pl.col("is_modern_concept"),
                pl.col("has_post_launch_expansion"),
                pl.col("is_pre_summer_launch"),
                (
                    pl.col("osm_match_tier").fill_null("Unmatched").alias("osm_tier")
                    if "osm_match_tier" in ec_sorted.columns
                    else pl.lit("Unmatched").alias("osm_tier")
                ),
                (
                    pl.col("is_transit_hub_catchment").fill_null(0).alias("is_transit_hub")
                    if "is_transit_hub_catchment" in ec_sorted.columns
                    else pl.lit(0).alias("is_transit_hub")
                ),
                pl.col("horeca_density_250m"),
                pl.col("nearest_client_km_default"),
                pl.col("commercial_segment_tier"),
                pl.col("city"),
                pl.col("street"),
                pl.col("street_number"),
                pl.col("zip_code"),
                pl.col("company_age_years").alias("age"),
                pl.col("branch_count").alias("branches"),
                pl.col("primary_kad").alias("kad"),
                pl.col("primary_kad_desc").alias("kad_desc"),
                pl.col("is_branch"),
                pl.col("latitude").fill_nan(None).alias("lat"),
                pl.col("longitude").fill_nan(None).alias("lon"),
            ]
        )

        self._client_sets: dict[str, pl.DataFrame] = {"default": self.ds.clients}
        self._client_set_names: dict[str, str] = {"default": "ForceCRM_Hackathon_Pelatologio.xlsx"}
        self._base_fs: dict[str, CompanyFeatureStore] = {}
        self._fs_cache: dict[tuple[str, float], CompanyFeatureStore] = {}
        self._llm_texts: dict[tuple[str, str, int, int], str] = {}

        self.default_params = self._default_params()
        self.last_params: RunParams | None = None
        self.last_result: ICPEngineResult | None = None
        self.last_fs: CompanyFeatureStore | None = None

        # Initial result: reads the SHA-256 reference cache when present (instant), else computes.
        init_params = replace(self.default_params, precision="full")
        with self._lock:
            self.initial_payload = self._run_locked(init_params, allow_sha_cache=True)
        self.startup_sec = round(time.perf_counter() - t0, 2)

        # Warm the clustering & spatial caches at full precision in the background so the first
        # slider interaction re-scores instantly instead of re-clustering.
        self.warm = False
        threading.Thread(target=self._warmup, daemon=True).start()

    # ------------------------------------------------------------------ setup helpers
    def _discover_models(self) -> list[dict[str, str]]:
        names: list[str] = []
        try:
            schema = pl.read_parquet_schema(self.defaults_cfg.cached_universe_features_path)
            for col in schema:
                if col.startswith("company_neural_svd_"):
                    names.append(col.removeprefix("company_neural_svd_"))
        except Exception:
            pass
        if self.defaults_cfg.embedding_model not in names:
            names.insert(0, self.defaults_cfg.embedding_model)
        out = []
        for n in names:
            info = EMBEDDING_MODELS_REGISTRY.get(n, {})
            out.append(
                {
                    "value": n,
                    "label": n,
                    "detail": f"{info.get('params', '')} · {info.get('origin', 'Τοπικό μοντέλο')}".strip(" ·"),
                }
            )
        return out

    def _default_params(self) -> RunParams:
        c = self.defaults_cfg
        use_emb = bool(c.use_embedding_models and c.embedding_model != "char-ngram-svd")
        default_alpha = float(c.hybrid_alpha) if float(c.hybrid_alpha) > 0.0 else 0.50
        return RunParams(
            client_set="default",
            use_embeddings=use_emb,
            embedding_model=c.embedding_model if c.embedding_model != "char-ngram-svd" else "granite-107m",
            hybrid_alpha=default_alpha if use_emb else 0.0,
            min_icps=int(c.min_icps),
            max_icps=int(c.max_icps),
            turnover_weighting_mode=c.turnover_weighting_mode,
            turnover_weight_alpha=float(c.turnover_weight_alpha),
            sim_weight_feature_cosine=float(c.sim_weight_feature_cosine),
            sim_weight_eb_lift_affinity=float(c.sim_weight_eb_lift_affinity),
            sim_weight_spatial_kernel=float(c.sim_weight_spatial_kernel),
            spatial_kernel_bandwidth_km=float(c.spatial_kernel_bandwidth_km),
            eb_prior_strength=float(c.eb_prior_strength),
            no_fit_min_score=float(c.no_fit_min_score),
            precision="full",
        )

    def _warmup(self) -> None:
        try:
            with self._lock:
                self._compute(replace(self.default_params, precision="full"), allow_sha_cache=False)
        except Exception:
            pass
        finally:
            self.warm = True

    def _feature_store(self, model: str, alpha: float, cfg: PipelineConfig) -> CompanyFeatureStore:
        if model not in self._base_fs:
            self._base_fs[model] = build_feature_store(
                self.ds, PipelineConfig(embedding_model=model, hybrid_alpha=0.0)
            )
        key = (model, round(float(alpha), 4))
        fs = self._fs_cache.get(key)
        if fs is None:
            fs = recombine_feature_store(self._base_fs[model], alpha, cfg)
            if len(self._fs_cache) >= 12:
                self._fs_cache.pop(next(iter(self._fs_cache)))
            self._fs_cache[key] = fs
        return fs

    # ------------------------------------------------------------------ public API
    def next_generation(self) -> int:
        with self._gen_lock:
            self._generation += 1
            return self._generation

    def parse_params(self, payload: dict[str, Any]) -> RunParams:
        base = asdict(self.last_params or self.default_params)
        for key, raw in (payload or {}).items():
            if key == "precision":
                base["precision"] = "full" if raw == "full" else "fast"
            elif key == "client_set":
                if raw not in self._client_sets:
                    raise ValueError("Άγνωστο αρχείο πελατών — ανεβάστε το ξανά.")
                base["client_set"] = raw
            elif key in SPEC_BY_KEY:
                spec = SPEC_BY_KEY[key]
                if spec["type"] == "toggle":
                    base[key] = bool(raw is True or raw == "true" or raw == 1 or raw == "1" or str(raw).lower() == "true")
                elif spec["type"] == "select":
                    allowed = (
                        [m["value"] for m in self.available_models]
                        if key == "embedding_model"
                        else [o["value"] for o in spec["options"]]
                    )
                    if raw not in allowed:
                        raise ValueError(f"Μη έγκυρη τιμή για {spec['label']}.")
                    base[key] = raw
                else:
                    val = float(raw)
                    if not math.isfinite(val):
                        raise ValueError(f"Μη έγκυρη τιμή για {spec['label']}.")
                    val = min(max(val, spec["min"]), spec["max"])
                    base[key] = int(round(val)) if key in ("min_icps", "max_icps") else val
        if base["min_icps"] > base["max_icps"]:
            base["min_icps"], base["max_icps"] = base["max_icps"], base["min_icps"]
        if (
            base["sim_weight_feature_cosine"]
            + base["sim_weight_eb_lift_affinity"]
            + base["sim_weight_spatial_kernel"]
        ) <= 0:
            raise ValueError("Τουλάχιστον ένα βάρος βαθμολόγησης πρέπει να είναι μεγαλύτερο από 0.")
        return RunParams(**base)

    def run(self, params: RunParams, generation: int) -> dict[str, Any]:
        with self._lock:
            if generation < self._generation:
                raise Superseded()
            return self._run_locked(params, allow_sha_cache=False)

    def bootstrap(self) -> dict[str, Any]:
        is_avail, model, installed = check_local_ollama_status(self.defaults_cfg.ollama_model)
        opt = load_optimal_weights_dict()
        specs = []
        for s in PARAM_SPECS:
            s2 = dict(s)
            if s["key"] == "embedding_model":
                s2["options"] = self.available_models
            specs.append(s2)
        return _clean(
            {
                "defaults": asdict(self.default_params),
                "param_specs": specs,
                "precision": {
                    "fast": {"n_init": FAST_N_INIT, "n_bootstrap": FAST_N_BOOT},
                    "full": {"n_init": self.full_n_init, "n_bootstrap": self.full_n_boot},
                },
                "benchmark": {
                    "r500_mean": opt.get("r500_mean"),
                    "r500_std": opt.get("r500_std"),
                    "r2000_mean": opt.get("r2000_mean"),
                    "r2000_std": opt.get("r2000_std"),
                    "baseline_r500": opt.get("baseline_r500_analytical"),
                    "baseline_r2000": opt.get("baseline_r2000_analytical"),
                    "cv_total_evaluations": opt.get("cv_total_evaluations"),
                    "profile_name": opt.get("profile_name"),
                },
                "optimal_weights": opt,
                "ollama": {"available": is_avail, "model": model, "installed": installed},
                "startup_sec": self.startup_sec,
                "initial": self.initial_payload,
                "excluded_companies": self.get_excluded_companies(),
            }
        )

    def upload_clients(self, raw: bytes, filename: str) -> dict[str, Any]:
        if not raw:
            raise ValueError("Το αρχείο είναι κενό.")
        try:
            df = load_clients_excel(raw)
        except Exception as exc:
            raise ValueError(f"Δεν ήταν δυνατή η ανάγνωση του Excel: {exc}") from exc
        matched_df = df.filter(pl.col("vat").is_in(list(self._universe_vats)))
        if matched_df.height == 0:
            raise ValueError("Κανένα ΑΦΜ του αρχείου δεν βρέθηκε στη βάση των 24.424 επιχειρήσεων.")
        set_id = compute_clients_sha256(df)[:16]
        self._client_sets[set_id] = df
        self._client_set_names[set_id] = filename or "clients.xlsx"
        return {
            "client_set": set_id,
            "name": self._client_set_names[set_id],
            "rows": df.height,
            "matched": matched_df.height,
            "turnover": float(matched_df["turnover"].sum()),
        }

    def company(self, company_id: int) -> dict[str, Any] | None:
        row_df = self._meta_full.filter(pl.col("company_id") == company_id)
        if row_df.height == 0:
            return None
        row = row_df.row(0, named=True)
        if "name" not in row and "company_name_el" in row:
            row["name"] = row["company_name_el"]
        kads = (
            self.ds.company_activities.filter(pl.col("company_id") == company_id)
            .join(self.ds.activities, on="activity_id", how="left")
            .with_columns((pl.col("activity_type") == "Κύρια").alias("_is_primary"))
            .sort(["_is_primary", "date_from"], descending=[True, False], nulls_last=True)
            .select(["activity_type", "activity_id", "description", "date_from"])
            .to_dicts()
        )
        context: dict[str, Any] = {}
        excluded_map = icp_engine.load_excluded_companies_map()
        vat_val = str(row.get("vat") or "")
        is_excl = vat_val in excluded_map
        context["is_excluded"] = is_excl
        if is_excl:
            context["exclusion_info"] = excluded_map[vat_val]

        # Automatic operational status verification check (Google Places / G.E.MI. / Heuristics)
        verifier = get_verifier()
        name_val = str(row.get("company_name_el") or row.get("name") or "")
        city_val = str(row.get("city") or "")
        street_val = str(row.get("street") or "")
        context["verification"] = verifier.check_company(vat_val, name_val, city_val, street_val)

        res = self.last_result
        if res is not None:
            look = res.lookalikes_df.filter(pl.col("company_id") == company_id)
            if look.height:
                look_row = dict(look.row(0, named=True))
                if look_row.get("nearest_client_km") is None and row.get("nearest_client_km_default") is not None:
                    look_row["nearest_client_km"] = row.get("nearest_client_km_default")
                context["lookalike"] = look_row
            assign = res.assignments_df.filter(pl.col("company_id") == company_id)
            if assign.height:
                context["assignment"] = assign.row(0, named=True)

            # Compute per-company Contrastive Trait Contribution breakdown (Semantic vs EB Lift vs Spatial)
            try:
                p_Target = None
                if "lookalike" in context:
                    pid = context["lookalike"].get("profile_id")
                    p_Target = next((p for p in res.profiles if p.profile_id == pid), None)
                elif "assignment" in context:
                    pid = context["assignment"].get("profile_id")
                    p_Target = next((p for p in res.profiles if p.profile_id == pid), None)
                if p_Target is None and res.profiles:
                    p_Target = res.profiles[0]

                fs = self.last_fs
                if p_Target is not None and fs is not None:
                    idx_arr = np.where(fs.company_ids == company_id)[0]
                    if len(idx_arr) > 0:
                        c_idx = int(idx_arr[0])
                        cos_val = float(np.clip((fs.unified_features[c_idx] @ p_Target.centroid_vector + 1.0) / 2.0, 0.0, 1.0))

                        # EB probability
                        eb_sum = 0.0
                        ll_map = p_Target.eb_log_lifts
                        trait_pairs = [
                            ("prefecture_name", str(row.get("prefecture_name") or "UNKNOWN"), 1.0),
                            ("municipality_name", str(row.get("municipality_name") or "UNKNOWN"), 0.85),
                            ("legal_type", str(row.get("legal_type") or "OTHER"), 1.25),
                            ("primary_kad", str(row.get("primary_kad") or "56110000"), 1.35),
                            ("kad_4digit", str(row.get("primary_kad") or "56110000")[:4], 0.90),
                        ]
                        for t_col, t_val, t_w in trait_pairs:
                            eb_sum += ll_map.get((t_col, t_val), 0.0) * t_w
                        eb_prob = float(1.0 / (1.0 + math.exp(-0.55 * eb_sum)))

                        dist_km = row.get("nearest_client_km_default")
                        bw = float((self.last_params or self.default_params).spatial_kernel_bandwidth_km or 25.0)
                        if dist_km is not None and math.isfinite(float(dist_km)):
                            sp_val = float(np.clip(math.exp(-0.5 * (float(dist_km) / max(bw, 1.0)) ** 2), 0.15, 1.0))
                        else:
                            sp_val = 0.50

                        p_cur = self.last_params or self.default_params
                        w_c = float(p_cur.sim_weight_feature_cosine)
                        w_e = float(p_cur.sim_weight_eb_lift_affinity)
                        w_s = float(p_cur.sim_weight_spatial_kernel)
                        raw_c = w_c * cos_val
                        raw_e = w_e * eb_prob
                        raw_s = w_s * sp_val
                        tot_raw = max(raw_c + raw_e + raw_s, 1e-6)
                        context["trait_contributions"] = {
                            "semantic_pct": round(100.0 * raw_c / tot_raw, 1),
                            "eb_lift_pct": round(100.0 * raw_e / tot_raw, 1),
                            "spatial_pct": round(100.0 * raw_s / tot_raw, 1),
                            "semantic_score": round(cos_val * 100.0, 1),
                            "eb_lift_score": round(eb_prob * 100.0, 1),
                            "spatial_score": round(sp_val * 100.0, 1),
                        }
            except Exception:
                pass
        return _clean({"company": row, "kads": kads, "context": context})

    def verify_company(self, company_id: int, force: bool = True) -> dict[str, Any]:
        row_df = self._meta_full.filter(pl.col("company_id") == company_id)
        if row_df.height == 0:
            raise ValueError(f"Company ID {company_id} not found.")
        row = row_df.row(0, named=True)
        vat = str(row["vat"])
        name = str(row.get("company_name_el") or row.get("name") or "")
        city = str(row.get("city") or "")
        street = str(row.get("street") or "")

        verifier = get_verifier()
        verif = verifier.check_company(vat, name, city, street, force_refresh=force)

        # If verified as closed, automatically exclude it from lookalikes and refresh
        payload = None
        if verif.get("is_closed"):
            excl_res = self.exclude_company(
                company_id,
                reason=f"Επιβεβαιωμένη διακοπή ({verif.get('source')}): {verif.get('details')}"
            )
            payload = excl_res.get("payload")

        return _clean({
            "status": "ok",
            "company_id": company_id,
            "verification": verif,
            "excluded_triggered": bool(verif.get("is_closed")),
            "payload": payload,
        })

    def verify_top_leads(self, count: int = 25) -> dict[str, Any]:
        res = self.last_result
        if res is None or res.lookalikes_df.height == 0:
            return {"status": "ok", "checked": 0, "closed_detected": 0, "closed_companies": []}

        top_lookalikes = res.lookalikes_df.head(count).join(self._meta_full, on="company_id", how="left")
        verifier = get_verifier()
        closed_cids = []

        for r in top_lookalikes.iter_rows(named=True):
            cid = r["company_id"]
            vat = str(r["vat"])
            name = str(r.get("company_name_el") or r.get("name") or "")
            city = str(r.get("city") or "")
            street = str(r.get("street") or "")
            v = verifier.check_company(vat, name, city, street, force_refresh=False, auto_save=False)
            if v.get("is_closed"):
                closed_cids.append((cid, vat, name, city, v.get("details", "")))

        # Flush verification cache once per batch
        from src.verifier import save_verification_cache
        save_verification_cache(verifier.cache)

        payload = None
        if closed_cids:
            excluded_map = icp_engine.load_excluded_companies_map()
            for cid, vat, name, city, details in closed_cids:
                excluded_map[vat] = {
                    "company_id": cid,
                    "vat": vat,
                    "name": name,
                    "city": city,
                    "reason": f"Αυτόματος έλεγχος: {details}",
                    "date_excluded": datetime.now().strftime("%Y-%m-%d"),
                }
            icp_engine.save_excluded_companies_map(excluded_map)
            with self._lock:
                target_params = self.last_params or self.default_params
                payload = self._run_locked(target_params, allow_sha_cache=False)

        return _clean({
            "status": "ok",
            "checked": top_lookalikes.height,
            "closed_detected": len(closed_cids),
            "closed_companies": [{"company_id": c[0], "vat": c[1], "name": c[2]} for c in closed_cids],
            "payload": payload,
        })

    def exclude_company(self, company_id: int, reason: str = "") -> dict[str, Any]:
        row_df = self._meta_full.filter(pl.col("company_id") == company_id)
        if row_df.height == 0:
            raise ValueError(f"Company ID {company_id} not found.")
        row = row_df.row(0, named=True)
        vat = str(row["vat"])
        name = row.get("company_name_el") or row.get("name") or "Επιχείρηση"
        
        excluded_map = icp_engine.load_excluded_companies_map()
        excluded_map[vat] = {
            "company_id": company_id,
            "vat": vat,
            "name": name,
            "city": row.get("city") or "",
            "reason": reason or "Διακοπή εργασιών / Ανενεργή επιχείρηση (χειροκίνητη αφαίρεση από χρήστη)",
            "source": "user",
            "date_excluded": datetime.now().strftime("%Y-%m-%d"),
        }
        icp_engine.save_excluded_companies_map(excluded_map)
        
        with self._lock:
            target_params = self.last_params or self.default_params
            payload = self._run_locked(target_params, allow_sha_cache=False)
        return _clean({"status": "ok", "vat": vat, "name": name, "payload": payload})

    def unexclude_company(self, company_id: int) -> dict[str, Any]:
        row_df = self._meta_full.filter(pl.col("company_id") == company_id)
        if row_df.height == 0:
            raise ValueError(f"Company ID {company_id} not found.")
        vat = str(row_df.row(0, named=True)["vat"])
        
        excluded_map = icp_engine.load_excluded_companies_map()
        if vat in excluded_map:
            del excluded_map[vat]
            icp_engine.save_excluded_companies_map(excluded_map)
            
        with self._lock:
            target_params = self.last_params or self.default_params
            payload = self._run_locked(target_params, allow_sha_cache=False)
        return _clean({"status": "ok", "vat": vat, "payload": payload})

    def get_excluded_companies(self) -> list[dict[str, Any]]:
        return _clean(list(icp_engine.load_excluded_companies_map().values()))

    def llm_prompt(self, profile_id: str) -> dict[str, Any]:
        prof = self._find_profile(profile_id)
        sys_p, usr_p = build_grounded_prompt_el(prof)
        is_avail, model, installed = check_local_ollama_status(self.defaults_cfg.ollama_model)
        return {
            "system_prompt": sys_p,
            "user_prompt": usr_p,
            "ollama": {"available": is_avail, "model": model, "installed": installed},
        }

    def llm_generate(self, profile_id: str, model: str | None = None) -> dict[str, Any]:
        prof = self._find_profile(profile_id)
        is_avail, resolved, installed = check_local_ollama_status(self.defaults_cfg.ollama_model)
        
        # If Ollama is offline or deterministic generation requested, use grounded deterministic engine
        if not is_avail or model == "deterministic":
            title_el, desc_el = icp_engine._build_icp_title_and_deterministic_description(
                profile_id=prof.profile_id,
                client_count=prof.client_count,
                client_share=prof.client_share,
                turnover_share=prof.turnover_share,
                turnover_median=prof.turnover_median,
                stability_score=prof.stability_score,
                confidence_score=prof.confidence_score,
                proofs=prof.significant_features,
            )
            prof.ai_description_el = desc_el
            if self.last_params is not None:
                self._llm_texts[self._llm_key(self.last_params.client_set, prof)] = desc_el
            return {
                "accepted": True,
                "notes": ["100% Grounded Deterministic Explainer (Εγγύηση μηδενικού hallucination)"],
                "model": "deterministic-engine",
                "elapsed_sec": 0.01,
                "text": desc_el,
            }

        if model and model in installed:
            resolved = model
        import ollama

        cfg = self.defaults_cfg
        sys_p, usr_p = build_grounded_prompt_el(prof)
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=resolved,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            options={
                "temperature": cfg.llm_temperature,
                "seed": cfg.llm_seed,
                "top_p": 1.0,
                "num_predict": cfg.llm_max_tokens,
            },
        )
        msg = getattr(resp, "message", None)
        content = getattr(msg, "content", None) if msg else None
        if content is None and isinstance(resp, dict):
            content = resp.get("message", {}).get("content", "")
        body = (content or "").strip()
        ok, notes = validate_grounded_llm_output(body, prof)
        elapsed = round(time.perf_counter() - t0, 2)
        if not ok:
            return {"accepted": False, "notes": notes, "model": resolved, "elapsed_sec": elapsed, "raw": body}
        text = (
            f"{AI_DISCLOSURE_PREFIX_EL}\n"
            f"*(Μοντέλο: `{resolved}` | temp={cfg.llm_temperature} | seed={cfg.llm_seed} | "
            f"Σταθερότητα: {100*prof.stability_score:.1f}% | Εμπιστοσύνη: {100*prof.confidence_score:.1f}%)*\n\n"
            f"{body}\n\n{EU_AI_ACT_FOOTER_EL}"
        )
        prof.ai_description_el = text
        if self.last_params is not None:
            self._llm_texts[self._llm_key(self.last_params.client_set, prof)] = text
        return {"accepted": True, "notes": notes, "model": resolved, "elapsed_sec": elapsed, "text": text}

    def export(self, name: str) -> tuple[bytes, Path]:
        if name not in EXPORT_NAMES:
            raise ValueError("Άγνωστο αρχείο εξαγωγής.")
        with self._lock:
            params = replace(self.last_params or self.default_params, precision="full")
            res, cfg, *_ = self._compute(params, allow_sha_cache=False)
            is_reference = replace(params, precision="full") == replace(self.default_params, precision="full")
            out_dir = Path(cfg.output_dir) if is_reference else Path(cfg.output_dir) / "exports" / params.client_set
            paths = export_deliverables(res, out_dir)
            return paths[name].read_bytes(), paths[name]

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _llm_key(client_set: str, prof: Any) -> tuple[str, str, int, int]:
        return (client_set, prof.profile_id, int(prof.client_count), int(round(prof.turnover_total)))

    def _find_profile(self, profile_id: str):
        res = self.last_result
        if res is None:
            raise KeyError(profile_id)
        norm_id = str(profile_id).replace("-", "_").upper()
        for p in res.profiles:
            if p.profile_id == profile_id or str(p.profile_id).replace("-", "_").upper() == norm_id:
                return p
        raise KeyError(profile_id)

    def _compute(self, params: RunParams, allow_sha_cache: bool):
        clients_df = self._client_sets[params.client_set]
        full = params.precision == "full"
        effective_model = params.embedding_model if params.use_embeddings else "char-ngram-svd"
        if params.use_embeddings:
            effective_alpha = float(params.hybrid_alpha) if float(params.hybrid_alpha) > 0.0 else 0.50
        else:
            effective_alpha = 0.0

        cfg_dict = {
            k: getattr(params, k)
            for k in CFG_KEYS
            if k not in ("use_embeddings", "embedding_model", "hybrid_alpha")
        }
        cfg_dict["use_embedding_models"] = params.use_embeddings
        cfg_dict["embedding_model"] = effective_model
        cfg_dict["hybrid_alpha"] = effective_alpha

        cfg = PipelineConfig.from_optimal_weights(
            **cfg_dict,
            kmeans_n_init=self.full_n_init if full else FAST_N_INIT,
            n_bootstrap_runs=self.full_n_boot if full else FAST_N_BOOT,
            use_sha256_client_cache=bool(full and allow_sha_cache and params.use_embeddings),
        )
        fs = self._feature_store(effective_model, effective_alpha, cfg)
        t0 = time.perf_counter()
        res = run_icp_engine(self.ds, fs, cfg, custom_clients_df=clients_df)
        engine_ms = (time.perf_counter() - t0) * 1000.0
        info = dict(icp_engine._LAST_CLUSTER_STAGE_INFO)
        enrich_profiles_with_local_llm(res, cfg, force_deterministic=True)
        for prof in res.profiles:
            txt = self._llm_texts.get(self._llm_key(params.client_set, prof))
            if txt:
                prof.ai_description_el = txt
        return res, cfg, fs, clients_df, info, engine_ms

    def _run_locked(self, params: RunParams, allow_sha_cache: bool) -> dict[str, Any]:
        t0 = time.perf_counter()
        res, cfg, fs, clients_df, info, engine_ms = self._compute(params, allow_sha_cache)
        self.last_params, self.last_result, self.last_fs = params, res, fs

        if res.loaded_from_sha256_cache:
            effective, reused = "full", True
        else:
            effective = (
                "full"
                if info["n_init"] >= self.full_n_init and info["n_boot"] >= self.full_n_boot
                else "fast"
            )
            reused = bool(info["hit"])

        payload = self._payload(params, res, fs, clients_df)
        payload["meta"].update(
            {
                "precision_requested": params.precision,
                "precision_effective": effective,
                "cluster_reused": reused,
                "loaded_from_cache": bool(res.loaded_from_sha256_cache),
                "engine_ms": round(engine_ms, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        )
        return payload

    def _clean_muni_name(self, m: str | None) -> str:
        if not m:
            return "Άγνωστη Περιοχή"
        parts = m.split("/")
        name = parts[0].strip()
        return f"Δήμος {name}" if not name.startswith("Δήμος") else name

    def _build_sales_intelligence(
        self,
        p: Any,
        look_p: pl.DataFrame,
        client_p: pl.DataFrame,
    ) -> dict[str, Any]:
        total_leads = look_p.height
        if total_leads == 0:
            return {
                "lead_funnel": {
                    "total": 0,
                    "hot": {"count": 0, "share": 0.0, "share_pct": 0.0, "label": "Hot Leads", "min_score": "≥90%", "action": "Άμεση Κλήση & Demo (48h)"},
                    "warm": {"count": 0, "share": 0.0, "share_pct": 0.0, "label": "Warm Leads", "min_score": "80-89%", "action": "SDR Outbound Sequence"},
                    "nurture": {"count": 0, "share": 0.0, "share_pct": 0.0, "label": "Nurture Leads", "min_score": "<80%", "action": "Marketing Lead Drip"},
                },
                "territories": [],
                "territory_tip": "",
                "firmographics": {
                    "top_legal": [],
                    "dominant_legal": "ΙΚΕ",
                    "multi_branch_share": 0.0,
                    "multi_branch_share_pct": 0.0,
                    "multi_branch_count": 0,
                    "median_age_years": 0.0,
                },
                "playbook": {},
            }

        # 1. Lead Funnel
        hot_df = look_p.filter(pl.col("score") >= 0.90)
        warm_df = look_p.filter((pl.col("score") >= 0.80) & (pl.col("score") < 0.90))
        nurture_df = look_p.filter(pl.col("score") < 0.80)

        hot_cnt = hot_df.height
        warm_cnt = warm_df.height
        nurture_cnt = nurture_df.height

        # 2. Top Territories (Municipalities)
        muni_df = (
            look_p.filter(pl.col("municipality").is_not_null() & (pl.col("municipality") != ""))
            .group_by("municipality")
            .len()
            .sort("len", descending=True)
            .head(5)
        )
        territories = []
        top3_cnt = 0
        for idx, row in enumerate(muni_df.iter_rows(named=True)):
            cnt = int(row["len"])
            if idx < 3:
                top3_cnt += cnt
            territories.append({
                "name": self._clean_muni_name(row["municipality"]),
                "raw_name": row["municipality"],
                "count": cnt,
                "share": round(cnt / total_leads, 3),
                "share_pct": round((cnt / total_leads) * 100, 1),
            })

        top3_share_pct = round((top3_cnt / total_leads) * 100, 1)

        # 3. Top Legal Forms
        legal_df = (
            look_p.filter(pl.col("legal_type").is_not_null() & (pl.col("legal_type") != ""))
            .group_by("legal_type")
            .len()
            .sort("len", descending=True)
            .head(3)
        )
        top_legal = [
            {
                "legal_type": row["legal_type"],
                "count": int(row["len"]),
                "share": round(row["len"] / total_leads, 3),
                "share_pct": round((row["len"] / total_leads) * 100, 1),
            }
            for row in legal_df.iter_rows(named=True)
        ]
        dominant_legal = top_legal[0]["legal_type"] if top_legal else "ΙΚΕ"

        # 4. Multi-branch & Age
        multi_branch_cnt = (
            look_p.filter(pl.col("branches") > 1).height
            if "branches" in look_p.columns
            else look_p.filter(pl.col("is_branch") == 1).height
        )
        multi_branch_share = round(multi_branch_cnt / total_leads, 3)

        ages = [a for a in look_p["age"].to_list() if a is not None and not math.isnan(a)]
        median_age = round(float(np.median(ages)), 1) if ages else 4.0

        # 5. Playbook & Commercial Angle
        title_lower = (p.title_el or "").lower()
        if any(w in title_lower for w in ["τροφοδοσ", "catering", "εστιασ", "εστιατορ", "γευμα"]):
            sector_name = "Εταιρική Εστίαση, Catering & B2B Τροφοδοσία"
            sector_name_en = "Corporate Dining, Catering & B2B Foodservice"
            deal_tier = "Mid-Market / High Velocity"
            deal_size_label = f"€{round(p.turnover_median or 230000):,} διάμεσος κύκλος εργασιών".replace(",", ".")
            deal_size_label_en = f"€{round(p.turnover_median or 230000):,} median annual revenue"
            cycle = "7 - 14 ημέρες"
            cycle_en = "7 - 14 days"
            persona = f"Διαχειριστής / Operations Director ({dominant_legal})"
            persona_en = f"Managing Partner / Operations Director ({dominant_legal})"
            persona_desc = "Ευέλικτη διοίκηση με άμεση εξουσία υπογραφής. Εστιάζει στο operational efficiency και τη μείωση λαθών τιμολόγησης."
            persona_desc_en = "Agile leadership with direct signing authority. Focuses on operational efficiency and eliminating invoicing errors."
            pain_points = [
                "Διαχείριση υψηλού όγκου καθημερινών δελτίων αποστολής & τιμολογίων",
                "Αυστηρή συμμόρφωση myDATA χωρίς καθυστερήσεις στα σημεία παράδοσης",
                "Παρακολούθηση ανοιχτών πιστωτικών υπολοίπων προμηθευτών & πελατών B2B",
                "Έλεγχος κόστους πρώτων υλών και συντονισμός κεντρικής κουζίνας / αποθήκης",
            ]
            pain_points_en = [
                "Managing high daily volume of delivery notes & B2B invoices",
                "Real-time myDATA tax compliance without bottlenecks at delivery points",
                "Tracking open supplier & B2B customer account balances",
                "Raw material food-cost control & central kitchen/warehouse coordination",
            ]
            pitch_hook = (
                "«Καλημέρα σας, επικοινωνώ γιατί συνεργαζόμαστε με κορυφαίες επιχειρήσεις catering & εστίασης στην περιοχή σας. "
                "Τους βοηθάμε να αυτοματοποιήσουν πλήρως τη ροή παραγγελιών και την τιμολόγηση, εξοικονομώντας 15+ ώρες γραφειοκρατίας "
                "την εβδομάδα χωρίς να αλλάξουν τις συνήθειες του προσωπικού. Θα είχατε 10 λεπτά την Πέμπτη για ένα γρήγορο demo;»"
            )
            pitch_hook_en = (
                "\"Good morning, I'm reaching out because we partner with leading catering and hospitality groups in your area. "
                "We help them automate order routing and B2B invoicing end-to-end, saving 15+ hours of admin work per week "
                "without disrupting staff workflows. Would you have 10 minutes on Thursday for a quick live demo?\""
            )
            objection_rebuttal = (
                "«Χρησιμοποιούμε ήδη άλλο εμπορικό πρόγραμμα / Excel» ➔ "
                "«Ακριβώς επειδή έχετε μεγάλο όγκο, τα παλιά συστήματα κολλάνε στη διαβίβαση myDATA και δεν σας δίνουν ζωντανή εικόνα ανά σημείο. "
                "Εμείς κουμπώνουμε άμεσα χωρίς καμία διακοπή της καθημερινής σας λειτουργίας.»"
            )
            objection_rebuttal_en = (
                "\"We already use legacy ERP / Excel\" ➔ "
                "\"Because of your high transaction volume, legacy tools lag on myDATA transmission and lack real-time multi-site visibility. "
                "We integrate seamlessly with zero downtime for your daily operations.\""
            )
            channel = "Inside Sales με τηλεφωνική κλήση & 15-λεπτο Online Demo · Επιτόπια επίσκεψη σε accounts >€300k."
            channel_en = "Inside Sales phone outreach & 15-min Online Demo · Field visit for accounts >€300k."
        elif any(w in title_lower for w in ["τουρισ", "ξενοδοχ", "καταλυμ", "διαμον", "χαλκιδικ", "νησι", "κρητ"]):
            sector_name = "Τουρισμός, Φιλοξενία & Εποχικές Επιχειρήσεις"
            sector_name_en = "Tourism, Hospitality & Seasonal Resorts"
            deal_tier = "Seasonal Enterprise / Regional Accounts"
            deal_size_label = f"€{round(p.turnover_median or 180000):,} διάμεσος κύκλος εργασιών".replace(",", ".")
            deal_size_label_en = f"€{round(p.turnover_median or 180000):,} median annual revenue"
            cycle = "10 - 20 ημέρες (Κρίσιμο παράθυρο: Ιανουάριος - Απρίλιος)"
            cycle_en = "10 - 20 days (Critical window: January - April)"
            persona = f"Ιδιοκτήτης / Γενικός Διευθυντής ({dominant_legal})"
            persona_en = f"Owner / General Manager ({dominant_legal})"
            persona_desc = "Προσωπική λήψη αποφάσεων από τον επιχειρηματία. Απαιτεί απλότητα για εκπαίδευση εποχικού προσωπικού σε 1 ώρα."
            persona_desc_en = "Direct owner decision-making. Requires turnkey simplicity to onboard seasonal staff in under 1 hour."
            pain_points = [
                "Ακραία συγκέντρωση κύκλου εργασιών στο 5μηνο της σεζόν (Μάιος - Οκτώβριος)",
                "Ανάγκη άμεσης εκπαίδευσης εποχικού προσωπικού χωρίς τεχνικές περιπλοκές",
                "Αδιάλειπτη διασύνδεση POS / ταμειακών συστημάτων ακόμα και με ασταθές internet",
                "Ενιαία παρακολούθηση πολλαπλών σημείων πώλησης (Beach bar, Εστιατόριο, Reception)",
            ]
            pain_points_en = [
                "Extreme revenue concentration during the 5-month peak season (May - October)",
                "Rapid seasonal staff onboarding with zero technical friction",
                "Offline-resilient POS & fiscal terminal integration",
                "Unified tracking across multiple outlets (Beach bar, Restaurant, Reception)",
            ]
            pitch_hook = (
                "«Καλημέρα σας, επικοινωνούμε με επιλεγμένες τουριστικές επιχειρήσεις της περιοχής ενόψει της νέας σεζόν. "
                "Βοηθάμε τους ιδιοκτήτες να διασφαλίσουν μηδενικές ουρές στα ταμεία και απόλυτο έλεγχο συναλλαγών, με σύστημα που το εποχικό προσωπικό "
                "μαθαίνει σε 45 λεπτά. Προλαβαίνουμε να το δείτε πριν ξεκινήσει το άνοιγμα της σεζόν;»"
            )
            pitch_hook_en = (
                "\"Good morning, we're contacting select hospitality venues in your region ahead of the upcoming season. "
                "We help owners eliminate checkout queues and maintain 100% transaction control with a system seasonal staff "
                "master in 45 minutes. Can we show you a quick walkthrough before pre-season opening?\""
            )
            objection_rebuttal = (
                "«Είμαστε κλειστά τώρα / θα το δούμε την άνοιξη» ➔ "
                "«Ακριβώς γι' αυτό σας καλούμε τώρα: όταν ανοίξει η σεζόν δεν υπάρχει χρόνος για ρυθμίσεις. "
                "Τώρα είναι η στιγμή να το στήσουμε δωρεάν, ώστε την πρώτη μέρα λειτουργίας να είστε 100% έτοιμοι χωρίς άγχος.»"
            )
            objection_rebuttal_en = (
                "\"We're closed for winter / let's talk in spring\" ➔ "
                "\"That's exactly why we reach out now: once peak season starts, there is zero time for setup. "
                "Deploying now means you launch on day one 100% ready and stress-free.\""
            )
            channel = "Στοχευμένη εξόρμηση πεδίου (Field Sales Blitz) στους τουριστικούς άξονες & τηλεφωνικό follow-up."
            channel_en = "Targeted Field Sales Blitz across tourist corridors & phone follow-up."
        else:
            sector_name = "Εμπορικές & Παραγωγικές Επιχειρήσεις B2B"
            sector_name_en = "Commercial & Production B2B Enterprises"
            deal_tier = "Commercial B2B Growth"
            deal_size_label = f"€{round(p.turnover_median or 200000):,} διάμεσος κύκλος εργασιών".replace(",", ".")
            deal_size_label_en = f"€{round(p.turnover_median or 200000):,} median annual revenue"
            cycle = "14 - 21 ημέρες"
            cycle_en = "14 - 21 days"
            persona = f"Εμπορικός Διευθυντής / Ιδιοκτήτης ({dominant_legal})"
            persona_en = f"Commercial Director / Owner ({dominant_legal})"
            persona_desc = "Εστιάζει στην αύξηση πωλήσεων και την ταχύτερη είσπραξη οφειλών."
            persona_desc_en = "Focuses on sales expansion and faster receivables collection."
            pain_points = [
                "Χειροκίνητη παρακολούθηση παραγγελιών και κατακερματισμός δεδομένων",
                "Καθυστερήσεις στην ενημέρωση υπολοίπων και είσπραξη απαιτήσεων",
                "Ανάγκη ενοποίησης πωλητών και back-office σε ενιαία πλατφόρμα",
            ]
            pain_points_en = [
                "Manual order tracking and fragmented customer data",
                "Delays in balance reconciliation and receivables collection",
                "Need to unify field sales reps and back-office on a single platform",
            ]
            pitch_hook = (
                "«Καλημέρα σας, βοηθάμε επιχειρήσεις στον κλάδο σας να μειώσουν τον χρόνο διαχείρισης παραγγελιών κατά 30% "
                "και να επιταχύνουν τις εισπράξεις τους με αυτοματοποιημένες υπενθυμίσεις. Θα σας ενδιέφερε μια 10-λεπτη επίδειξη;»"
            )
            pitch_hook_en = (
                "\"Good morning, we help businesses in your sector cut order processing time by 30% "
                "and accelerate collections with automated reminders. Would you be open to a 10-minute demo?\""
            )
            objection_rebuttal = (
                "«Δεν έχουμε budget αυτή τη στιγμή» ➔ "
                "«Το σύστημά μας αυτοχρηματοδοτείται από τον πρώτο μήνα, μειώνοντας τις χαμένες εργατοώρες και τις επισφάλειες. "
                "Ας δούμε μαζί τα νούμερα χωρίς καμία δέσμευση.»"
            )
            objection_rebuttal_en = (
                "\"We don't have budget right now\" ➔ "
                "\"Our platform pays for itself in month one by cutting wasted admin hours and bad debt. "
                "Let's review the ROI numbers together with zero obligation.\""
            )
            channel = "Συνδυασμός email outreach με τηλεφωνική κλήση εντός 48 ωρών."
            channel_en = "Combined email outreach with phone follow-up within 48 hours."

        return {
            "lead_funnel": {
                "total": total_leads,
                "hot": {
                    "count": hot_cnt,
                    "share": round(hot_cnt / total_leads, 3),
                    "share_pct": round((hot_cnt / total_leads) * 100, 1),
                    "label": "Hot Leads",
                    "min_score": "≥90%",
                    "action": "Άμεση Κλήση & Demo (48h)",
                    "action_en": "Immediate Call & Demo (48h)",
                },
                "warm": {
                    "count": warm_cnt,
                    "share": round(warm_cnt / total_leads, 3),
                    "share_pct": round((warm_cnt / total_leads) * 100, 1),
                    "label": "Warm Leads",
                    "min_score": "80-89%",
                    "action": "SDR Outbound Sequence",
                    "action_en": "SDR Outbound Sequence",
                },
                "nurture": {
                    "count": nurture_cnt,
                    "share": round(nurture_cnt / total_leads, 3),
                    "share_pct": round((nurture_cnt / total_leads) * 100, 1),
                    "label": "Nurture Leads",
                    "min_score": "<80%",
                    "action": "Marketing Lead Drip",
                    "action_en": "Marketing Lead Drip",
                },
            },
            "territories": territories,
            "territory_tip": f"Συγκέντρωση Αγοράς: Το {top3_share_pct}% των ευκαιριών βρίσκεται στους 3 κυριότερους δήμους. Προγραμματίστε στοχευμένες ημερήσιες εξορμήσεις πεδίου (field sales days) για μέγιστη απόδοση συναντήσεων.",
            "territory_tip_en": f"Market Concentration: {top3_share_pct}% of high-fit prospects are concentrated in the top 3 municipalities. Schedule targeted field sales blitz days for maximum meeting density.",
            "firmographics": {
                "top_legal": top_legal,
                "dominant_legal": dominant_legal,
                "multi_branch_share": multi_branch_share,
                "multi_branch_share_pct": round(multi_branch_share * 100, 1),
                "multi_branch_count": multi_branch_cnt,
                "median_age_years": median_age,
            },
            "playbook": {
                "sector_name": sector_name,
                "sector_name_en": sector_name_en,
                "deal_tier": deal_tier,
                "deal_size_label": deal_size_label,
                "deal_size_label_en": deal_size_label_en,
                "sales_cycle": cycle,
                "sales_cycle_en": cycle_en,
                "buyer_persona": persona,
                "buyer_persona_en": persona_en,
                "buyer_persona_desc": persona_desc,
                "buyer_persona_desc_en": persona_desc_en,
                "pain_points": pain_points,
                "pain_points_en": pain_points_en,
                "pitch_hook": pitch_hook,
                "pitch_hook_en": pitch_hook_en,
                "objection_rebuttal": objection_rebuttal,
                "objection_rebuttal_en": objection_rebuttal_en,
                "recommended_channel": channel,
                "recommended_channel_en": channel_en,
            },
        }

    def _payload(
        self,
        params: RunParams,
        res: ICPEngineResult,
        fs: CompanyFeatureStore,
        clients_df: pl.DataFrame,
    ) -> dict[str, Any]:
        look_base = res.lookalikes_df
        if "nearest_client_km" in look_base.columns:
            meta_join = self._meta_slim.drop("nearest_client_km_default")
            look = (
                look_base.join(meta_join, on="company_id", how="left")
                .with_columns(pl.col("score").round(4))
                .rename({"αιτιολόγηση": "reason"})
                .sort("rank")
            )
        else:
            look = (
                look_base.join(self._meta_slim, on="company_id", how="left")
                .with_columns(
                    pl.col("score").round(4),
                    pl.col("nearest_client_km_default").alias("nearest_client_km"),
                )
                .drop("nearest_client_km_default")
                .rename({"αιτιολόγηση": "reason"})
                .sort("rank")
            )
        look_counts = dict(
            look.group_by("profile_id").len().iter_rows()
        ) if look.height else {}

        matched_clients = clients_df.filter(pl.col("vat").is_in(list(self._universe_vats)))
        assign = (
            res.assignments_df.join(self._meta_slim, on="company_id", how="left")
            .join(matched_clients.select(["vat", "turnover"]), on="vat", how="left")
            .with_columns(pl.col("score").round(4))
        )
        assigned = assign.filter(pl.col("profile_id") != "NO_FIT").height
        no_fit = assign.height - assigned

        profiles = []
        for idx, p in enumerate(res.profiles):
            parts = [s.strip() for s in p.title_el.split("—", 1)]
            desc = p.ai_description_el or ""
            look_p = look.filter(pl.col("profile_id") == p.profile_id)
            client_p = assign.filter(pl.col("profile_id") == p.profile_id)
            sales_intel = self._build_sales_intelligence(p, look_p, client_p)
            profiles.append(
                {
                    "id": p.profile_id,
                    "index": idx,
                    "title": p.title_el,
                    "short_title": parts[0],
                    "region": parts[1] if len(parts) > 1 else "",
                    "client_count": p.client_count,
                    "client_share": p.client_share,
                    "turnover_total": p.turnover_total,
                    "turnover_share": p.turnover_share,
                    "turnover_median": p.turnover_median,
                    "turnover_mean": p.turnover_mean,
                    "stability": p.stability_score,
                    "confidence": p.confidence_score,
                    "medoid": list(p.spatial_medoid_latlon_deg),
                    "lookalike_count": int(look_counts.get(p.profile_id, 0)),
                    "ai_description": desc,
                    "ai_source": "ollama" if "Μοντέλο:" in desc else "deterministic",
                    "sales_intelligence": sales_intel,
                    "features": [
                        {
                            "type": f.feature_type,
                            "value": f.feature_value,
                            "label": f.feature_label_el,
                            "icp_count": f.icp_count,
                            "icp_share": f.icp_share,
                            "universe_count": f.universe_count,
                            "universe_share": f.universe_share,
                            "raw_lift": f.raw_lift,
                            "eb_lift": f.eb_lift,
                            "p_fdr": f.p_value_fdr,
                            "p_raw": f.p_value_raw,
                        }
                        for f in p.significant_features
                    ],
                }
            )

        total_turnover = float(matched_clients["turnover"].sum()) if matched_clients.height else 0.0
        meta = {
            "client_set": params.client_set,
            "client_set_name": self._client_set_names.get(params.client_set, params.client_set),
            "client_sha256": res.client_sha256[:16],
            "client_count": matched_clients.height,
            "client_rows": assign.height,
            "total_turnover": total_turnover,
            "mean_turnover": total_turnover / max(matched_clients.height, 1),
            "optimal_k": res.optimal_k,
            "global_ari": res.global_bootstrap_ari,
            "silhouette_by_k": {str(k): v for k, v in sorted(res.silhouette_by_k.items())},
            "lookalike_count": look.height,
            "assigned": assigned,
            "no_fit": no_fit,
            "universe_size": self.ds.enriched_companies.height,
            "telemetry": fs.embedding_telemetry,
        }
        return _clean(
            {
                "params": asdict(params),
                "meta": meta,
                "profiles": profiles,
                "lookalikes": look.to_dicts(),
                "clients": assign.sort("score", descending=True).to_dicts(),
            }
        )

