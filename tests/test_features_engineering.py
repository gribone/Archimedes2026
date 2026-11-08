"""
tests/test_features_engineering.py
===================================
Unit and regression tests for Milestone 2:
1. Greek transit hub coordinates loading & fallback (Feature 10)
2. Feature extraction relocations: expansion velocity, seasonality, brand, municipality typology (Feature 7)
3. Dynamic ingestion size assertions (Feature 8)
4. Fast contiguous 2D NumPy feature store serialization (.npz) without pickle dependency (Feature 9)
5. Package re-export backward compatibility shim
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import PipelineConfig
from src.features.engineering import (
    _compute_brand_features,
    _compute_municipality_typology,
    _compute_post_launch_expansion,
    _compute_seasonality_features,
    compute_brand_features,
    compute_municipality_typology,
    compute_post_launch_expansion,
    compute_seasonality_features,
)
from src.features.store import (
    CompanyFeatureStore,
    load_feature_store_npz,
    save_feature_store_npz,
)
from src.features.transit import (
    FALLBACK_GREEK_TRANSIT_HUBS,
    GREEK_TRANSIT_FOOT_TRAFFIC_HUBS,
    load_greek_transit_hubs,
)
from src.ingest import IngestedDataset, print_sanity_check_report


# =============================================================================
# 1. TRANSIT HUBS EXTERNALIZATION TESTS (Feature 10)
# =============================================================================

def test_transit_hubs_loading_and_fallback(tmp_path: Path) -> None:
    """Verify loading from JSON asset and robust fallback when file is absent."""
    # 1. Normal load from data asset
    hubs = load_greek_transit_hubs()
    assert len(hubs) == 37, f"Expected 37 transit hubs, got {len(hubs)}"
    assert hubs[0] == (37.9753, 23.7361, "Syntagma Metro / Transit Hub")

    # Verify module constant matches
    assert len(GREEK_TRANSIT_FOOT_TRAFFIC_HUBS) == 37
    assert GREEK_TRANSIT_FOOT_TRAFFIC_HUBS[0] == hubs[0]

    # 2. Fallback when given invalid / missing path
    missing_file = tmp_path / "does_not_exist.json"
    fallback_result = load_greek_transit_hubs(missing_file)
    assert len(fallback_result) == 37
    assert fallback_result == FALLBACK_GREEK_TRANSIT_HUBS

    # 3. Custom valid JSON file
    custom_json = tmp_path / "custom_hubs.json"
    custom_json.write_text(
        '[{"name": "Custom Hub", "lat": 38.0, "lon": 23.0}]',
        encoding="utf-8",
    )
    custom_loaded = load_greek_transit_hubs(custom_json)
    assert len(custom_loaded) == 1
    assert custom_loaded[0] == (38.0, 23.0, "Custom Hub")


# =============================================================================
# 2. FEATURE EXTRACTION FUNCTIONS TESTS (Feature 7)
# =============================================================================

def test_compute_post_launch_expansion() -> None:
    """Test expansion days computation and >=180 day commercial pivot threshold."""
    companies = pl.DataFrame({
        "company_id": [1, 2, 3],
        "incorporation_date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1)],
    })
    company_activities = pl.DataFrame({
        "company_id": [1, 1, 2, 3],
        "activity_type": ["Κύρια", "Δευτερεύουσα", "Δευτερεύουσα", "Κύρια"],
        "date_from": [date(2020, 1, 1), date(2020, 8, 1), date(2020, 3, 1), date(2020, 1, 1)],
    })

    res = compute_post_launch_expansion(companies, company_activities)
    assert res.height == 2  # Only companies with secondary KADs appear

    r1 = res.filter(pl.col("company_id") == 1).to_dicts()[0]
    # Jan 1 to Aug 1 is 213 days (>= 180)
    assert r1["post_launch_expansion_days"] == 213
    assert r1["has_post_launch_expansion"] is True

    r2 = res.filter(pl.col("company_id") == 2).to_dicts()[0]
    # Jan 1 to Mar 1 is 60 days (< 180)
    assert r2["post_launch_expansion_days"] == 60
    assert r2["has_post_launch_expansion"] is False

    # Verify alias identical behavior
    res_alias = _compute_post_launch_expansion(companies, company_activities)
    assert res.sort("company_id").equals(res_alias.sort("company_id"))


def test_compute_seasonality_features() -> None:
    """Test cyclical month trigonometric projections and pre-summer launch indicator."""
    companies = pl.DataFrame({
        "company_id": [1, 2, 3],
        "incorporation_date": [date(2020, 3, 15), date(2020, 7, 20), None],
    })

    res = compute_seasonality_features(companies)
    assert res.height == 3
    assert "incorporation_month" in res.columns
    assert "inc_month_sin" in res.columns
    assert "inc_month_cos" in res.columns
    assert "is_pre_summer_launch" in res.columns

    rows = res.to_dicts()
    # Company 1: March (month 3) -> Pre-summer (months 3-6)
    assert rows[0]["incorporation_month"] == 3
    assert rows[0]["is_pre_summer_launch"] is True
    assert rows[0]["inc_month_sin"] == pytest.approx(1.0, abs=1e-5)
    assert rows[0]["inc_month_cos"] == pytest.approx(0.0, abs=1e-5)

    # Company 2: July (month 7) -> Not pre-summer
    assert rows[1]["incorporation_month"] == 7
    assert rows[1]["is_pre_summer_launch"] is False

    # Company 3: None -> defaults to month 1
    assert rows[2]["incorporation_month"] == 1
    assert rows[2]["is_pre_summer_launch"] is False

    # Verify alias identical behavior
    res_alias = _compute_seasonality_features(companies)
    assert res.equals(res_alias)


def test_compute_brand_features() -> None:
    """Test brand sophistication, modern hospitality keywords, and network size."""
    companies = pl.DataFrame({
        "company_id": [1, 2, 3, 4],
        "vat": ["111111111", "222222222", "333333333", "444444444"],
        "company_name_el": [
            "Επιχείρηση Εστίασης",
            "Artisan Coffee Roasters ΙΚΕ",
            "Artisan Coffee Roasters ΟΕ",
            "Bistro & Cocktail Bar Project",
        ],
    })

    res = compute_brand_features(companies)
    assert res.height == 4

    rows = {r["company_id"]: r for r in res.to_dicts()}

    # Company 1: Generic default
    assert rows[1]["has_distinctive_brand"] is False
    assert rows[1]["is_modern_concept"] is False

    # Company 2 & 3: Distinctive and modern keyword (coffee/roasters), shared network = 2
    assert rows[2]["has_distinctive_brand"] is True
    assert rows[2]["is_modern_concept"] is True
    assert rows[2]["shared_brand_network_size"] == 2
    assert rows[3]["shared_brand_network_size"] == 2

    # Company 4: Modern keywords (bistro, cocktail, bar, project)
    assert rows[4]["has_distinctive_brand"] is True
    assert rows[4]["is_modern_concept"] is True
    assert rows[4]["shared_brand_network_size"] == 1

    # Verify alias identical behavior
    res_alias = _compute_brand_features(companies)
    assert res.equals(res_alias)


def test_compute_municipality_typology() -> None:
    """Test Greek municipality demographic classification and coastal zoning."""
    companies = pl.DataFrame({
        "company_id": [1, 2, 3, 4, 5],
        "municipality_name": ["ΜΥΚΟΝΟΥ", "ΚΑΣΣΑΝΔΡΑΣ", "ΑΘΗΝΑΙΩΝ", "ΠΑΤΡΕΩΝ", "ΤΡΙΚΑΛΩΝ"],
        "prefecture_name": ["ΚΥΚΛΑΔΩΝ", "ΧΑΛΚΙΔΙΚΗΣ", "ΑΤΤΙΚΗΣ", "ΑΧΑΙΑΣ", "ΤΡΙΚΑΛΩΝ"],
    })

    res = compute_municipality_typology(companies)
    assert res.height == 5

    rows = {r["company_id"]: r for r in res.to_dicts()}

    # Mykonos -> Island Tourist, Coastal
    assert rows[1]["municipality_typology"] == "Island Tourist"
    assert rows[1]["is_coastal_zone"] is True

    # Kassandra -> Seasonal Coastal Resort, Coastal
    assert rows[2]["municipality_typology"] == "Seasonal Coastal Resort"
    assert rows[2]["is_coastal_zone"] is True

    # Athens -> Urban Metropolitan
    assert rows[3]["municipality_typology"] == "Urban Metropolitan"

    # Patras -> Regional Commercial City
    assert rows[4]["municipality_typology"] == "Regional Commercial City"

    # Trikala -> Provincial / Rural
    assert rows[5]["municipality_typology"] == "Provincial / Rural"

    # Verify alias identical behavior
    res_alias = _compute_municipality_typology(companies)
    assert res.equals(res_alias)


# =============================================================================
# 3. DYNAMIC INGESTION SIZE ASSERTIONS TESTS (Feature 8)
# =============================================================================

def test_dynamic_ingestion_size_checks() -> None:
    """Verify that print_sanity_check_report operates dynamically on non-standard sizes."""
    n = 10
    cids = list(range(1, n + 1))
    vats = [f"9999999{i:02d}" for i in range(n)]

    ec = pl.DataFrame({
        "company_id": cids,
        "vat": vats,
        "vat_total_locations": [1] * n,
        "is_client_vat": [True, True] + [False] * (n - 2),
        "is_client_representative": [True, True] + [False] * (n - 2),
        "is_branch": [0] * n,
        "prefecture_name": ["ΑΤΤΙΚΗΣ"] * n,
        "legal_type": ["ΙΚΕ"] * n,
        "latitude": [37.98] * n,
        "longitude": [23.72] * n,
        "has_gps": [True] * n,
        "is_sole_proprietorship": [False] * n,
        "primary_kad": ["56101100"] * n,
        "all_primary_kads": [["56101100"]] * n,
        "num_primary_kads": [1] * n,
        "total_kads": [1] * n,
    })

    clients = pl.DataFrame({
        "vat": vats[:2],
        "turnover": [12000.0, 15000.0],
    })

    ds = IngestedDataset(
        companies=ec,
        activities=pl.DataFrame({"activity_id": ["56101100"], "description": ["Εστιατόριο"]}),
        company_activities=pl.DataFrame({
            "company_id": cids,
            "activity_id": ["56101100"] * n,
            "activity_type": ["Κύρια"] * n,
            "activity_type_norm": ["Κύρια"] * n,
        }),
        clients=clients,
        enriched_companies=ec,
    )

    # Must pass without raising AssertionError (Feature 8 dynamic checks)
    print_sanity_check_report(ds)


# =============================================================================
# 4. CONTIGUOUS 2D NUMPY FEATURE STORE PERSISTENCE TESTS (Feature 9)
# =============================================================================

def test_feature_store_npz_persistence_roundtrip(tmp_path: Path) -> None:
    """Verify contiguous 2D NumPy array persistence, zero pickle dependency, and fast load."""
    n = 25
    cids = np.arange(1, n + 1, dtype=np.int64)
    vats = [f"0941234{i:02d}" for i in range(n)]

    fs = CompanyFeatureStore(
        company_ids=cids,
        vats=vats,
        semantic_vectors=np.random.randn(n, 64).astype(np.float32),
        semantic_svd=np.random.randn(n, 64).astype(np.float32),
        kad_hierarchy_features=np.random.randn(n, 32).astype(np.float32),
        spatial_features=np.random.randn(n, 16).astype(np.float32),
        legal_features=np.random.randn(n, 8).astype(np.float32),
        age_branch_features=np.random.randn(n, 12).astype(np.float32),
        unified_features=np.random.randn(n, 132).astype(np.float32),
        latlon_rad=np.random.randn(n, 2).astype(np.float64),
        has_gps=np.ones(n, dtype=bool),
        embedding_telemetry={"model_key": "test_model"},
        company_neural_svd=np.random.randn(n, 64).astype(np.float32),
        company_algebraic_svd=np.random.randn(n, 64).astype(np.float32),
    )

    npz_file = tmp_path / "test_universe.npz"
    model_col = "company_neural_svd_test_model"

    # 1. Save to NPZ
    save_feature_store_npz(fs, npz_file, model_col=model_col, compress=False)
    assert npz_file.exists()

    # 2. Load from NPZ using allow_pickle=False (MANDATORY security / compatibility check)
    t0 = time.perf_counter()
    loaded = load_feature_store_npz(npz_file, model_col=model_col)
    dt = time.perf_counter() - t0

    assert loaded is not None
    assert dt < 0.10, f"Expected load time < 0.10s, got {dt:.4f}s"

    # 3. Verify exact numerical and structural equality
    np.testing.assert_array_equal(loaded["company_id"], fs.company_ids)
    assert loaded["vat"] == fs.vats
    np.testing.assert_array_equal(loaded["has_gps"], fs.has_gps)
    np.testing.assert_allclose(loaded["company_algebraic_svd"], fs.company_algebraic_svd, rtol=1e-6)
    np.testing.assert_allclose(loaded["company_neural_svd"], fs.company_neural_svd, rtol=1e-6)
    np.testing.assert_allclose(loaded["spatial_features"], fs.spatial_features, rtol=1e-6)
    np.testing.assert_allclose(loaded["unified_features"], fs.unified_features, rtol=1e-6)
    np.testing.assert_allclose(loaded["latlon_rad"], fs.latlon_rad, rtol=1e-6)

    # 4. Verify C-contiguous layout
    assert loaded["company_algebraic_svd"].flags.c_contiguous
    assert loaded["spatial_features"].flags.c_contiguous
    assert loaded["unified_features"].flags.c_contiguous


# =============================================================================
# 5. BACKWARD-COMPATIBILITY SHIM TESTS
# =============================================================================

def test_backward_compatibility_reexports() -> None:
    """Verify that importing from `src.features` re-exports all required symbols."""
    import src.features as f

    required_symbols = [
        "CompanyFeatureStore",
        "EARTH_RADIUS_KM",
        "GREEK_TO_LATIN_TABLE",
        "GREEK_TRANSIT_FOOT_TRAFFIC_HUBS",
        "FALLBACK_GREEK_TRANSIT_HUBS",
        "load_greek_transit_hubs",
        "build_feature_store",
        "recombine_feature_store",
        "save_feature_store_npz",
        "load_feature_store_npz",
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
        "build_spatial_ckdtree_features",
        "build_structured_company_blocks",
        "build_external_enrichment",
        "normalize_greek_text",
        "transliterate_greek",
    ]

    for sym in required_symbols:
        assert hasattr(f, sym), f"Missing required symbol '{sym}' in src.features"


def test_compute_geo_regime() -> None:
    """Verify compute_geo_regime cleanly partitions into 4 expected regimes."""
    from src.features.engineering import compute_geo_regime

    sample_df = pl.DataFrame({
        "company_id": [1, 2, 3, 4],
        "municipality_name": ["ΑΘΗΝΑΙΩΝ", "ΜΥΚΟΝΟΥ", "ΚΑΣΣΑΝΔΡΑΣ", "ΛΑΡΙΣΑΙΩΝ"],
        "prefecture_name": ["ΑΤΤΙΚΗΣ", "ΚΥΚΛΑΔΩΝ", "ΧΑΛΚΙΔΙΚΗΣ", "ΛΑΡΙΣΑΣ"],
    })
    res = compute_geo_regime(sample_df)
    assert res.height == 4
    assert res["geo_regime"].to_list() == [
        "METRO_ATHENS_THESS",
        "ISLAND_TOURISM",
        "COASTAL_RESORT",
        "MAINLAND_REGIONAL",
    ]
