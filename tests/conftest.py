"""
ForceCRM Archimedes2026 Test Fixture Harness (`tests/conftest.py`)
==================================================================
Provides lightweight, deterministic shared fixtures for fast unit testing (<1s total)
and integration testing against Section 10 deliverable baselines and Starlette API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from starlette.testclient import TestClient

from src.config import PipelineConfig
from src.icp_engine import DiscoveredICP, StatisticalFeatureProof
from src.ingest import IngestedDataset


# -----------------------------------------------------------------------------
# 1. Project Directory & Baseline Deliverable Paths
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def output_dir(project_root: Path) -> Path:
    """Return the deliverables output directory."""
    return project_root / "output"


@pytest.fixture(scope="session")
def baseline_paths(output_dir: Path) -> dict[str, Path]:
    """Map deliverable names to their file paths."""
    return {
        "profiles.json": output_dir / "profiles.json",
        "assignments.csv": output_dir / "assignments.csv",
        "lookalikes.csv": output_dir / "lookalikes.csv",
    }


@pytest.fixture(scope="session")
def baseline_expected_hashes() -> dict[str, dict[str, str]]:
    """Expected SHA-256 hashes for reference and neural-zero baseline deliverables."""
    return {
        "reference": {
            "assignments.csv": "049af7fad63302ac6e2cc666d1c4419207a951c6dcab31634d2042c054ca4baa",
            "assignments.csv_m3": "6497015014937e42deed39bbb095bfecd887f44dfc271d68b3e695637913ff94",
            "lookalikes.csv": "0c5ac09a30c9aa9c562636b9a5262335bfe000be42fa3932f47b55650438aaa8",
            "lookalikes.csv_m3": "41771699f3e0473cee2801b05f925e14f88d9262c6f73eb0f87117c27fc140c2",
            "lookalikes.csv_merged_exclusions": "d10d82592b42c0a78b750fa2c909019ed22b95fe5bb2642559b577912e391ca5",
            "profiles.json": "2a400a4660c83c5837a1d5276f509e3abbc406b61293586a030f03e4ded2b9ec",
            "profiles.json_m3": "b4bd81761374e8cbe968f04b754ac8ca07aefe1823223bd21da86760cd114380",
            "profiles.json_salesperson": "9e9b8bd1f8c870fcd75f9a360263066816437f2b4400267ed6a8544476e3c0fd",
            "profiles.json_current": "8e7dbcfe0b9222cb05bcd9a95d1992e1054996be6fde4ee200299fac03fc327e",
            "profiles.json_legacy": "c27312cb2bd7d449144eae6ae836ab01a50e9932673ddb01e846f20cb6ccde1f",
        },
        "neural_zero": {
            "assignments.csv": "beeb675fed4345e1b1015cc50c077adbd8541766ff9b234a559f5caed8ba6efd",
            "lookalikes.csv": "31b89c03e94e4657f3941db1946052e708fe7ae2af8781125f6ba94a5c3d5e5c",
            "profiles.json": "6861460b322b5676603fb75e04c72547b7b6af3630c46c40b43e1d7892b392d6",
        },
    }


# -----------------------------------------------------------------------------
# 2. Fast Pipeline Configuration Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def fast_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Return a minimal PipelineConfig configured for deterministic, ultra-fast test execution."""
    return PipelineConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        kmeans_n_init=2,
        n_bootstrap_runs=2,
        min_icps=1,
        max_icps=3,
        use_sha256_client_cache=False,
        use_embedding_models=False,
        embedding_model="char-ngram-svd",
        hybrid_alpha=0.0,
    )


# -----------------------------------------------------------------------------
# 3. Synthetic Ingestion Data Fixtures (Zero Disk I/O, Instant Execution)
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_raw_companies_df() -> pl.DataFrame:
    """Synthetic company records containing VAT formatting, multi-branch, and geographic variants."""
    from datetime import date

    return pl.DataFrame(
        {
            "company_id": [1, 2, 3, 4, 5, 6],
            "vat": ["094123456", "094123456", "094123456", "088888888", "077777777", "077777777"],
            "company_name_el": [
                "Alpha HQ",
                "Alpha Branch 1",
                "Alpha Branch 2",
                "Beta Restaurant",
                "Gamma Branch A",
                "Gamma Branch B",
            ],
            "legal_type": ["ΑΕ", "ΑΕ", "ΑΕ", "ΙΚΕ", "ΟΕ", "ΟΕ"],
            "is_branch": [0, 1, 1, 0, 1, 1],  # Gamma has only branches, no HQ
            "street": ["Stadiou", "Tsimiski", "Kifisias", "Ermou", "Mitropoleos", "Egnatia"],
            "street_number": ["10", "20", "30", "40", "50", "60"],
            "zip_code": ["10564", "54624", "15124", "10563", "54623", "54622"],
            "city": ["Athens", "Thessaloniki", "Marousi", "Athens", "Thessaloniki", "Thessaloniki"],
            "municipality_name": [
                "ΑΘΗΝΑΙΩΝ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
                "ΑΜΑΡΟΥΣΙΟΥ",
                "ΑΘΗΝΑΙΩΝ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
            ],
            "prefecture_name": [
                "ΑΤΤΙΚΗΣ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
                "ΑΤΤΙΚΗΣ",
                "ΑΤΤΙΚΗΣ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
                "ΘΕΣΣΑΛΟΝΙΚΗΣ",
            ],
            "incorporation_date": [
                date(2015, 1, 15),
                date(2017, 3, 20),
                date(2019, 5, 10),
                date(2021, 6, 1),
                date(2018, 9, 12),
                date(2019, 11, 5),
            ],
            "latitude": [37.9792, 40.6322, 38.0489, 37.9765, 40.6311, 40.6345],
            "longitude": [23.7314, 22.9421, 23.8062, 23.7291, 22.9432, 22.9410],
            "company_age_years": [11.8, 9.6, 7.5, 5.4, 8.1, 7.0],
            "has_gps": [True, True, True, True, True, True],
            "is_sole_proprietorship": [False, False, False, False, False, False],
            "branch_count": [2, 2, 2, 0, 1, 1],
        }
    )


@pytest.fixture
def sample_company_activities_df() -> pl.DataFrame:
    """Synthetic company activities demonstrating multi-KAD, date priority, and 56-priority."""
    from datetime import date

    return pl.DataFrame(
        {
            "company_id": [
                1, 1,
                2,
                3, 3,
                4,
                5, 5,
                6, 6,
            ],
            "activity_id": [
                "47111000", "56101100",  # Company 1: retail (2010) vs restaurant (2020) -> 56101100 must win
                "56101100",              # Company 2
                "56101102", "56101101",  # Company 3: 2022 vs 2018 -> 56101101 must win
                "70220000",              # Company 4: non-56 sole primary
                "56300002", "56300001",  # Company 5: same date -> 56300001 must win lexicographically
                "47112000", "47111000",  # Company 6: null date vs 2020-01-01 -> 47111000 must win
            ],
            "activity_type": ["Κύρια"] * 10,
            "activity_type_norm": ["Κύρια"] * 10,
            "date_from": [
                date(2010, 1, 1), date(2020, 1, 1),
                date(2017, 3, 20),
                date(2022, 6, 1), date(2018, 3, 15),
                date(2021, 6, 1),
                date(2020, 1, 1), date(2020, 1, 1),
                None, date(2020, 1, 1),
            ],
        }
    )


@pytest.fixture
def sample_activities_df() -> pl.DataFrame:
    """KAD definitions dictionary."""
    return pl.DataFrame(
        {
            "activity_id": [
                "47111000",
                "47112000",
                "56101100",
                "56101101",
                "56101102",
                "56300001",
                "56300002",
                "70220000",
            ],
            "description": [
                "Λιανικό εμπόριο τροφίμων",
                "Λιανικό εμπόριο ποτών",
                "Υπηρεσίες εστιατορίου",
                "Υπηρεσίες παροχής γευμάτων",
                "Υπηρεσίες ψητοπωλείου",
                "Υπηρεσίες καφενείου",
                "Υπηρεσίες καφετέριας",
                "Δραστηριότητες παροχής επιχειρηματικών συμβουλών",
            ],
        }
    )


@pytest.fixture
def sample_clients_df() -> pl.DataFrame:
    """Client pelatologio matching VAT 094123456 (3 branches) and 077777777 (2 branches)."""
    return pl.DataFrame(
        {
            "vat": ["094123456", "077777777"],
            "turnover": [250000.0, 120000.0],
        }
    )


@pytest.fixture
def sample_ingested_dataset(
    sample_raw_companies_df: pl.DataFrame,
    sample_company_activities_df: pl.DataFrame,
    sample_activities_df: pl.DataFrame,
    sample_clients_df: pl.DataFrame,
) -> IngestedDataset:
    """Return an IngestedDataset assembled purely in-memory from synthetic fixtures."""
    joined = sample_raw_companies_df.join(
        sample_clients_df.select(["vat", "turnover"]),
        on="vat",
        how="left",
    ).with_columns(
        pl.col("turnover").is_not_null().alias("is_client_vat"),
        (pl.col("turnover").is_not_null() & (pl.col("is_branch") == 0)).alias("is_client_representative"),
        pl.lit("56101101").alias("primary_kad"),
        pl.lit("Υπηρεσίες εστιατορίου").alias("primary_kad_desc"),
        pl.lit("56").alias("kad_2digit"),
        pl.lit("5610").alias("kad_4digit"),
        pl.lit(True).alias("has_gps"),
        pl.lit(False).alias("is_sole_proprietorship"),
        pl.lit(1).alias("vat_total_locations"),
        pl.lit(1).alias("branch_count"),
        pl.lit(3.0).alias("company_age_years"),
        pl.lit("Urban Metropolitan").alias("municipality_typology"),
        pl.lit(False).alias("is_coastal_zone"),
        pl.lit(0.85).alias("purchasing_power_index"),
        pl.lit(5.0).alias("dist_to_transit_hub_km"),
        pl.lit(1).alias("is_transit_hub_catchment"),
    )

    return IngestedDataset(
        companies=sample_raw_companies_df,
        company_activities=sample_company_activities_df,
        activities=sample_activities_df,
        clients=sample_clients_df,
        enriched_companies=joined,
    )


# -----------------------------------------------------------------------------
# 4. Mathematical Engine & Scoring Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_feature_matrix() -> np.ndarray:
    """Return an L2-normalized synthetic feature matrix of shape (20, 8)."""
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal((20, 8)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


@pytest.fixture
def synthetic_discovered_icps() -> list[DiscoveredICP]:
    """Return 2 mock DiscoveredICP instances with valid statistical proofs."""
    rng = np.random.default_rng(seed=42)
    c1 = rng.standard_normal(8).astype(np.float32)
    c1 /= np.linalg.norm(c1)
    c2 = rng.standard_normal(8).astype(np.float32)
    c2 /= np.linalg.norm(c2)

    proof1 = StatisticalFeatureProof(
        feature_type="legal_type",
        feature_value="ΙΚΕ",
        feature_label_el="Νομική Μορφή: ΙΚΕ",
        icp_count=80,
        icp_share=0.80,
        universe_share=0.15,
        raw_lift=5.33,
        eb_lift=4.90,
        p_value_raw=1e-10,
        p_value_fdr=1e-9,
        is_significant=True,
    )

    icp1 = DiscoveredICP(
        profile_id="ICP_1",
        title_el="Εταιρικές Επιχειρήσεις Εστίασης (ΙΚΕ)",
        description_el="Προφίλ ΙΚΕ εστίασης υψηλής συγκέντρωσης",
        ai_description_el="Περιγραφή AI ICP 1",
        client_count=115,
        client_share=0.3833,
        turnover_total=3130260.0,
        turnover_share=0.5751,
        turnover_median=24610.0,
        turnover_mean=27219.65,
        stability_score=0.9688,
        confidence_score=0.9248,
        centroid_vector=c1,
        spatial_medoid_latlon_deg=(37.98, 23.73),
        significant_features=[proof1],
        eb_log_lifts={("legal_type", "ΙΚΕ"): 1.589, ("prefecture_name", "ΑΤΤΙΚΗΣ"): 0.85},
    )

    icp2 = DiscoveredICP(
        profile_id="ICP_2",
        title_el="Περιφερειακές Επιχειρήσεις Εστίασης",
        description_el="Προφίλ περιφερειακών επιχειρήσεων",
        ai_description_el="Περιγραφή AI ICP 2",
        client_count=171,
        client_share=0.5700,
        turnover_total=2312910.0,
        turnover_share=0.4249,
        turnover_median=12500.0,
        turnover_mean=13525.78,
        stability_score=0.9520,
        confidence_score=0.9010,
        centroid_vector=c2,
        spatial_medoid_latlon_deg=(40.64, 22.94),
        significant_features=[],
        eb_log_lifts={("prefecture_name", "ΘΕΣΣΑΛΟΝΙΚΗΣ"): 1.25},
    )

    return [icp1, icp2]


# -----------------------------------------------------------------------------
# 5. Starlette TestClient & Mock Service Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_company_1_record() -> dict[str, Any]:
    """Synthetic company record returned for company_id=1 to satisfy HTTP 200 contract."""
    return {
        "company": {
            "company_id": 1,
            "vat": "012345678",
            "name": "ΔΟΚΙΜΑΣΤΙΚΗ ΕΠΙΧΕΙΡΗΣΗ ΕΣΤΙΑΣΗΣ Α.Ε.",
            "company_name_el": "ΔΟΚΙΜΑΣΤΙΚΗ ΕΠΙΧΕΙΡΗΣΗ ΕΣΤΙΑΣΗΣ Α.Ε.",
            "legal_type": "ΑΕ",
            "prefecture_name": "ΑΤΤΙΚΗΣ",
            "municipality_name": "ΑΘΗΝΑΙΩΝ",
            "primary_kad": "56101101",
            "primary_kad_desc": "Υπηρεσίες παροχής γευμάτων από εστιατόριο",
        },
        "kads": [
            {
                "activity_type": "Κύρια",
                "activity_id": "56101101",
                "description": "Υπηρεσίες παροχής γευμάτων από εστιατόριο",
                "date_from": "2020-01-01",
            }
        ],
        "context": {
            "is_excluded": False,
            "verification": {
                "vat": "012345678",
                "name": "ΔΟΚΙΜΑΣΤΙΚΗ ΕΠΙΧΕΙΡΗΣΗ ΕΣΤΙΑΣΗΣ Α.Ε.",
                "status": "OPERATIONAL",
                "is_closed": False,
                "confidence": 0.95,
            },
        },
    }


@pytest.fixture
def valid_dataset_company_id() -> int:
    """Return a company ID known to exist in the real dataset."""
    return 92460


@pytest.fixture
def test_client(monkeypatch: pytest.MonkeyPatch, mock_company_1_record: dict[str, Any]) -> TestClient:
    """
    Provide an initialized Starlette TestClient with company_id=1 fallback
    ensuring /api/health, /api/bootstrap, /api/company/1, and /api/run
    respond quickly with HTTP 200.
    """
    from src.server import app
    from src.service import EngineService

    orig_company = EngineService.company

    def patched_company(self: EngineService, company_id: int) -> dict[str, Any] | None:
        if company_id == 1:
            return mock_company_1_record
        return orig_company(self, company_id)

    monkeypatch.setattr(EngineService, "company", patched_company)
    return TestClient(app)
