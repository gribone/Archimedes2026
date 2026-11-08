"""
tests/test_ingest.py
====================
Unit and integration tests for Archimedes2026 Stage 1 Ingestion Pipeline:
1. VAT zero-padding: 9 digits, leading zeros, float string trailing '.0' removal, whitespace stripping.
2. Canonical Primary KAD selection: '56*' priority, earliest date_from, lexicographical tie-breaking,
   null date handling, retention in all_primary_kads.
3. Client VAT branch isolation: is_client_vat == True across all branch records of client VATs,
   with exactly one canonical representative (is_client_representative == True).
"""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl
import polars as pl
import pytest

from src.ingest import (
    build_enriched_companies,
    ingest_all,
    load_clients_excel,
    load_companies,
)


# =============================================================================
# 1. VAT ZERO-PADDING & NUMBER SANITIZATION TESTS
# =============================================================================

@pytest.mark.ingest
def test_vat_zero_padding_synthetic_excel() -> None:
    """
    Verify that load_clients_excel correctly handles:
    - Normal 9-digit string VATs starting with zero.
    - 8-digit numeric and string VATs missing leading zero -> padded to 9 digits.
    - Floating point values and trailing '.0' (e.g. 35526279.0 -> '035526279').
    - Whitespace trimming.
    - Greek formatted turnover string parsing ('100.000,50' -> 100000.5).
    - Deduplication by summing turnover when duplicate VATs are encountered.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Clients"
    ws.append(["ΑΦΜ", "Τζίρος"])
    ws.append(["094123456", 50000])        # Already 9 digits with leading 0
    ws.append(["94123456", 60000])         # 8-digit string missing leading 0
    ws.append([94123456, 70000])           # Integer missing leading 0
    ws.append([94123456.0, 80000])         # Float cell value
    ws.append(["094123456.0", 90000])       # Float string with trailing .0
    ws.append(["  094123456  ", 10000])     # Whitespace padded
    ws.append(["12345", 15000])             # 5 digits -> 4 leading zeroes
    ws.append(["999999999", "125.450,75"])  # Greek formatted currency string

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    clients = load_clients_excel(buf)

    # Asserts
    assert clients.height == 3, f"Expected 3 unique VATs after deduplication, got {clients.height}"
    vats = clients["vat"].to_list()
    assert "094123456" in vats
    assert "000012345" in vats
    assert "999999999" in vats

    for vat in vats:
        assert len(vat) == 9, f"VAT {vat} is not 9 characters long"
        assert vat.isdigit(), f"VAT {vat} contains non-digit characters"
        assert not vat.endswith(".0"), f"VAT {vat} contains trailing .0"

    # Verify deduplicated turnover summation for 094123456
    # 50000 + 60000 + 70000 + 80000 + 90000 + 10000 = 360000.0
    turnover_094 = clients.filter(pl.col("vat") == "094123456")["turnover"][0]
    assert turnover_094 == pytest.approx(360000.0)

    # Verify Greek decimal parsing for 999999999
    turnover_999 = clients.filter(pl.col("vat") == "999999999")["turnover"][0]
    assert turnover_999 == pytest.approx(125450.75)


@pytest.mark.ingest
def test_vat_zero_padding_csv_companies(tmp_path: Path) -> None:
    """Verify that load_companies zero-pads VAT to 9 digits and handles Greek GPS commas."""
    csv_content = (
        "company_id;vat;company_name_el;legal_type;is_branch;street;street_number;"
        "zip_code;city;municipality_name;prefecture_name;incorporation_date;latitude;longitude\n"
        "101; 94123456 ;Test Alpha;ΑΕ;0;Stadiou;1;10564;Athens;ΑΘΗΝΑΙΩΝ;ΑΤΤΙΚΗΣ;2020-01-01;37,9845;23,7281\n"
        "102;084123456;Test Beta;ΙΚΕ;1;Ermou;2;10563;Athens;ΑΘΗΝΑΙΩΝ;ΑΤΤΙΚΗΣ;2021-02-02;37.9765;23.7291\n"
    )
    test_csv = tmp_path / "test_companies.csv"
    test_csv.write_text(csv_content, encoding="utf-8")

    companies = load_companies(test_csv)
    assert companies.height == 2

    # Check zero-padding
    assert companies["vat"][0] == "094123456"
    assert companies["vat"][1] == "084123456"

    # Check GPS coordinates parsed as Float64
    assert companies["latitude"][0] == pytest.approx(37.9845)
    assert companies["longitude"][0] == pytest.approx(23.7281)
    assert companies["has_gps"].to_list() == [True, True]


@pytest.mark.ingest
@pytest.mark.skipif(not Path("data/vwHackCompany.csv").exists(), reason="Real dataset not present")
def test_vat_zero_padding_real_datasets() -> None:
    """Verify that every VAT in real production CSV and Excel is strictly 9 digits."""
    dataset = ingest_all(data_dir="data")

    # 1. Clients Excel
    for vat in dataset.clients["vat"]:
        assert len(vat) == 9, f"Client VAT {vat} is not 9 digits"
        assert vat.isdigit(), f"Client VAT {vat} contains non-digits"

    # 2. Universe Companies CSV
    for vat in dataset.companies["vat"]:
        assert len(vat) == 9, f"Company VAT {vat} is not 9 digits"
        assert vat.isdigit(), f"Company VAT {vat} contains non-digits"

    # 3. All 300 client VATs match in enriched companies
    matched_vats = dataset.enriched_companies.filter(pl.col("is_client_vat"))["vat"].n_unique()
    assert matched_vats == 300, f"Expected 300 matched client VATs, got {matched_vats}"


# =============================================================================
# 2. CANONICAL PRIMARY KAD SELECTION TESTS
# =============================================================================

@pytest.mark.ingest
def test_canonical_primary_kad_selection_rules(
    sample_raw_companies_df: pl.DataFrame,
    sample_company_activities_df: pl.DataFrame,
    sample_activities_df: pl.DataFrame,
    sample_clients_df: pl.DataFrame,
) -> None:
    """
    Verify all priority rules in canonical primary KAD selection:
    1. Hospitality 56 priority: 56 KAD beats older non-56 KAD.
    2. Date tie-breaker: Earlier date_from wins between two 56 KADs.
    3. Lexicographical tie-breaker: Lower activity_id wins when dates match.
    4. Null date handling: Valid date beats null date (nulls_last=True).
    5. Non-56 sole KAD: Successfully chosen as primary.
    6. all_primary_kads preserves the complete list of primary KADs.
    """
    ec = build_enriched_companies(
        companies=sample_raw_companies_df,
        company_activities=sample_company_activities_df,
        activities=sample_activities_df,
        clients=sample_clients_df,
    )

    # 1. Company 1: Non-56 (47111000 from 2010) vs 56 (56101100 from 2020)
    # Hospitality 56 priority overrides earlier registration date!
    c1 = ec.filter(pl.col("company_id") == 1)
    assert c1["primary_kad"][0] == "56101100"
    assert c1["primary_kad_desc"][0] == "Υπηρεσίες εστιατορίου"
    assert set(c1["all_primary_kads"][0]) == {"47111000", "56101100"}
    assert c1["num_primary_kads"][0] == 2

    # 2. Company 3: 56101102 (2022-06-01) vs 56101101 (2018-03-15)
    # Earlier date wins!
    c3 = ec.filter(pl.col("company_id") == 3)
    assert c3["primary_kad"][0] == "56101101"
    assert c3["primary_kad_desc"][0] == "Υπηρεσίες παροχής γευμάτων"
    assert set(c3["all_primary_kads"][0]) == {"56101101", "56101102"}

    # 3. Company 5: 56300002 (2020-01-01) vs 56300001 (2020-01-01)
    # Same date -> Lexicographical order wins (56300001 < 56300002)!
    c5 = ec.filter(pl.col("company_id") == 5)
    assert c5["primary_kad"][0] == "56300001"
    assert c5["primary_kad_desc"][0] == "Υπηρεσίες καφενείου"

    # 4. Company 6: 47112000 (None date) vs 47111000 (2020-01-01)
    # Non-null date wins over None date!
    c6 = ec.filter(pl.col("company_id") == 6)
    assert c6["primary_kad"][0] == "47111000"

    # 5. Company 4: Single non-56 KAD (70220000)
    c4 = ec.filter(pl.col("company_id") == 4)
    assert c4["primary_kad"][0] == "70220000"
    assert c4["primary_kad_desc"][0] == "Δραστηριότητες παροχής επιχειρηματικών συμβουλών"


@pytest.mark.ingest
@pytest.mark.skipif(not Path("data/vwHackCompany.csv").exists(), reason="Real dataset not present")
def test_canonical_primary_kad_real_dataset_integrity() -> None:
    """Verify that across all 24,424 universe companies, 100% have a valid primary KAD."""
    dataset = ingest_all(data_dir="data")
    ec = dataset.enriched_companies

    # Zero companies with null primary KAD
    null_kads = ec.filter(pl.col("primary_kad").is_null()).height
    assert null_kads == 0, f"Found {null_kads} companies with null primary KAD"

    # 993 companies have >1 primary KAD due to 2026 NACE 2.1 transition
    multi_primary = ec.filter(pl.col("num_primary_kads") > 1)
    assert multi_primary.height == 993, f"Expected 993 multi-primary KAD companies, got {multi_primary.height}"

    # Verify all multi-primary companies have valid descriptions and list representations
    for row in multi_primary.select(["primary_kad", "all_primary_kads", "num_primary_kads"]).iter_rows(named=True):
        assert row["primary_kad"] in row["all_primary_kads"]
        assert len(row["all_primary_kads"]) == row["num_primary_kads"]


# =============================================================================
# 3. CLIENT VAT BRANCH ISOLATION & REPRESENTATIVE ARBITRATION TESTS
# =============================================================================

@pytest.mark.ingest
def test_client_vat_branch_isolation_and_representative(
    sample_raw_companies_df: pl.DataFrame,
    sample_company_activities_df: pl.DataFrame,
    sample_activities_df: pl.DataFrame,
    sample_clients_df: pl.DataFrame,
) -> None:
    """
    Verify client branch isolation rules:
    - is_client_vat == True for ALL records sharing a client VAT (both HQ and branches).
    - Exactly one canonical representative (is_client_representative == True) per client VAT.
    - Representative arbitration prefers HQ (is_branch == 0) over branches.
    - If no HQ exists (all branches), lowest company_id is selected.
    - Non-client companies have is_client_vat == False and is_client_representative == False.
    """
    ec = build_enriched_companies(
        companies=sample_raw_companies_df,
        company_activities=sample_company_activities_df,
        activities=sample_activities_df,
        clients=sample_clients_df,
    )

    # 1. VAT 094123456 (Company IDs 1, 2, 3: HQ is ID 1)
    client_alpha = ec.filter(pl.col("vat") == "094123456").sort("company_id")
    assert client_alpha.height == 3
    # All 3 records must be flagged as client VATs
    assert client_alpha["is_client_vat"].to_list() == [True, True, True]
    # Exactly company 1 (HQ, is_branch=0) must be the canonical representative
    assert client_alpha["is_client_representative"].to_list() == [True, False, False]

    # 2. VAT 077777777 (Company IDs 5, 6: both are branches with is_branch=1)
    client_gamma = ec.filter(pl.col("vat") == "077777777").sort("company_id")
    assert client_gamma.height == 2
    # All records must be flagged as client VATs
    assert client_gamma["is_client_vat"].to_list() == [True, True]
    # Lowest company_id (5 < 6) must be selected as representative
    assert client_gamma["is_client_representative"].to_list() == [True, False]

    # 3. VAT 088888888 (Company ID 4: non-client)
    non_client = ec.filter(pl.col("vat") == "088888888")
    assert non_client.height == 1
    assert non_client["is_client_vat"][0] is False
    assert non_client["is_client_representative"][0] is False


@pytest.mark.ingest
def test_client_representative_global_invariants(
    sample_raw_companies_df: pl.DataFrame,
    sample_company_activities_df: pl.DataFrame,
    sample_activities_df: pl.DataFrame,
    sample_clients_df: pl.DataFrame,
) -> None:
    """Verify mathematical and relational invariants on client flags."""
    ec = build_enriched_companies(
        companies=sample_raw_companies_df,
        company_activities=sample_company_activities_df,
        activities=sample_activities_df,
        clients=sample_clients_df,
    )

    n_client_vats = ec.filter(pl.col("is_client_vat"))["vat"].n_unique()
    n_client_reps = ec.filter(pl.col("is_client_representative")).height
    n_client_rows = ec.filter(pl.col("is_client_vat")).height

    # Invariant 1: Exactly one representative per distinct client VAT
    assert n_client_reps == n_client_vats == 2

    # Invariant 2: Total client rows >= total client representatives
    assert n_client_rows >= n_client_reps

    # Invariant 3: Every representative must have is_client_vat == True
    rep_non_clients = ec.filter(pl.col("is_client_representative") & (~pl.col("is_client_vat"))).height
    assert rep_non_clients == 0
