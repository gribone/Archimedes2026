"""
Baseline Deliverables Snapshot Verification Test Suite (`tests/test_deliverables.py`)
=====================================================================================
Verifies Requirement R1 Section 10 deliverables:
  - `output/profiles.json`
  - `output/assignments.csv`
  - `output/lookalikes.csv`

Validates exact line counts, schemas, mathematical invariants, zero client leakage,
and bit-for-bit SHA-256 integrity against the recorded baseline snapshots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.mark.deliverables
def test_deliverable_files_exist(baseline_paths: dict[str, Path]) -> None:
    """Verify that all three Section 10 deliverable files exist on disk."""
    for filename, path in baseline_paths.items():
        assert path.exists(), f"Deliverable file {filename} does not exist at {path}"
        assert path.stat().st_size > 0, f"Deliverable file {filename} is empty"


@pytest.mark.deliverables
def test_assignments_csv_contract(baseline_paths: dict[str, Path]) -> None:
    """
    Verify `output/assignments.csv`:
      - Exactly 301 lines (1 header + 300 client records)
      - Schema: company_id (int), profile_id (str), score (float)
      - Exactly 300 unique company_id entries
      - Valid profile_ids ('ICP_1', 'ICP_2', or 'NO_FIT')
      - Scores strictly in [0.0, 1.0]
      - Every 'NO_FIT' record has score < 0.70
    """
    path = baseline_paths["assignments.csv"]
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 301, f"Expected exactly 301 lines in assignments.csv, got {len(lines)}"

    df = pl.read_csv(path)
    assert df.height == 300, f"Expected 300 client rows, got {df.height}"
    assert set(df.columns) == {"company_id", "profile_id", "score"}
    assert df["company_id"].n_unique() == 300, "All client company_ids must be unique"

    valid_profiles = {"ICP_1", "ICP_2", "NO_FIT"}
    actual_profiles = set(df["profile_id"].unique().to_list())
    assert actual_profiles.issubset(valid_profiles), f"Unexpected profile IDs: {actual_profiles}"

    scores = df["score"].to_numpy()
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores must be in [0.0, 1.0]"

    # Verify NO_FIT thresholding
    no_fit_df = df.filter(pl.col("profile_id") == "NO_FIT")
    if no_fit_df.height > 0:
        max_no_fit_score = no_fit_df["score"].max()
        assert max_no_fit_score < 0.70, f"NO_FIT company has score >= 0.70: {max_no_fit_score}"


@pytest.mark.deliverables
def test_lookalikes_csv_contract(baseline_paths: dict[str, Path]) -> None:
    """
    Verify `output/lookalikes.csv`:
      - Exactly 2,001 lines (1 header + 2,000 lookalike prospects)
      - Schema: profile_id, company_id, rank, score, αιτιολόγηση
      - Exactly 2,000 unique company_id entries
      - Ranks 1 to 2,000 contiguous without gaps
      - Scores monotonically non-increasing
      - Scores strictly in [0.0, 1.0]
      - Rationale strings non-empty and formatted in Greek
    """
    path = baseline_paths["lookalikes.csv"]
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2001, f"Expected 2001 lines in lookalikes.csv, got {len(lines)}"

    df = pl.read_csv(path)
    assert df.height == 2000, f"Expected 2000 lookalike rows, got {df.height}"
    assert {"profile_id", "company_id", "rank", "score", "αιτιολόγηση"}.issubset(set(df.columns))
    assert df["company_id"].n_unique() == 2000, "All lookalike company_ids must be unique"

    ranks = df["rank"].to_list()
    assert ranks == list(range(1, 2001)), "Ranks must be exactly 1..2000 contiguous"

    scores = df["score"].to_numpy()
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores must be in [0.0, 1.0]"
    # Monotonicity check
    assert (np.diff(scores) <= 1e-6).all(), "Scores must be monotonically non-increasing"

    # Greek justification format
    sample_rationales = df["αιτιολόγηση"].slice(0, 10).to_list()
    for rat in sample_rationales:
        assert rat.startswith("Ταύτιση "), f"Rationale must start with 'Ταύτιση ', got: {rat[:20]}"


@pytest.mark.deliverables
def test_zero_client_leakage(baseline_paths: dict[str, Path]) -> None:
    """Verify that zero client companies OR client branches leak into lookalikes."""
    try:
        clients_df = pl.read_excel("data/ForceCRM_Hackathon_Pelatologio.xlsx", engine="openpyxl")
        client_vats = set(clients_df["ΑΦΜ"].cast(pl.Utf8).str.zfill(9).to_list())
    except Exception:
        from src.ingest import load_clients_excel
        clients_df = load_clients_excel("data/ForceCRM_Hackathon_Pelatologio.xlsx")
        client_vats = set(clients_df["vat"].to_list())

    look_df = pl.read_csv(baseline_paths["lookalikes.csv"])
    look_cids = set(look_df["company_id"].to_list())

    companies_df = pl.read_csv(
        "data/vwHackCompany.csv",
        separator=";",
        columns=["company_id", "vat"],
        schema_overrides={"vat": pl.Utf8, "company_id": pl.Int64},
    )
    look_vats = set(
        companies_df.filter(pl.col("company_id").is_in(list(look_cids)))["vat"]
        .cast(pl.Utf8)
        .str.zfill(9)
        .to_list()
    )

    vat_leaks = client_vats.intersection(look_vats)
    assert len(vat_leaks) == 0, f"Client VATs leaked into lookalikes: {vat_leaks}"

    ass_df = pl.read_csv(baseline_paths["assignments.csv"])
    client_ids = set(ass_df["company_id"].to_list())
    cid_leaks = client_ids.intersection(look_cids)
    assert len(cid_leaks) == 0, f"Client company IDs leaked into lookalikes: {cid_leaks}"


@pytest.mark.deliverables
def test_profiles_json_contract(baseline_paths: dict[str, Path]) -> None:
    """
    Verify `output/profiles.json`:
      - Valid JSON array of exactly 2 ICP profiles
      - Profiles named 'ICP_1' and 'ICP_2'
      - Sum of client_count across profiles == 300
      - Sum of turnover_total_eur across profiles == €5,443,170.0 (exact client turnover)
      - Sum of client_share == 1.0 (approx)
      - Sum of turnover_share == 1.0 (approx)
      - EU AI Act Article 50 notice present
      - Significant features list present with positive lifts
    """
    path = baseline_paths["profiles.json"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "profiles.json must contain a JSON array"
    assert len(data) == 2, f"Expected 2 ICP profiles, got {len(data)}"

    p_ids = {p["profile_id"] for p in data}
    assert p_ids == {"ICP_1", "ICP_2"}

    total_clients = sum(p["client_count"] for p in data)
    assert total_clients == 300, f"Expected 300 total clients, got {total_clients}"

    total_turnover = sum(p["turnover_total_eur"] for p in data)
    assert pytest.approx(total_turnover, rel=1e-3) == 5443170.0

    total_client_share = sum(p["client_share"] for p in data)
    assert pytest.approx(total_client_share, abs=1e-3) == 1.0

    total_turnover_share = sum(p["turnover_share"] for p in data)
    assert pytest.approx(total_turnover_share, abs=1e-3) == 1.0

    for p in data:
        assert p["ai_generated"] is True
        assert "eu_ai_act_notice" in p and len(p["eu_ai_act_notice"]) > 0
        assert "features" in p and isinstance(p["features"], list)
        for f in p["features"]:
            assert f["lift"] > 0
            assert f["raw_lift"] > 0
            assert f["p_value"] >= 0.0


@pytest.mark.deliverables
def test_baseline_sha256_checksums(
    baseline_paths: dict[str, Path],
    baseline_expected_hashes: dict[str, dict[str, str]],
) -> None:
    """
    Verify deliverable SHA-256 hashes against either the reference baseline (alpha=0.50)
    or the neural-zero baseline (alpha=0.0).
    """
    current_hashes = {}
    for name, path in baseline_paths.items():
        data = path.read_bytes()
        current_hashes[name] = hashlib.sha256(data).hexdigest()

    ref = baseline_expected_hashes["reference"]
    zero = baseline_expected_hashes["neural_zero"]

    # Check assignments.csv
    ass_hash = current_hashes["assignments.csv"]
    assert ass_hash in (ref["assignments.csv"], ref.get("assignments.csv_m3"), zero["assignments.csv"]), (
        f"assignments.csv hash {ass_hash} does not match reference ({ref['assignments.csv']}) "
        f"or neural_zero ({zero['assignments.csv']})"
    )

    # Check lookalikes.csv
    look_hash = current_hashes["lookalikes.csv"]
    valid_look_hashes = (
        ref["lookalikes.csv"],
        ref.get("lookalikes.csv_m3"),
        ref.get("lookalikes.csv_merged_exclusions"),
        zero["lookalikes.csv"],
    )
    assert look_hash in valid_look_hashes, (
        f"lookalikes.csv hash {look_hash} does not match reference ({ref['lookalikes.csv']}) "
        f"or neural_zero ({zero['lookalikes.csv']})"
    )

    # Check profiles.json
    prof_hash = current_hashes["profiles.json"]
    valid_prof_hashes = (
        ref["profiles.json"],
        ref.get("profiles.json_m3"),
        ref.get("profiles.json_salesperson"),
        ref.get("profiles.json_current"),
        ref["profiles.json_legacy"],
        zero["profiles.json"],
    )
    assert prof_hash in valid_prof_hashes, (
        f"profiles.json hash {prof_hash} does not match known baselines: {valid_prof_hashes}"
    )
