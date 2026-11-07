"""
Stage 1: Data Ingestion & Sanity Checks (`src/ingest.py`)
=========================================================
Fast, deterministic ingestion of ForceCRM Hackathon CSVs and Client Excel (`Πελατολόγιο`)
using Polars / DuckDB / OpenPyXL.

Key responsibilities:
1. Parse `;`-delimited UTF-8 CSVs with BOM handling, `"NULL"` strings, and `,` decimal floats.
2. Preserve leading zeros on `vat` (9-digit ΑΦΜ), `activity_id` (8-digit ΚΑΔ), and `zip_code`.
3. Compute `vat_total_locations` and `branch_count` per VAT across the entire Universe.
4. Resolve the Multi-Branch VAT trap: flag ALL `company_id`s belonging to any client VAT
   so lookalike rankings strictly exclude every branch of existing clients.
5. Handle the multi-Primary-KAD quirk (993 companies have >1 `Κύρια` KAD due to the 2026-03-01
   KAD 2025 / NACE 2.1 transition, plus 1 `Λοιπή` activity_type row).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO

import openpyxl
import polars as pl


REFERENCE_DATE = date(2026, 11, 1)


@dataclass
class IngestedDataset:
    """Container holding cleaned Polars DataFrames and sanity-check metadata."""

    companies: pl.DataFrame
    activities: pl.DataFrame
    company_activities: pl.DataFrame
    clients: pl.DataFrame
    enriched_companies: pl.DataFrame


def load_activities(csv_path: str | Path = "data/vwHackActivities.csv") -> pl.DataFrame:
    """Load `vwHackActivities.csv` (8-digit KAD codes and Greek descriptions)."""
    df = pl.read_csv(
        csv_path,
        separator=";",
        null_values=["NULL", ""],
        schema_overrides={
            "activity_id": pl.Utf8,
            "description": pl.Utf8,
        },
         encoding="utf8",
    )
    # Strip any BOM from first column name if present and clean whitespace
    rename_map = {col: col.lstrip("\ufeff").strip() for col in df.columns}
    df = df.rename(rename_map).with_columns(
        pl.col("activity_id").str.strip_chars().str.zfill(8),
        pl.col("description").str.strip_chars(),
    )
    return df.unique(subset=["activity_id"], keep="first")


def load_company_activities(
    csv_path: str | Path = "data/vwHackCompanyActivity.csv",
) -> pl.DataFrame:
    """Load `vwHackCompanyActivity.csv` and normalize activity types."""
    df = pl.read_csv(
        csv_path,
        separator=";",
        null_values=["NULL", ""],
        schema_overrides={
            "company_id": pl.Int64,
            "activity_id": pl.Utf8,
            "activity_type": pl.Utf8,
            "date_from": pl.Utf8,
        },
        encoding="utf8",
    )
    rename_map = {col: col.lstrip("\ufeff").strip() for col in df.columns}
    df = df.rename(rename_map).with_columns(
        pl.col("activity_id").str.strip_chars().str.zfill(8),
        pl.col("activity_type").str.strip_chars(),
        pl.col("date_from").str.strip_chars().str.to_date("%Y-%m-%d", strict=False),
    )
    # Normalize the single 'Λοιπή' row into 'Βοηθητική' while keeping raw type if needed
    df = df.with_columns(
        pl.when(pl.col("activity_type") == "Λοιπή")
        .then(pl.lit("Βοηθητική"))
        .otherwise(pl.col("activity_type"))
        .alias("activity_type_norm")
    )
    return df


def load_companies(csv_path: str | Path = "data/vwHackCompany.csv") -> pl.DataFrame:
    """
    Load `vwHackCompany.csv`, converting comma decimals in GPS coordinates,
    preserving leading zeros in `vat`, and computing `branch_count` per VAT.
    """
    df = pl.read_csv(
        csv_path,
        separator=";",
        null_values=["NULL", ""],
        schema_overrides={
            "company_id": pl.Int64,
            "vat": pl.Utf8,
            "company_name_el": pl.Utf8,
            "legal_type": pl.Utf8,
            "is_branch": pl.Int64,
            "street": pl.Utf8,
            "street_number": pl.Utf8,
            "zip_code": pl.Utf8,
            "city": pl.Utf8,
            "municipality_name": pl.Utf8,
            "prefecture_name": pl.Utf8,
            "incorporation_date": pl.Utf8,
            "latitude": pl.Utf8,
            "longitude": pl.Utf8,
        },
        encoding="utf8",
    )
    rename_map = {col: col.lstrip("\ufeff").strip() for col in df.columns}
    df = df.rename(rename_map)

    df = df.with_columns(
        pl.col("vat").str.strip_chars().str.zfill(9),
        pl.col("company_name_el").str.strip_chars().fill_null("Επιχείρηση Εστίασης"),
        pl.col("legal_type").str.strip_chars(),
        pl.col("is_branch").fill_null(0).cast(pl.Int64),
        pl.col("zip_code").str.strip_chars(),
        pl.col("city").str.strip_chars(),
        pl.col("municipality_name").str.strip_chars(),
        pl.col("prefecture_name").str.strip_chars(),
        pl.col("incorporation_date")
        .str.strip_chars()
        .str.to_date("%Y-%m-%d", strict=False),
        pl.col("latitude")
        .str.strip_chars()
        .str.replace(",", ".")
        .cast(pl.Float64, strict=False),
        pl.col("longitude")
        .str.strip_chars()
        .str.replace(",", ".")
        .cast(pl.Float64, strict=False),
    )

    # Calculate company age in years relative to REFERENCE_DATE (Nov 2026)
    df = df.with_columns(
        (
            (pl.lit(REFERENCE_DATE) - pl.col("incorporation_date")).dt.total_days()
            / 365.25
        )
        .clip(lower_bound=0.0)
        .alias("company_age_years"),
        (pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()).alias(
            "has_gps"
        ),
        (pl.col("legal_type") == "ΑΤΟΜΙΚΗ").alias("is_sole_proprietorship"),
    )

    # Compute VAT-level multi-branch features across the Universe
    vat_stats = df.group_by("vat").agg(
        pl.len().alias("vat_total_locations"),
        pl.col("is_branch").sum().alias("vat_branch_count"),
    )
    df = df.join(vat_stats, on="vat", how="left").with_columns(
        # Total extra locations or branches associated with this VAT
        pl.max_horizontal(
            pl.col("vat_branch_count"),
            pl.col("vat_total_locations") - 1,
        ).alias("branch_count")
    )
    return df


def load_clients_excel(
    excel_source: str | Path | bytes | BinaryIO = "data/ForceCRM_Hackathon_Pelatologio.xlsx",
) -> pl.DataFrame:
    """
    Load Client Excel (`Πελατολόγιο`) containing VAT (`ΑΦΜ`) and indicative `turnover` (`Τζίρος`).
    Supports file path, raw bytes, or binary file-like object (for Streamlit file_uploader).
    """
    if isinstance(excel_source, bytes):
        wb = openpyxl.load_workbook(io.BytesIO(excel_source), read_only=True, data_only=True)
    else:
        wb = openpyxl.load_workbook(excel_source, read_only=True, data_only=True)

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        raise ValueError("Client Excel file is empty.")

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        raw_vat = str(row[0]).strip()
        if not raw_vat:
            continue
        # Handle float-formatted VATs like '35526279.0' if Excel ever saved as number
        if raw_vat.endswith(".0"):
            raw_vat = raw_vat[:-2]
        vat = raw_vat.zfill(9)

        raw_turnover = row[1] if len(row) > 1 else 0.0
        if isinstance(raw_turnover, str):
            raw_turnover = float(raw_turnover.replace(".", "").replace(",", ".").strip() or 0.0)
        turnover = float(raw_turnover or 0.0)
        records.append({"vat": vat, "turnover": turnover})

    clients_df = pl.DataFrame(
        records, schema={"vat": pl.Utf8, "turnover": pl.Float64}
    )
    # Deduplicate by summing turnover if a VAT appears more than once in user upload
    clients_df = clients_df.group_by("vat", maintain_order=True).agg(
        pl.col("turnover").sum().alias("turnover")
    )
    return clients_df


def build_enriched_companies(
    companies: pl.DataFrame,
    company_activities: pl.DataFrame,
    activities: pl.DataFrame,
    clients: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join companies with summarized activity features and client status.
    - Selects canonical Primary KAD (`primary_kad`): prioritizes `Κύρια` starting with '56'
      (hospitality main KAD), and earliest `date_from` if a company has multiple `Κύρια`.
    - Also retains `all_primary_kads` list so no `Κύρια` KAD is lost.
    - Flags ALL `company_id`s belonging to a client VAT (`is_client_vat = True`).
    """
    ca_with_desc = company_activities.join(activities, on="activity_id", how="left")

    # Canonical primary KAD selection per company_id
    primary_rows = (
        ca_with_desc.filter(pl.col("activity_type") == "Κύρια")
        .with_columns(
            pl.col("activity_id").str.starts_with("56").alias("_is_kad56"),
        )
        .sort(
            by=["company_id", "_is_kad56", "date_from", "activity_id"],
            descending=[False, True, False, False],
            nulls_last=True,
        )
    )

    canonical_primary = primary_rows.group_by("company_id", maintain_order=True).agg(
        pl.col("activity_id").first().alias("primary_kad"),
        pl.col("description").first().alias("primary_kad_desc"),
        pl.col("activity_id").unique(maintain_order=True).alias("all_primary_kads"),
    )

    # Counts of KADs per company by type
    kad_counts = ca_with_desc.group_by("company_id").agg(
        pl.len().alias("total_kads"),
        (pl.col("activity_type") == "Κύρια").sum().alias("num_primary_kads"),
        (pl.col("activity_type") == "Δευτερεύουσα").sum().alias("num_secondary_kads"),
        (pl.col("activity_type_norm") == "Βοηθητική").sum().alias("num_auxiliary_kads"),
        pl.col("activity_id").unique(maintain_order=True).alias("all_kads"),
    )

    enriched = (
        companies.join(canonical_primary, on="company_id", how="left")
        .join(kad_counts, on="company_id", how="left")
        .join(clients, on="vat", how="left")
        .with_columns(
            pl.col("turnover").is_not_null().alias("is_client_vat"),
        )
    )

    # If a client VAT maps to multiple company_id rows (e.g. HQ + branch),
    # pick one canonical representative company_id per VAT (preferring is_branch == 0, then lowest company_id)
    # while keeping `is_client_vat == True` on ALL branches so they are excluded from lookalikes!
    client_rep = (
        enriched.filter(pl.col("is_client_vat"))
        .sort(["vat", "is_branch", "company_id"], descending=[False, False, False])
        .group_by("vat", maintain_order=True)
        .agg(pl.col("company_id").first().alias("_rep_company_id"))
    )

    enriched = (
        enriched.join(client_rep, on="vat", how="left")
        .with_columns(
            (
                pl.col("is_client_vat")
                & (pl.col("company_id") == pl.col("_rep_company_id"))
            ).alias("is_client_representative")
        )
        .drop("_rep_company_id")
    )

    return enriched


def ingest_all(
    data_dir: str | Path = "data",
    excel_source: str | Path | bytes | BinaryIO | None = None,
) -> IngestedDataset:
    """Run full Stage 1 ingestion pipeline and return structured dataset."""
    data_path = Path(data_dir)
    if excel_source is None:
        excel_source = data_path / "ForceCRM_Hackathon_Pelatologio.xlsx"

    activities = load_activities(data_path / "vwHackActivities.csv")
    company_activities = load_company_activities(data_path / "vwHackCompanyActivity.csv")
    companies = load_companies(data_path / "vwHackCompany.csv")
    clients = load_clients_excel(excel_source)
    enriched_companies = build_enriched_companies(
        companies=companies,
        company_activities=company_activities,
        activities=activities,
        clients=clients,
    )

    return IngestedDataset(
        companies=companies,
        activities=activities,
        company_activities=company_activities,
        clients=clients,
        enriched_companies=enriched_companies,
    )


def print_sanity_check_report(ds: IngestedDataset) -> None:
    """Print comprehensive sanity-check diagnostics for verification."""
    ec = ds.enriched_companies
    clients = ds.clients

    total_companies = ec.height
    unique_vats = ec["vat"].n_unique()
    multi_loc_vats = ec.filter(pl.col("vat_total_locations") > 1)["vat"].n_unique()
    branch_rows = ec.filter(pl.col("is_branch") == 1).height
    gps_count = ec.filter(pl.col("has_gps")).height
    gps_pct = 100.0 * gps_count / total_companies

    matched_client_vats = ec.filter(pl.col("is_client_vat"))["vat"].n_unique()
    matched_client_rows = ec.filter(pl.col("is_client_vat")).height
    rep_client_rows = ec.filter(pl.col("is_client_representative")).height
    client_branches = ec.filter(pl.col("is_client_vat") & (pl.col("is_branch") == 1)).height

    multi_prim = ec.filter(pl.col("num_primary_kads") > 1).height
    avg_kads = ec["total_kads"].mean()
    max_kads = ec["total_kads"].max()

    print("=" * 72)
    print("STAGE 1: DATA INGESTION & SANITY CHECK REPORT")
    print("=" * 72)
    print(f"[Universe Summary]")
    print(f"  - Total company_id records       : {total_companies:,} (expected: 24,424)")
    print(f"  - Unique VATs (ΑΦΜ)              : {unique_vats:,}")
    print(f"  - Multi-location VATs            : {multi_loc_vats:,}")
    print(f"  - Branch records (is_branch=1)   : {branch_rows:,}")
    print(f"  - GPS coverage                   : {gps_count:,} / {total_companies:,} ({gps_pct:.2f}%)")
    print(f"  - Sole Proprietorships (ΑΤΟΜΙΚΗ) : {ec.filter(pl.col('is_sole_proprietorship')).height:,} (EU AI Act natural persons)")
    print()
    print(f"[Activities & KAD Summary]")
    print(f"  - Unique KAD definitions         : {ds.activities.height:,}")
    print(f"  - Company-Activity links         : {ds.company_activities.height:,}")
    print(f"  - Companies with 0 Primary KADs  : {ec.filter(pl.col('primary_kad').is_null()).height}")
    print(f"  - Companies with >1 Primary KADs : {multi_prim:,} (2026 NACE 2.1 transition)")
    print(f"  - KADs per company (mean / max)  : {avg_kads:.2f} / {max_kads}")
    print()
    print(f"[Client Pelatologio Summary]")
    print(f"  - Client VATs in Excel           : {clients.height:,}")
    print(f"  - Matched Client VATs in Universe: {matched_client_vats:,}")
    print(f"  - Total client company_id rows   : {matched_client_rows:,} (all excluded from lookalikes)")
    print(f"  - Canonical client reps          : {rep_client_rows:,}")
    print(f"  - Client rows with is_branch=1   : {client_branches}")
    print(
        f"  - Client Turnover (€)            : min={clients['turnover'].min():,.0f} | "
        f"median={clients['turnover'].median():,.0f} | "
        f"mean={clients['turnover'].mean():,.0f} | "
        f"max={clients['turnover'].max():,.0f}"
    )
    print()
    print("[Top 5 Client Prefectures vs Universe]")
    client_pref = (
        ec.filter(pl.col("is_client_representative"))
        .group_by("prefecture_name")
        .agg(pl.len().alias("client_count"))
        .sort("client_count", descending=True)
        .head(5)
    )
    univ_pref = ec.group_by("prefecture_name").agg(pl.len().alias("univ_count"))
    pref_comp = client_pref.join(univ_pref, on="prefecture_name", how="left").with_columns(
        ((pl.col("client_count") / 300.0) / (pl.col("univ_count") / total_companies)).alias("raw_lift")
    )
    for row in pref_comp.iter_rows(named=True):
        print(
            f"  - {row['prefecture_name']:<18}: {row['client_count']:>3} clients "
            f"({100*row['client_count']/300:.1f}%) vs {row['univ_count']:>5} univ "
            f"({100*row['univ_count']/total_companies:.1f}%) | Lift = {row['raw_lift']:.2f}x"
        )

    print()
    print("[Client Legal Types vs Universe]")
    client_lt = (
        ec.filter(pl.col("is_client_representative"))
        .group_by("legal_type")
        .agg(pl.len().alias("client_count"))
        .sort("client_count", descending=True)
    )
    univ_lt = ec.group_by("legal_type").agg(pl.len().alias("univ_count"))
    lt_comp = client_lt.join(univ_lt, on="legal_type", how="left").with_columns(
        ((pl.col("client_count") / 300.0) / (pl.col("univ_count") / total_companies)).alias("raw_lift")
    )
    for row in lt_comp.iter_rows(named=True):
        print(
            f"  - {row['legal_type']:<18}: {row['client_count']:>3} clients "
            f"({100*row['client_count']/300:.1f}%) vs {row['univ_count']:>5} univ "
            f"({100*row['univ_count']/total_companies:.1f}%) | Lift = {row['raw_lift']:.2f}x"
        )
    print("=" * 72)

    # Sanity assertions
    assert total_companies == 24424, f"Expected 24,424 companies, got {total_companies}"
    assert matched_client_vats == 300, f"Expected 300 matched client VATs, got {matched_client_vats}"
    assert ec.filter(pl.col("primary_kad").is_null()).height == 0, "Every company must have a primary KAD"
    print("ALL SANITY ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    dataset = ingest_all()
    print_sanity_check_report(dataset)

