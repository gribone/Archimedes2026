"""
Feature Engineering Transformations (`src/features/engineering.py`)
====================================================================
Predictive feature extraction functions relocated from ingestion pipeline:
- Post-launch expansion velocity
- Seasonality and tourism regime indicators
- Brand sophistication and concept tier metrics
- Municipality demographics and regional typologies
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl


def compute_post_launch_expansion(
    companies: pl.DataFrame,
    company_activities: pl.DataFrame,
) -> pl.DataFrame:
    """
    Post-Launch Expansion Velocity (Feature 2.1):
    Calculates days between incorporation_date and secondary date_from.
    Flags businesses that added secondary KADs >= 180 days post-incorporation
    (real commercial pivot or expansion vs Day-1 accountant noise).
    """
    sec_ca = (
        company_activities.filter(pl.col("activity_type") == "Δευτερεύουσα")
        .join(companies.select(["company_id", "incorporation_date"]), on="company_id", how="inner")
        .filter(pl.col("date_from").is_not_null() & pl.col("incorporation_date").is_not_null())
        .with_columns(
            (pl.col("date_from") - pl.col("incorporation_date")).dt.total_days().alias("diff_days")
        )
    )
    res = (
        sec_ca.group_by("company_id")
        .agg(
            pl.col("diff_days").max().alias("post_launch_expansion_days"),
            (pl.col("diff_days").max() >= 180).alias("has_post_launch_expansion"),
        )
    )
    return res


def compute_seasonality_features(companies: pl.DataFrame) -> pl.DataFrame:
    """
    Seasonality & Tourism Regime (Feature 2.2):
    Extract cyclical month (sin/cos) and an `is_pre_summer_launch` indicator (Months 3–6)
    to separate seasonal tourist venues from year-round urban stores.
    """
    month = companies["incorporation_date"].dt.month().fill_null(1)
    m_arr = month.to_numpy().astype(np.float64)
    sin_arr = np.sin(2.0 * np.pi * m_arr / 12.0)
    cos_arr = np.cos(2.0 * np.pi * m_arr / 12.0)
    is_pre_summer = (month >= 3) & (month <= 6)

    return pl.DataFrame({
        "company_id": companies["company_id"],
        "incorporation_month": month,
        "inc_month_sin": sin_arr,
        "inc_month_cos": cos_arr,
        "is_pre_summer_launch": is_pre_summer,
    })


def compute_brand_features(companies: pl.DataFrame) -> pl.DataFrame:
    """
    Brand Sophistication & Concept Tier (Feature 2.3):
    - `has_distinctive_brand`: 0 if generic default «Επιχείρηση Εστίασης», 1 otherwise.
    - `is_modern_concept`: Regex match for modern hospitality keywords.
    - `shared_brand_network_size`: Count how many distinct VATs share the same normalized brand name.
    """
    names = companies["company_name_el"].fill_null("").str.strip_chars()
    is_generic = names.str.starts_with("Επιχείρηση Εστίασης") | (names == "")
    has_distinctive = ~is_generic

    modern_pattern = (
        r"(?i)(coffee|roasters|bistro|burger|street|brunch|bar\b|cocktail|project|all\s*day|gastro|"
        r"κοκτέιλ|μπιστρό|μπέργκερ|μπραντς|ρόαστερς|στριτ|καφέ|μπαρ)"
    )
    is_modern = names.str.contains(modern_pattern) & has_distinctive

    def _clean_brand_name(n: str) -> str:
        nu = n.upper()
        if not nu or nu.startswith("ΕΠΙΧΕΙΡΗΣΗ ΕΣΤΙΑΣΗΣ"):
            return ""
        nu = re.sub(
            r"\b(ΜΟΝΟΠΡΟΣΩΠΗ|ΙΔΙΩΤΙΚΗ|ΚΕΦΑΛΑΙΟΥΧΙΚΗ|ΕΤΑΙΡΕΙΑ|ΕΤΑΙΡΙΑ|ΙΚΕ|Ι\.Κ\.Ε\.|ΟΕ|Ο\.Ε\.|ΕΕ|Ε\.Ε\.|ΑΕ|Α\.Ε\.|ΕΠΕ|Ε\.Π\.Ε\.|ΥΠΟΚΑΤΑΣΤΗΜΑ)\b",
            "",
            nu,
        )
        nu = re.sub(r"[\(\)\.\,\-\/]", " ", nu)
        return " ".join(nu.split())

    cleaned_brands = [_clean_brand_name(x) for x in names.to_list()]
    df_temp = pl.DataFrame({
        "company_id": companies["company_id"],
        "vat": companies["vat"],
        "_brand_key": cleaned_brands,
    })

    valid_b = df_temp.filter(pl.col("_brand_key") != "")
    brand_vat_counts = valid_b.group_by("_brand_key").agg(
        pl.col("vat").n_unique().alias("shared_brand_network_size")
    )
    df_temp = df_temp.join(brand_vat_counts, on="_brand_key", how="left").with_columns(
        pl.col("shared_brand_network_size").fill_null(1).cast(pl.Int64)
    )

    return pl.DataFrame({
        "company_id": companies["company_id"],
        "has_distinctive_brand": has_distinctive,
        "is_modern_concept": is_modern,
        "shared_brand_network_size": df_temp["shared_brand_network_size"],
    })


def compute_municipality_typology(companies: pl.DataFrame) -> pl.DataFrame:
    """
    Municipality Demographics & Regional Typology (Feature X3):
    Classifies 330 Greek municipalities into:
      - Urban Metropolitan (Athens, Thessaloniki, Piraeus, suburban hubs)
      - Island Tourist (Cyclades, Dodecanese, Ionian, Crete islands)
      - Seasonal Coastal Resort (Halkidiki, Pieria coastal, etc.)
      - Regional Commercial City (Patras, Larissa, Heraklion, Volos, Ioannina)
      - Provincial / Rural
    """
    island_kw = [
        "ΜΥΚΟΝ", "ΘΗΡΑ", "ΣΑΝΤΟΡΙΝ", "ΠΑΡΟΥ", "ΝΑΞΟΥ", "ΜΗΛΟΥ", "ΡΟΔΟΥ", "ΚΩ", "ΚΑΡΠΑΘ",
        "ΚΕΡΚΥΡ", "ΖΑΚΥΝΘ", "ΚΕΦΑΛ", "ΛΕΥΚΑΔ", "ΣΚΙΑΘ", "ΣΚΟΠΕΛ", "ΣΑΜΟΥ", "ΧΙΟΥ", "ΛΕΣΒΟΥ", "ΙΚΑΡ", "ΛΗΜΝΟΥ", "ΤΗΝΟΥ"
    ]
    resort_kw = ["ΚΑΣΣΑΝΔΡ", "ΣΙΘΩΝ", "ΧΕΡΣΟΝΗΣ", "ΑΓΙΟΥ ΝΙΚΟΛΑ", "ΜΑΛΕΒΙΖ", "ΠΛΑΤΑΝΙΑ", "ΙΕΡΑΠΕΤΡ"]
    metro_kw = [
        "ΑΘΗΝΑΙ", "ΘΕΣΣΑΛΟΝΙΚ", "ΠΕΙΡΑΙ", "ΚΑΛΛΙΘΕ", "ΓΛΥΦΑΔ", "ΜΑΡΟΥΣ", "ΚΗΦΙΣΙ", "ΧΑΛΑΝΔΡ",
        "ΠΕΡΙΣΤΕΡ", "ΝΕΑΣ ΣΜΥΡΝ", "ΗΛΙΟΥΠΟΛ", "ΖΩΓΡΑΦ", "ΑΓΙΑΣ ΠΑΡΑΣΚΕΥ", "ΒΥΡΩΝ", "ΠΑΛΑΙΟΥ ΦΑΛΗΡ"
    ]
    city_kw = [
        "ΠΑΤΡΕ", "ΛΑΡΙΣΑΙ", "ΗΡΑΚΛΕΙ", "ΙΩΑΝΝΙΤ", "ΒΟΛΟΥ", "ΤΡΙΚΚΑΙ", "ΧΑΛΚΙΔΕ", "ΣΕΡΡΩΝ",
        "ΚΑΒΑΛ", "ΚΑΛΑΜΑΤ", "ΚΟΖΑΝ", "ΑΛΕΞΑΝΔΡΟΥΠΟΛ"
    ]

    types = []
    is_coastal = []
    for m_name, p_name in zip(companies["municipality_name"].to_list(), companies["prefecture_name"].to_list()):
        nu = (m_name or "").upper()
        pu = (p_name or "").upper()

        if any(k in nu for k in island_kw) or "ΚΥΚΛΑΔ" in pu or "ΔΩΔΕΚΑΝ" in pu or "ΙΟΝΙ" in pu:
            t = "Island Tourist"
            c = True
        elif any(k in nu for k in resort_kw) or "ΧΑΛΚΙΔΙΚ" in pu:
            t = "Seasonal Coastal Resort"
            c = True
        elif any(k in nu for k in metro_kw) or "ΚΕΝΤΡΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ" in nu or "ΒΟΡΕΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ" in nu or "ΝΟΤΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ" in nu:
            t = "Urban Metropolitan"
            c = "ΝΟΤΙΟΥ" in nu or "ΠΕΙΡΑΙ" in nu
        elif any(k in nu for k in city_kw):
            t = "Regional Commercial City"
            c = any(ck in nu for ck in ["ΠΑΤΡΕ", "ΗΡΑΚΛΕΙ", "ΒΟΛΟΥ", "ΧΑΛΚΙΔΕ", "ΚΑΒΑΛ", "ΚΑΛΑΜΑΤ", "ΑΛΕΞΑΝΔΡΟΥΠΟΛ"])
        else:
            t = "Provincial / Rural"
            c = any(ck in nu for ck in ["ΠΑΡΑΛ", "ΑΙΓΙΑΛ", "ΝΑΥΠΛΙ", "ΠΡΕΒΕΖ", "ΙΤΕΑ", "ΓΥΘΕΙ"])

        types.append(t)
        is_coastal.append(c)

    return pl.DataFrame({
        "company_id": companies["company_id"],
        "municipality_typology": types,
        "is_coastal_zone": is_coastal,
    })


def compute_geo_regime(companies: pl.DataFrame) -> pl.DataFrame:
    """
    Classify prefecture_name and municipality_name into 4 distinct commercial regimes:
      - METRO_ATHENS_THESS: Urban metropolitan hubs of Attica and Thessaloniki
      - ISLAND_TOURISM: Cyclades, Dodecanese, Ionian, and North Aegean islands
      - COASTAL_RESORT: Chalkidiki and high-density summer coastal resort zones
      - MAINLAND_REGIONAL: Regional capital cities and mainland commercial corridors
    """
    island_kw = [
        "ΜΥΚΟΝ", "ΘΗΡΑ", "ΣΑΝΤΟΡΙΝ", "ΠΑΡΟΥ", "ΝΑΞΟΥ", "ΜΗΛΟΥ", "ΡΟΔΟΥ", "ΚΩ", "ΚΑΡΠΑΘ",
        "ΚΕΡΚΥΡ", "ΖΑΚΥΝΘ", "ΚΕΦΑΛ", "ΛΕΥΚΑΔ", "ΣΚΙΑΘ", "ΣΚΟΠΕΛ", "ΣΑΜΟΥ", "ΧΙΟΥ", "ΛΕΣΒΟΥ", "ΙΚΑΡ", "ΛΗΜΝΟΥ", "ΤΗΝΟΥ"
    ]
    resort_kw = ["ΚΑΣΣΑΝΔΡ", "ΣΙΘΩΝ", "ΧΕΡΣΟΝΗΣ", "ΑΓΙΟΥ ΝΙΚΟΛΑ", "ΜΑΛΕΒΙΖ", "ΠΛΑΤΑΝΙΑ", "ΙΕΡΑΠΕΤΡ"]

    regimes = []
    for m_name, p_name in zip(companies["municipality_name"].to_list(), companies["prefecture_name"].to_list()):
        nu = (m_name or "").upper()
        pu = (p_name or "").upper()
        if "ΑΤΤΙΚ" in pu or "ΘΕΣΣΑΛΟΝΙΚ" in pu:
            r = "METRO_ATHENS_THESS"
        elif any(k in nu for k in island_kw) or any(k in pu for k in ["ΚΥΚΛΑΔ", "ΔΩΔΕΚΑΝ", "ΚΕΡΚΥΡ", "ΖΑΚΥΝΘ", "ΛΕΥΚΑΔ", "ΚΕΦΑΛ", "ΛΕΣΒ", "ΧΙΟ", "ΣΑΜ"]):
            r = "ISLAND_TOURISM"
        elif any(k in nu for k in resort_kw) or "ΧΑΛΚΙΔΙΚ" in pu:
            r = "COASTAL_RESORT"
        else:
            r = "MAINLAND_REGIONAL"
        regimes.append(r)

    return pl.DataFrame({
        "company_id": companies["company_id"],
        "geo_regime": regimes,
    })


# Backward compatibility aliases
_compute_post_launch_expansion = compute_post_launch_expansion
_compute_seasonality_features = compute_seasonality_features
_compute_brand_features = compute_brand_features
_compute_municipality_typology = compute_municipality_typology
_compute_geo_regime = compute_geo_regime

__all__ = [
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
]
