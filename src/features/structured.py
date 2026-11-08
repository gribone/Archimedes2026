"""
Structured Firmographic Feature Blocks (`src/features/structured.py`)
====================================================================
Deterministic L2-normalized feature blocks for KAD hierarchy (4-digit and
top-30 8-digit codes), legal entity forms, company age, branch networks,
and predictive firmographic signals.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.preprocessing import normalize


def build_structured_company_blocks(
    ec: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build deterministic L2-normalized blocks:
      1. `kad_hierarchy_features`
      2. `legal_features`
      3. `age_branch_features`
    """
    n = ec.height

    prim_kads = ec["primary_kad"].fill_null("56110000").to_list()
    kad4_list = [k[:4] for k in prim_kads]
    unique_kad4 = sorted(set(kad4_list))
    k4_map = {k: i for i, k in enumerate(unique_kad4)}

    top_k8_df = (
        ec.group_by("primary_kad")
        .len()
        .sort(["len", "primary_kad"], descending=[True, False])
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

    legal_types = ec["legal_type"].fill_null("OTHER").to_list()
    canonical_lts = ["ΑΤΟΜΙΚΗ", "ΙΚΕ", "ΕΕ", "ΟΕ", "ΑΕ", "ΕΠΕ"]
    lt_map = {lt: i for i, lt in enumerate(canonical_lts)}
    legal_mat = np.zeros((n, len(canonical_lts) + 1), dtype=np.float32)
    for i, lt in enumerate(legal_types):
        col = lt_map.get(lt, len(canonical_lts))
        legal_mat[i, col] = 1.0
    legal_mat = normalize(legal_mat, norm="l2", axis=1).astype(np.float32)

    age = ec["company_age_years"].fill_null(3.0).to_numpy().astype(np.float32)
    age_z = (age - np.mean(age)) / (np.std(age) or 1.0)

    is_branch = ec["is_branch"].fill_null(0).to_numpy().astype(np.float32)
    branch_cnt = np.log1p(ec["branch_count"].fill_null(0).to_numpy().astype(np.float32))
    branch_z = (branch_cnt - np.mean(branch_cnt)) / (np.std(branch_cnt) or 1.0)

    # Post-Launch Expansion Velocity (vwHackCompanyActivity.csv)
    exp_days = ec["post_launch_expansion_days"].fill_null(0.0).to_numpy().astype(np.float32)
    exp_days_z = (exp_days - np.mean(exp_days)) / (np.std(exp_days) or 1.0)
    has_exp = ec["has_post_launch_expansion"].fill_null(0).to_numpy().astype(np.float32)

    # Seasonality & Tourism Regime (incorporation_date)
    month_sin = ec["inc_month_sin"].fill_null(0.0).to_numpy().astype(np.float32)
    month_cos = ec["inc_month_cos"].fill_null(1.0).to_numpy().astype(np.float32)
    is_pre_summer = ec["is_pre_summer_launch"].fill_null(0).to_numpy().astype(np.float32)

    # Brand Sophistication & Concept Tier (company_name_el)
    has_dist = ec["has_distinctive_brand"].fill_null(1).to_numpy().astype(np.float32)
    is_mod = ec["is_modern_concept"].fill_null(0).to_numpy().astype(np.float32)
    brand_net = np.log1p(ec["shared_brand_network_size"].fill_null(1).to_numpy().astype(np.float32))
    brand_net_z = (brand_net - np.mean(brand_net)) / (np.std(brand_net) or 1.0)

    # Municipality Typology One-Hot & Coastal Flag (Extra Point X3)
    muni_typos = [
        "Αστικό Μητροπολιτικό Κέντρο",
        "Επαρχιακό / Αγροτικό",
        "Νησιωτικό Τουριστικό",
        "Περιφερειακό Εμπορικό Κέντρο",
        "Εποχικό Παράκτιο Θέρετρο",
    ]
    raw_muni_typo = ec["municipality_typology"].fill_null("Αστικό Μητροπολιτικό Κέντρο").to_list()
    muni_onehot = np.zeros((n, len(muni_typos)), dtype=np.float32)
    for row_i, mt in enumerate(raw_muni_typo):
        if mt in muni_typos:
            muni_onehot[row_i, muni_typos.index(mt)] = 1.0
        else:
            muni_onehot[row_i, 0] = 1.0

    is_coastal = ec["is_coastal_zone"].fill_null(0).to_numpy().astype(np.float32)

    firmographic = np.column_stack(
        [
            age_z,
            is_branch * 1.2,
            branch_z,
            exp_days_z * 0.8,
            has_exp * 1.1,
            month_sin * 0.7,
            month_cos * 0.7,
            is_pre_summer * 1.2,
            has_dist * 1.3,
            is_mod * 1.2,
            brand_net_z * 0.9,
            muni_onehot * 1.0,
            is_coastal * 1.1,
        ]
    )
    age_branch = normalize(firmographic, norm="l2", axis=1).astype(np.float32)

    return kad_hier, legal_mat, age_branch


__all__ = [
    "build_structured_company_blocks",
]
