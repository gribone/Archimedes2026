"""
Spatial & Geospatial Feature Engineering (`src/features/spatial.py`)
===================================================================
3D spherical cKDTree indexing, density estimation, competitor clustering,
and HoReCa delivery route gravity calculations.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize

from src.config import PipelineConfig

EARTH_RADIUS_KM: float = 6371.0088


def build_spatial_ckdtree_features(
    ec: pl.DataFrame,
    cfg: PipelineConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geographic features using 3D unit-sphere coordinates and `scipy.spatial.cKDTree`
    (Requirement A1 & Extra Points X1).
    """
    if cfg is None:
        cfg = PipelineConfig()

    n = ec.height
    lat_deg = ec["latitude"].to_numpy()
    lon_deg = ec["longitude"].to_numpy()
    has_gps = ~np.isnan(lat_deg) & ~np.isnan(lon_deg)

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    latlon_rad = np.column_stack([lat_rad, lon_rad])

    valid_idx = np.where(has_gps)[0]
    cos_lat = np.cos(lat_rad[valid_idx])
    xyz_valid = np.column_stack(
        [
            cos_lat * np.cos(lon_rad[valid_idx]),
            cos_lat * np.sin(lon_rad[valid_idx]),
            np.sin(lat_rad[valid_idx]),
        ]
    )

    tree = cKDTree(xyz_valid)

    density_cols = []
    for radius_km in cfg.spatial_radii_km:
        chord = 2.0 * np.sin(radius_km / (2.0 * EARTH_RADIUS_KM))
        counts_valid = tree.query_ball_point(xyz_valid, r=chord, return_length=True)
        log_dens_valid = np.log1p(np.maximum(counts_valid - 1, 0)).astype(np.float32)

        full_col = np.full(n, float(np.median(log_dens_valid)), dtype=np.float32)
        full_col[valid_idx] = log_dens_valid
        std = float(np.std(full_col)) or 1.0
        full_col = (full_col - float(np.mean(full_col))) / std
        density_cols.append(full_col)

    # HoReCa Micro-Cluster Density & Gravity (Extra Point X1 - Geographical ICP)
    # (a) Competitor density within 250m & 1km
    chord_250m = 2.0 * np.sin(0.25 / (2.0 * EARTH_RADIUS_KM))
    chord_1km = 2.0 * np.sin(1.0 / (2.0 * EARTH_RADIUS_KM))
    dens_250m_valid = np.log1p(
        np.maximum(tree.query_ball_point(xyz_valid, r=chord_250m, return_length=True) - 1, 0)
    ).astype(np.float32)
    dens_1km_valid = np.log1p(
        np.maximum(tree.query_ball_point(xyz_valid, r=chord_1km, return_length=True) - 1, 0)
    ).astype(np.float32)

    col_250m = np.full(n, float(np.median(dens_250m_valid)), dtype=np.float32)
    col_250m[valid_idx] = dens_250m_valid
    col_250m = (col_250m - float(np.mean(col_250m))) / (float(np.std(col_250m)) or 1.0)

    col_1km = np.full(n, float(np.median(dens_1km_valid)), dtype=np.float32)
    col_1km[valid_idx] = dens_1km_valid
    col_1km = (col_1km - float(np.mean(col_1km))) / (float(np.std(col_1km)) or 1.0)

    # (b) Local ratio of Bars/Cafes (5630) to Restaurants (5610) within 1km & 500m share
    pkads = ec["primary_kad"].fill_null("").to_numpy()[valid_idx]
    is_bar = np.array([k.startswith("5630") for k in pkads], dtype=np.int32)
    is_rest = np.array([k.startswith("5610") for k in pkads], dtype=np.int32)
    neighbors_1km = tree.query_ball_point(xyz_valid, r=chord_1km)
    bar_counts = np.array([is_bar[indices].sum() for indices in neighbors_1km], dtype=np.float32)
    rest_counts = np.array([is_rest[indices].sum() for indices in neighbors_1km], dtype=np.float32)
    bar_rest_valid = np.log((bar_counts + 1.0) / (rest_counts + 1.0)).astype(np.float32)

    col_bar_rest = np.full(n, float(np.median(bar_rest_valid)), dtype=np.float32)
    col_bar_rest[valid_idx] = bar_rest_valid
    col_bar_rest = (col_bar_rest - float(np.mean(col_bar_rest))) / (float(np.std(col_bar_rest)) or 1.0)

    # 500m local bar/cafe share (5630 vs 5610)
    chord_500m = 2.0 * np.sin(0.5 / (2.0 * EARTH_RADIUS_KM))
    neighbors_500m = tree.query_ball_point(xyz_valid, r=chord_500m)
    bar_counts_500 = np.array([is_bar[indices].sum() for indices in neighbors_500m], dtype=np.float32)
    rest_counts_500 = np.array([is_rest[indices].sum() for indices in neighbors_500m], dtype=np.float32)
    tot_500 = bar_counts_500 + rest_counts_500
    bar_share_500_valid = np.where(tot_500 > 0, bar_counts_500 / np.maximum(tot_500, 1.0), 0.5).astype(np.float32)

    col_bar_share = np.full(n, float(np.median(bar_share_500_valid)), dtype=np.float32)
    col_bar_share[valid_idx] = bar_share_500_valid
    col_bar_share = (col_bar_share - float(np.mean(col_bar_share))) / (float(np.std(col_bar_share)) or 1.0)

    # (c) Revenue-weighted proximity to existing clients (B2B Delivery Route Gravity)
    is_client = ec["is_client_representative"].to_numpy()
    client_valid_mask = is_client[valid_idx]
    client_indices_in_valid = np.where(client_valid_mask)[0]
    if len(client_indices_in_valid) > 0:
        client_xyz = xyz_valid[client_indices_in_valid]
        client_rev = ec["turnover"].to_numpy()[valid_idx][client_indices_in_valid]
        client_weights = np.log1p(np.maximum(client_rev, 1000.0))
        client_weights /= (client_weights.sum() or 1.0)
        client_dists_chord = np.linalg.norm(xyz_valid[:, None, :] - client_xyz[None, :, :], axis=2)
        client_dists_km = client_dists_chord * EARTH_RADIUS_KM
        gravity_valid = np.sum(client_weights[None, :] * np.exp(-0.5 * (client_dists_km / 15.0) ** 2), axis=1).astype(np.float32)
    else:
        gravity_valid = np.zeros(len(valid_idx), dtype=np.float32)

    col_gravity = np.full(n, float(np.median(gravity_valid)), dtype=np.float32)
    col_gravity[valid_idx] = gravity_valid
    col_gravity = (col_gravity - float(np.mean(col_gravity))) / (float(np.std(col_gravity)) or 1.0)

    med_lat = float(np.nanmedian(lat_deg))
    med_lon = float(np.nanmedian(lon_deg))
    lat_filled = np.where(has_gps, lat_deg, med_lat)
    lon_filled = np.where(has_gps, lon_deg, med_lon)
    lat_z = ((lat_filled - np.mean(lat_filled)) / (np.std(lat_filled) or 1.0)).astype(np.float32)
    lon_z = ((lon_filled - np.mean(lon_filled)) / (np.std(lon_filled) or 1.0)).astype(np.float32)

    # (d) Geographical Regime (Metro / Island / Resort / Mainland)
    regimes = (
        ec["geo_regime"].fill_null("MAINLAND_REGIONAL").to_list()
        if "geo_regime" in ec.columns
        else ["MAINLAND_REGIONAL"] * n
    )
    regime_order = ["METRO_ATHENS_THESS", "ISLAND_TOURISM", "COASTAL_RESORT", "MAINLAND_REGIONAL"]
    regime_to_col = {r: i for i, r in enumerate(regime_order)}
    regime_onehot = np.zeros((n, 4), dtype=np.float32)
    for row_i, r in enumerate(regimes):
        if r in regime_to_col:
            regime_onehot[row_i, regime_to_col[r]] = 1.0

    prefs = ec["prefecture_name"].fill_null("UNKNOWN").to_list()
    pref_counts = ec.group_by("prefecture_name").len().filter(pl.col("len") >= 150)
    top_prefs = sorted(
        [p for p in pref_counts["prefecture_name"].to_list() if p is not None]
    )
    pref_to_col = {p: i for i, p in enumerate(top_prefs)}
    pref_onehot = np.zeros((n, len(top_prefs)), dtype=np.float32)
    for row_i, p in enumerate(prefs):
        if p in pref_to_col:
            pref_onehot[row_i, pref_to_col[p]] = 1.0

    raw_spatial = np.column_stack(
        [
            lat_z * 0.7,
            lon_z * 0.7,
            *density_cols,
            col_250m * 0.85,
            col_1km * 0.85,
            col_bar_rest * 0.65,
            col_bar_share * 0.70,
            col_gravity * 0.90,
            regime_onehot * 1.15,
            pref_onehot * 1.2,
        ]
    )
    spatial_norm = normalize(raw_spatial, norm="l2", axis=1).astype(np.float32)
    return spatial_norm, latlon_rad, has_gps


__all__ = [
    "EARTH_RADIUS_KM",
    "build_spatial_ckdtree_features",
]
