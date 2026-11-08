"""
Greek Transit & Foot-Traffic Hubs Module (`src/features/transit.py`)
===================================================================
Loads coordinates of major Greek transit hubs, metro stations, ports,
airports, and university campuses used for foot-traffic catchment analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback hardcoded transit hubs (37 generators) if data asset is unavailable
FALLBACK_GREEK_TRANSIT_HUBS: list[tuple[float, float, str]] = [
    # Athens Metro Key Transit Hubs
    (37.9753, 23.7361, "Syntagma Metro / Transit Hub"),
    (37.9842, 23.7282, "Omonia Metro / Commercial Hub"),
    (37.9760, 23.7258, "Monastiraki Metro / Tourist Hub"),
    (37.9800, 23.7330, "Panepistimio Metro"),
    (37.9692, 23.7283, "Akropoli Metro / Historic Hub"),
    (37.9781, 23.7126, "Kerameikos / Gazi Nightlife"),
    (37.9868, 23.7654, "Ambelokipi Metro / Medical Hub"),
    (38.0075, 23.7997, "Halandri Commercial Center"),
    (38.0560, 23.8080, "Kifisia Center"),
    (37.9423, 23.6465, "Piraeus Port & Metro Terminal"),
    (38.0194, 24.0094, "Rafina Port Ferry Hub"),
    (37.7145, 24.0562, "Lavrio Port Ferry Hub"),
    # Thessaloniki Key Hubs & Metro
    (40.6350, 22.9340, "Thessaloniki Port & Ladadika Strip"),
    (40.6380, 22.9420, "Venizelou Metro / Center"),
    (40.6330, 22.9490, "Agia Sofia Metro / Commercial Axis"),
    (40.6300, 22.9600, "Sintrivani / AUTH Campus Hub"),
    (40.6430, 22.9190, "Thessaloniki New Railway Station"),
    # Major Regional Commercial Ports & Airports
    (35.3420, 25.1430, "Heraklion Passenger Port"),
    (35.3390, 25.1800, "Heraklion Airport HER"),
    (35.4870, 24.0720, "Souda / Chania Ferry Terminal"),
    (35.5310, 24.1500, "Chania Airport CHQ"),
    (38.2430, 21.7290, "Patras Port Passenger Terminal"),
    (36.4440, 28.2280, "Rhodes Commercial Port"),
    (37.4580, 25.3280, "Mykonos Tourlos Port"),
    (36.3860, 25.4310, "Santorini Athinios Port"),
    (39.6270, 19.9050, "Corfu Central Port"),
    (39.3560, 22.9460, "Volos Port Passenger Hub"),
    (37.9360, 23.9450, "AIA Athens International Airport"),
    (40.5200, 22.9710, "SKG Thessaloniki Airport"),
    # Major Greek University Campuses
    (37.9770, 23.7780, "NTUA / EKPA Zografou Campus"),
    (37.9940, 23.7320, "AUEB Patision Campus"),
    (37.9930, 23.6760, "UniWA Egaleo Campus"),
    (38.2880, 21.7880, "UPatras Rio Campus"),
    (35.3050, 25.0760, "UoC Voutes Heraklion Campus"),
    (35.5330, 24.0680, "TUC Kounoupidiana Chania Campus"),
    (39.6180, 20.8380, "UoI Dourouti Ioannina Campus"),
    (41.1440, 25.3720, "DUTH Komotini Campus"),
]


def load_greek_transit_hubs(
    path: str | Path | None = None,
) -> list[tuple[float, float, str]]:
    """
    Load transit hub coordinates from JSON file.

    Parameters
    ----------
    path : str | Path | None
        Path to greek_transit_hubs.json. If None, checks 'data/greek_transit_hubs.json'
        relative to current working directory and relative to repository root.

    Returns
    -------
    list[tuple[float, float, str]]
        List of (latitude, longitude, hub_name) tuples.
        Falls back to FALLBACK_GREEK_TRANSIT_HUBS if file is missing or invalid.
    """
    target_path: Path | None = None
    if path is not None:
        target_path = Path(path)
    else:
        p_cwd = Path("data/greek_transit_hubs.json")
        if p_cwd.exists():
            target_path = p_cwd
        else:
            p_repo = Path(__file__).resolve().parent.parent.parent / "data" / "greek_transit_hubs.json"
            if p_repo.exists():
                target_path = p_repo

    if target_path is None or not target_path.exists():
        logger.warning(
            "Greek transit hubs file not found at %s. Using embedded fallback (%d hubs).",
            target_path,
            len(FALLBACK_GREEK_TRANSIT_HUBS),
        )
        return list(FALLBACK_GREEK_TRANSIT_HUBS)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        hubs: list[tuple[float, float, str]] = []
        for item in raw_data:
            if isinstance(item, dict):
                lat = float(item.get("lat", item.get("latitude", 0.0)))
                lon = float(item.get("lon", item.get("longitude", item.get("lng", 0.0))))
                name = str(item.get("name", item.get("label", "")))
                hubs.append((lat, lon, name))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                hubs.append((float(item[0]), float(item[1]), str(item[2])))

        if not hubs:
            logger.warning("Empty transit hubs loaded from %s. Using fallback.", target_path)
            return list(FALLBACK_GREEK_TRANSIT_HUBS)

        return hubs
    except Exception as exc:
        logger.warning(
            "Failed to load transit hubs from %s (%s). Using embedded fallback.",
            target_path,
            exc,
        )
        return list(FALLBACK_GREEK_TRANSIT_HUBS)


# Module-level constant populated via loader
GREEK_TRANSIT_FOOT_TRAFFIC_HUBS: list[tuple[float, float, str]] = load_greek_transit_hubs()

__all__ = [
    "FALLBACK_GREEK_TRANSIT_HUBS",
    "GREEK_TRANSIT_FOOT_TRAFFIC_HUBS",
    "load_greek_transit_hubs",
]
