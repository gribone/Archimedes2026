/**
 * Search & Quick Filter Chips (`src/static/js/components/searchFilter.js`)
 * Filters lookalike prospects by quick chips, search text, ICP, prefecture, and legal form.
 */

export function applyQuickChipFilter(rows, chipKey) {
  if (!chipKey || chipKey === "all") return rows;
  switch (chipKey) {
    case "hot":
      return rows.filter((r) => (r.score || 0) >= 0.85);
    case "route":
      return rows.filter(
        (r) => r.nearest_client_km != null && Number(r.nearest_client_km) <= 1.5
      );
    case "multi":
      return rows.filter((r) => (r.branches || 0) >= 1 || r.is_branch === 1);
    case "tier1":
      return rows.filter((r) => String(r.osm_tier || "").startsWith("Tier 1"));
    case "transit":
      return rows.filter((r) => Number(r.is_transit_hub) === 1);
    default:
      return rows;
  }
}

