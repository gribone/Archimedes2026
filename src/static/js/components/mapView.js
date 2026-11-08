/**
 * Map View & Single DOM Overlay Halo Marker (`src/static/js/components/mapView.js`)
 * Uses a single L.divIcon overlay marker on top of L.canvas() for 60 FPS hover priming,
 * plus 1km Delivery Route Rings and Nearest-Neighbor Route Polyline builder.
 */

export function createHoverHaloOverlay(L) {
  if (!L) return null;
  const icon = L.divIcon({
    className: "map-pulse-halo",
    html: '<div class="pulse-ring"></div><div class="pulse-core"></div>',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
  return L.marker([0, 0], {
    icon,
    interactive: false,
    zIndexOffset: 1000,
  });
}

export function computeHaversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371.0088;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

export function buildNearestNeighborRoute(leads) {
  const valid = (leads || []).filter(
    (r) => r.lat != null && r.lon != null && !Number.isNaN(Number(r.lat)) && !Number.isNaN(Number(r.lon))
  );
  if (valid.length < 2) return { ordered: valid, totalKm: 0 };

  const remaining = [...valid];
  const ordered = [remaining.shift()];
  let totalKm = 0;

  while (remaining.length > 0) {
    const last = ordered[ordered.length - 1];
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const d = computeHaversineKm(
        Number(last.lat),
        Number(last.lon),
        Number(remaining[i].lat),
        Number(remaining[i].lon)
      );
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    totalKm += bestDist;
    ordered.push(remaining.splice(bestIdx, 1)[0]);
  }

  return { ordered, totalKm: Math.round(totalKm * 100) / 100 };
}

