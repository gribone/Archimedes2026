/**
 * Archimedes 2026 — Map Module (`src/static/js/map.js`)
 * Leaflet map rendering, layer management, halo animation, and regional presets.
 */

import { state, ICP_COLORS } from './state.js';

export const REGIONS = {
  greece: { center: [38.2, 23.7], zoom: 7, label: 'Όλη η Ελλάδα' },
  attica: { center: [37.98, 23.73], zoom: 11, label: 'Αττική' },
  thessaloniki: { center: [40.63, 22.94], zoom: 11, label: 'Θεσσαλονίκη' },
  halkidiki: { center: [40.35, 23.50], zoom: 10, label: 'Χαλκιδική' },
  crete: { center: [35.33, 25.14], zoom: 9, label: 'Κρήτη' },
  cyclades: { center: [37.08, 25.15], zoom: 9, label: 'Κυκλάδες' },
  dodecanese: { center: [36.43, 28.22], zoom: 9, label: 'Δωδεκάνησα' },
  ionian: { center: [38.75, 20.65], zoom: 9, label: 'Ιόνια Νησιά' },
};

export function initMap() {
  if (state.map) return;
  const mapEl = document.getElementById('map');
  if (!mapEl) return;

  state.canvasRenderer = L.canvas({ padding: 0.5 });
  state.map = L.map('map', {
    center: REGIONS.greece.center,
    zoom: REGIONS.greece.zoom,
    zoomControl: false,
    preferCanvas: true,
  });

  L.control.zoom({ position: 'topright' }).addTo(state.map);
  switchBasemap('esri');

  state.markersLayer = L.layerGroup().addTo(state.map);
}

export function switchBasemap(style) {
  state.mapFilter.basemap = style;
  if (state.currentTileLayer && state.map) {
    state.map.removeLayer(state.currentTileLayer);
  }

  const url = style === 'esri'
    ? 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

  const attribution = style === 'esri' ? '&copy; Esri &mdash; Esri, DeLorme, NAVTEQ' : '&copy; OpenStreetMap & CARTO';
  state.currentTileLayer = L.tileLayer(url, { maxZoom: 16, attribution }).addTo(state.map);
}

export function zoomToRegion(regionKey) {
  const reg = REGIONS[regionKey];
  if (!reg || !state.map) return;
  state.map.flyTo(reg.center, reg.zoom, { duration: 1.2 });
}

export function focusOnMap(lat, lon, companyId, companyName, profileId) {
  if (!state.map || lat == null || lon == null || isNaN(lat) || isNaN(lon)) return;
  state.map.setView([lat, lon], 15, { animate: true });
  showMapHalo([lat, lon]);
}

export function showMapHalo(latLon) {
  if (!state.map) return;
  if (state.haloMarker) {
    state.map.removeLayer(state.haloMarker);
    state.haloMarker = null;
  }
  const haloIcon = L.divIcon({
    className: 'map-focus-halo',
    iconSize: [60, 60],
    iconAnchor: [30, 30],
  });
  state.haloMarker = L.marker(latLon, { icon: haloIcon, interactive: false }).addTo(state.map);
  setTimeout(() => {
    if (state.haloMarker) {
      state.map.removeLayer(state.haloMarker);
      state.haloMarker = null;
    }
  }, 4500);
}

