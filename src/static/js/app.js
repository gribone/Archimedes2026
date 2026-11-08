/**
 * Archimedes 2026 — Master Application Entry Point (`src/static/js/app.js`)
 * Modular ES6 entry point with zero-dependency backward-compatible window bindings.
 */

import { state, subscribe } from './state.js';
import * as api from './api.js';
import * as mapMod from './map.js';
import * as chartMod from './charts.js';
import * as tableMod from './tables.js';
import * as drawerMod from './drawer.js';

// Bind to window for HTML inline event compatibility (onclick="focusOnMap(...)")
window.focusOnMap = mapMod.focusOnMap;
window.zoomToRegion = mapMod.zoomToRegion;
window.switchBasemap = mapMod.switchBasemap;
window.inspectCompany = drawerMod.inspectCompany;
window.closeCompanyDrawer = drawerMod.closeCompanyDrawer;
window.api = api;

export async function bootstrapApp() {
  try {
    mapMod.initMap();
    chartMod.initDonutCharts();
    chartMod.initSilhouetteChart();

    const data = await api.fetchBootstrap();
    if (data.initial) {
      applyPayload(data.initial);
    }
  } catch (err) {
    console.error('Bootstrap initialization error:', err);
  }
}

export function applyPayload(payload) {
  state.profiles = payload.profiles || [];
  state.lookalikes = payload.lookalikes || [];
  state.clients = payload.clients || [];
  state.meta = payload.meta || {};

  chartMod.renderDonutCharts();
  tableMod.renderLeadsTable();
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrapApp);
  } else {
    bootstrapApp();
  }
}

