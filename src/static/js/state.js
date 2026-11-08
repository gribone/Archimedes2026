/**
 * Archimedes 2026 — State Management (`src/static/js/state.js`)
 * Centralized, observable application state for executive cockpit.
 */

export const ICP_COLORS = ['#2563eb', '#0d9488', '#d97706', '#e11d48', '#7c3aed', '#0284c7'];

export const state = {
  params: {},
  defaults: {},
  paramSpecs: [],
  meta: {},
  profiles: [],
  lookalikes: [],
  clients: [],
  benchmark: {},
  ollama: { available: false, model: '', installed: [] },
  selectedCompanyId: null,
  activeTab: 'overview',
  
  // Leads Table Filter & Pagination
  leadsFilter: {
    search: '',
    profile: 'ALL',
    prefecture: 'ALL',
    legal: 'ALL',
    quickFilter: 'all', // 'all' | 'top100' | 'attica' | 'thessaloniki' | 'islands' | 'corporate' | 'hq'
    page: 1,
    pageSize: 50,
  },
  
  // Clients Table Filter
  clientsFilter: {
    search: '',
    profile: 'ALL',
    quickFilter: 'all', // 'all' | 'vip' | 'icp1' | 'icp2' | 'nofit' | 'branches'
  },

  // Map Filter & Basemap
  mapFilter: {
    layer: 'ALL', // 'ALL' | 'CLIENTS' | 'LEADS'
    region: 'greece',
    basemap: 'esri', // 'esri' | 'osm'
  },
  
  // Active map & chart instances
  map: null,
  markersLayer: null,
  currentTileLayer: null,
  canvasRenderer: null,
  haloMarker: null,
  miniMap: null,
  chartRevDonut: null,
  chartClientsDonut: null,
  chartSilhouette: null,
  compCharts: {},
  
  // Debounce timer for instant slider updates
  sliderDebounceTimer: null,
  isComputing: false,
};

const subscribers = new Set();

export function subscribe(callback) {
  subscribers.add(callback);
  return () => subscribers.delete(callback);
}

export function notify(event, data) {
  subscribers.forEach(cb => {
    try { cb(event, data); } catch (e) { console.error('State subscriber error:', e); }
  });
}

// Bind to window for debugging and inline event compatibility
if (typeof window !== 'undefined') {
  window.__ARCHIMEDES_STATE__ = state;
}

