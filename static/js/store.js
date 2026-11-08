/**
 * Reactive State Store (`src/static/js/store.js`)
 * Lightweight Proxy store with strict equality guard (`prevVal === value`)
 * to prevent infinite re-render cascades across table, map, and drawer.
 */

export function createStore(initialState) {
  const listeners = new Map();

  const state = new Proxy(initialState, {
    set(target, prop, value) {
      const prevVal = target[prop];
      if (prevVal === value) return true;
      target[prop] = value;

      const subs = listeners.get(prop);
      if (subs) {
        subs.forEach((fn) => fn(value, prevVal, target));
      }
      return true;
    },
  });

  return {
    state,
    subscribe(keys, callback) {
      const keyList = Array.isArray(keys) ? keys : [keys];
      keyList.forEach((k) => {
        if (!listeners.has(k)) listeners.set(k, new Set());
        listeners.get(k).add(callback);
      });
      return () => keyList.forEach((k) => listeners.get(k)?.delete(callback));
    },
  };
}

export const appStore = createStore({
  lang: localStorage.getItem("archimedes_lang") || "el",
  activeTab: "cockpit",
  layoutMode: "split",       // 'split' | 'table' | 'map'
  tableDensity: "standard",  // 'compact' | 'standard' | 'detailed'
  execSummaryExpanded: false,
  hoveredCompanyId: null,
  activeCompanyId: null,
  selectedLeadIds: new Set(),
  deliveryRingsVisible: false,
  activeRouteWaypoints: [],
  sortCol: "rank",
  sortDir: "asc",
  quickChip: "all",
  page: 1,
  pageSize: 25,
});

export const actions = {
  setLanguage(lang) {
    appStore.state.lang = lang === "en" ? "en" : "el";
  },
  setLayoutMode(mode) {
    if (["split", "table", "map"].includes(mode)) {
      appStore.state.layoutMode = mode;
    }
  },
  setTableDensity(density) {
    if (["compact", "standard", "detailed"].includes(density)) {
      appStore.state.tableDensity = density;
    }
  },
  hoverCompany(companyId) {
    appStore.state.hoveredCompanyId = companyId;
  },
  selectCompany(companyId) {
    appStore.state.activeCompanyId = companyId;
  },
  toggleLeadSelection(companyId) {
    const next = new Set(appStore.state.selectedLeadIds);
    if (next.has(companyId)) next.delete(companyId);
    else next.add(companyId);
    appStore.state.selectedLeadIds = next;
  },
  clearLeadSelection() {
    appStore.state.selectedLeadIds = new Set();
  },
};

