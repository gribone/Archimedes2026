/**
 * 55/45 Synchronized Split-Screen Controller (`src/static/js/components/splitLayout.js`)
 * Controls Layout Modes: 'split' (55% table / 45% map), 'table' (100% table), 'map' (100% map).
 */

import { appStore } from "../store.js";

export function setSplitLayoutMode(mode, leafletMap = null) {
  if (!["split", "table", "map"].includes(mode)) return;
  appStore.state.layoutMode = mode;

  const container = document.getElementById("cockpit-split-container");
  if (container) {
    container.setAttribute("data-layout", mode);
  }

  document.querySelectorAll("[data-layout-btn]").forEach((btn) => {
    const isMatch = btn.getAttribute("data-layout-btn") === mode;
    btn.classList.toggle("active", isMatch);
    btn.setAttribute("aria-pressed", String(isMatch));
  });

  if (leafletMap && (mode === "split" || mode === "map")) {
    setTimeout(() => {
      leafletMap.invalidateSize({ animate: false });
    }, 60);
  }
}

