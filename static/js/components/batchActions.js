/**
 * Floating Batch Action Bar (`src/static/js/components/batchActions.js`)
 * Manages multi-row selection export, batch GEMI check, route building, and exclusion.
 */

import { appStore } from "../store.js";
import { t } from "../utils/i18n.js";

export function updateBatchActionBar(activeRouteKm = 0) {
  const bar = document.getElementById("batch-action-bar");
  if (!bar) return;
  const count = appStore.state.selectedLeadIds.size;
  bar.classList.toggle("visible", count > 0 || activeRouteKm > 0);

  const countEl = document.getElementById("fab-selected-count");
  if (countEl) {
    countEl.textContent = t("fab.selectedCount", { count });
  }

  const routeBtn = document.getElementById("fab-route-btn");
  if (routeBtn) {
    if (activeRouteKm > 0) {
      routeBtn.textContent = t("fab.clearRoute", { km: activeRouteKm.toFixed(1) });
      routeBtn.classList.add("route-active");
    } else {
      routeBtn.textContent = t("fab.buildRoute");
      routeBtn.classList.remove("route-active");
    }
  }
}

