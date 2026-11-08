/**
 * Pinned Leads Table & Density Controller (`src/static/js/components/leadsTable.js`)
 * Supports 3-way row density ('compact' | 'standard' | 'detailed'), 2-way column sorting,
 * sticky left/right columns, and pagination-aware focus scrolling.
 */

import { appStore } from "../store.js";

export function setLeadsTableDensity(density) {
  if (!["compact", "standard", "detailed"].includes(density)) return;
  appStore.state.tableDensity = density;

  const table = document.getElementById("lookalikes-table");
  if (table) {
    table.classList.remove("density-compact", "density-standard", "density-detailed");
    table.classList.add(`density-${density}`);
  }

  document.querySelectorAll("[data-density-btn]").forEach((btn) => {
    const isMatch = btn.getAttribute("data-density-btn") === density;
    btn.classList.toggle("active", isMatch);
    btn.setAttribute("aria-pressed", String(isMatch));
  });
}

export function cycleLeadsTableDensity() {
  const order = ["compact", "standard", "detailed"];
  const cur = appStore.state.tableDensity || "standard";
  const next = order[(order.indexOf(cur) + 1) % order.length];
  setLeadsTableDensity(next);
}

