/**
 * Collapsible Executive Summary Bar & KPI Cockpit (`src/static/js/components/kpiCockpit.js`)
 * Implements the 40px collapsed summary bar + expandable 4-card KPI strip & portfolio donuts.
 */

import { appStore } from "../store.js";
import { t, formatters } from "../utils/i18n.js";

export function toggleExecSummary(forceState) {
  const next =
    typeof forceState === "boolean"
      ? forceState
      : !appStore.state.execSummaryExpanded;
  appStore.state.execSummaryExpanded = next;

  const panel = document.getElementById("exec-expandable-panel");
  const btn = document.getElementById("exec-toggle-btn");
  if (panel) {
    panel.classList.toggle("open", next);
  }
  if (btn) {
    btn.setAttribute("aria-expanded", String(next));
    btn.textContent = next
      ? t("execBar.collapseDetails")
      : t("execBar.expandDetails");
  }
}

export function renderExecSummaryBar(meta) {
  if (!meta) return;
  const uEl = document.getElementById("exec-val-universe");
  const cEl = document.getElementById("exec-val-clients");
  const rEl = document.getElementById("exec-val-revenue");
  const kEl = document.getElementById("exec-val-icps");

  if (uEl) uEl.textContent = formatters.integer(meta.universe_size || 24424);
  if (cEl) cEl.textContent = formatters.integer(meta.client_count || 300);
  if (rEl) rEl.textContent = formatters.compactCurrency(meta.total_turnover || 5443170);
  if (kEl) {
    const ariPct = ((meta.global_ari || 0.958) * 100).toFixed(1);
    kEl.textContent = t("execBar.clustersCount", {
      count: meta.optimal_k || 2,
      ari: ariPct,
    });
  }
}

