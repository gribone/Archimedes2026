/**
 * Archimedes 2026 — Drawer & Modals Module (`src/static/js/drawer.js`)
 * Company inspector, mini-map, LLM prompts, and CRM blacklist modals.
 */

import { state } from './state.js';
import * as api from './api.js';

let drawerMiniMap = null;

export async function inspectCompany(companyId) {
  state.selectedCompanyId = companyId;
  const drawer = document.getElementById('company-drawer-backdrop');
  const content = document.getElementById('company-drawer-content');
  const body = document.getElementById('company-drawer-body');
  if (!drawer || !content || !body) return;

  if (drawerMiniMap) {
    try { drawerMiniMap.remove(); } catch (e) {}
    drawerMiniMap = null;
  }

  drawer.classList.remove('opacity-0', 'pointer-events-none');
  content.classList.remove('scale-95', 'opacity-0');
  content.classList.add('scale-100', 'opacity-100');

  body.innerHTML = `
    <div class="py-12 text-center text-slate-400">
      <div class="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin mx-auto mb-3"></div>
      <div>Φόρτωση καρτέλας επιχείρησης...</div>
    </div>
  `;

  try {
    const data = await api.fetchCompany(companyId);
    if (!data.company) {
      body.innerHTML = `<div class="py-8 text-center text-rose-400">Η επιχείρηση δεν βρέθηκε.</div>`;
      return;
    }
    renderCompanyDrawerContent(data, body);
  } catch (err) {
    body.innerHTML = `<div class="py-8 text-center text-rose-400">Σφάλμα φόρτωσης: ${err.message}</div>`;
  }
}

export function closeCompanyDrawer() {
  const drawer = document.getElementById('company-drawer-backdrop');
  const content = document.getElementById('company-drawer-content');
  if (!drawer || !content) return;

  content.classList.remove('scale-100', 'opacity-100');
  content.classList.add('scale-95', 'opacity-0');
  drawer.classList.add('opacity-0', 'pointer-events-none');

  if (drawerMiniMap) {
    try { drawerMiniMap.remove(); } catch (e) {}
    drawerMiniMap = null;
  }
}

function renderCompanyDrawerContent(data, container) {
  const c = data.company;
  const verif = data.context?.verification || {};
  const isClosed = verif.is_closed;

  container.innerHTML = `
    <div class="space-y-4 text-xs">
      <div class="p-4 rounded-lg bg-slate-900 border border-slate-800">
        <h3 class="text-base font-bold text-white mb-1">${escapeHtml(c.name || c.company_name_el)}</h3>
        <div class="text-slate-400 font-mono">ΑΦΜ: ${c.vat} · ${c.legal_type || ''}</div>
        <div class="text-slate-400 mt-1">${c.street || ''} ${c.street_number || ''}, ${c.city || ''} (${c.prefecture_name || ''})</div>
      </div>
      <div class="p-3 rounded-lg border ${isClosed ? 'bg-rose-500/10 border-rose-500/25 text-rose-300' : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'}">
        <span class="font-semibold">${isClosed ? '🔴 Μη Ενεργή Επιχείρηση' : '🟢 Ενεργή Λειτουργία'}</span>
        <div class="text-[11px] mt-1 text-slate-400">${escapeHtml(verif.details || 'Καταγεγραμμένη στο Μητρώο.')}</div>
      </div>
    </div>
  `;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

