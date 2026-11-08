/**
 * Archimedes 2026 — Tables Module (`src/static/js/tables.js`)
 * High-performance lead and client table rendering, pagination, and CSV export.
 */

import { state } from './state.js';

export function renderLeadsTable() {
  const container = document.getElementById('leads-tbody');
  if (!container) return;

  const filtered = getFilteredLeads();
  const total = filtered.length;
  const page = state.leadsFilter.page;
  const pageSize = state.leadsFilter.pageSize;
  const start = (page - 1) * pageSize;
  const pageItems = filtered.slice(start, start + pageSize);

  if (pageItems.length === 0) {
    container.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-slate-500">Δεν βρέθηκαν αποτελέσματα</td></tr>`;
    return;
  }

  const rowsHtml = pageItems.map(row => {
    const scorePct = Math.round(row.score * 100);
    const scoreColor = scorePct >= 90 ? 'text-emerald-400 bg-emerald-500/10' :
                       scorePct >= 80 ? 'text-blue-400 bg-blue-500/10' :
                                        'text-amber-400 bg-amber-500/10';

    return `
      <tr class="hover:bg-slate-900/60 border-b border-slate-800/50 transition-colors">
        <td class="px-4 py-3 font-mono text-slate-400">#${row.rank}</td>
        <td class="px-4 py-3">
          <button onclick="window.inspectCompany(${row.company_id})" class="text-white hover:text-blue-400 font-medium text-left transition-colors">
            ${escapeHtml(row.name || 'Επιχείρηση')}
          </button>
          <div class="text-[11px] text-slate-500 font-mono">ΑΦΜ: ${row.vat} · ${escapeHtml(row.legal_type || '')}</div>
        </td>
        <td class="px-4 py-3 text-slate-300">
          <div>${escapeHtml(row.municipality || row.city || '—')}</div>
          <div class="text-[11px] text-slate-500">${escapeHtml(row.prefecture || '')}</div>
        </td>
        <td class="px-4 py-3">
          <span class="px-2 py-0.5 rounded text-xs font-semibold ${scoreColor}">
            ${scorePct}%
          </span>
        </td>
        <td class="px-4 py-3 text-xs text-slate-300 truncate max-w-xs" title="${escapeHtml(row.reason || '')}">
          ${escapeHtml(row.reason || '—')}
        </td>
        <td class="px-4 py-3 text-right">
          <button onclick="window.focusOnMap(${row.lat}, ${row.lon}, ${row.company_id}, '${escapeHtml(row.name)}', '${row.profile_id}')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors">
            📍 Χάρτης
          </button>
        </td>
      </tr>
    `;
  }).join('');

  container.innerHTML = rowsHtml;
}

export function getFilteredLeads() {
  const f = state.leadsFilter;
  let list = state.lookalikes || [];

  if (f.profile !== 'ALL') {
    list = list.filter(r => r.profile_id === f.profile);
  }
  if (f.search) {
    const q = f.search.toLowerCase();
    list = list.filter(r => 
      (r.name && r.name.toLowerCase().includes(q)) ||
      (r.vat && r.vat.includes(q)) ||
      (r.municipality && r.municipality.toLowerCase().includes(q))
    );
  }
  return list;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

