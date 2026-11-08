/**
 * Archimedes 2026 — Charts Module (`src/static/js/charts.js`)
 * Chart.js instances for portfolio turnover, client count distribution, and silhouette analysis.
 */

import { state, ICP_COLORS } from './state.js';

export function initDonutCharts() {
  const revCanvas = document.getElementById('chart-rev-donut');
  const clientsCanvas = document.getElementById('chart-clients-donut');
  
  if (revCanvas && !state.chartRevDonut) {
    state.chartRevDonut = new Chart(revCanvas, {
      type: 'doughnut',
      data: { labels: [], datasets: [{ data: [], backgroundColor: ICP_COLORS, borderWidth: 0 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '75%',
        plugins: { legend: { display: false }, tooltip: { enabled: true } }
      }
    });
  }

  if (clientsCanvas && !state.chartClientsDonut) {
    state.chartClientsDonut = new Chart(clientsCanvas, {
      type: 'doughnut',
      data: { labels: [], datasets: [{ data: [], backgroundColor: ICP_COLORS, borderWidth: 0 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '75%',
        plugins: { legend: { display: false }, tooltip: { enabled: true } }
      }
    });
  }
}

export function renderDonutCharts() {
  if (!state.profiles || state.profiles.length === 0) return;
  const labels = state.profiles.map(p => p.short_title || p.title);
  const revData = state.profiles.map(p => p.turnover_share * 100);
  const clientData = state.profiles.map(p => p.client_share * 100);

  if (state.chartRevDonut) {
    state.chartRevDonut.data.labels = labels;
    state.chartRevDonut.data.datasets[0].data = revData;
    state.chartRevDonut.update();
  }

  if (state.chartClientsDonut) {
    state.chartClientsDonut.data.labels = labels;
    state.chartClientsDonut.data.datasets[0].data = clientData;
    state.chartClientsDonut.update();
  }
}

export function initSilhouetteChart() {
  const canvas = document.getElementById('chart-silhouette');
  if (!canvas || state.chartSilhouette) return;

  state.chartSilhouette = new Chart(canvas, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: '#3b82f6', borderRadius: 4 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { grid: { display: false } }
      }
    }
  });
}

