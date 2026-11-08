/**
 * Charts & Visuals Helper (`src/static/js/components/charts.js`)
 * Renders portfolio share donuts and statistical proof bars.
 */

export const ICP_COLORS = [
  "#38bdf8",
  "#10b981",
  "#818cf8",
  "#f59e0b",
  "#ec4899",
  "#a855f7",
];

export function buildDonutGradient(items) {
  let acc = 0;
  const stops = [];
  items.forEach((it, idx) => {
    const share = Math.max(0, Math.min(1, Number(it.share) || 0));
    const degStart = (acc * 360).toFixed(2);
    acc += share;
    const degEnd = (acc * 360).toFixed(2);
    const color = it.color || ICP_COLORS[idx % ICP_COLORS.length];
    stops.push(`${color} ${degStart}deg ${degEnd}deg`);
  });
  return `conic-gradient(${stops.join(", ") || "#334155 0deg 360deg"})`;
}

