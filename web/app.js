"use strict";
const $ = (id) => document.getElementById(id);
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (n) => new Intl.NumberFormat("el-GR").format(n);
const percent = (n) =>
  new Intl.NumberFormat("el-GR", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(n);
let state = null,
  jobId = null,
  page = 0,
  busy = false;
const pageSize = 30;
const views = {
  overview: [
    "Επισκόπηση",
    "Ποιος είναι ο επόμενος πελάτης σας;",
    "Γνωρίστε τους καλύτερους πελάτες σας. Βρείτε τις επιχειρήσεις που τους μοιάζουν.",
  ],
  profiles: [
    "Προφίλ πελατών",
    "Τα χαρακτηριστικά που ενώνουν τους πελάτες σας.",
    "Εξερευνήστε την αξία, τη σταθερότητα και την τεκμηρίωση κάθε προφίλ.",
  ],
  prospects: [
    "Νέες ευκαιρίες",
    "Οι επόμενες επιχειρήσεις στη λίστα σας.",
    "Μία πρόταση ανά ΑΦΜ, με το κατάστημα που ταιριάζει περισσότερο.",
  ],
  validation: [
    "Τεκμηρίωση",
    "Κάθε αποτέλεσμα έχει τη δική του τεκμηρίωση.",
    "Δείτε την ποιότητα δεδομένων, τη μέθοδο και τους περιορισμούς της ανάλυσης.",
  ],
};
function showView(view) {
  document
    .querySelectorAll(".view")
    .forEach((el) => el.classList.add("hidden"));
  $("view-" + view).classList.remove("hidden");
  document
    .querySelectorAll(".nav")
    .forEach((el) => el.classList.toggle("active", el.dataset.view === view));
  [
    $("breadcrumb").textContent,
    $("page-heading").textContent,
    $("page-subtitle").textContent,
  ] = views[view];
  if (view === "prospects" && state) drawMap();
}
document
  .querySelectorAll("[data-view]")
  .forEach((el) =>
    el.addEventListener("click", () => showView(el.dataset.view)),
  );
function error(message) {
  $("error").textContent = message;
  $("error").classList.toggle("hidden", !message);
}
function setBusy(value) {
  busy = value;
  ["analyze-button", "demo-button", "portfolio", "weights-toggle", "live-toggle"].forEach(
    (id) => ($(id).disabled = value),
  );
  $("progress").classList.toggle("hidden", !value);
}
async function request(url, options) {
  const response = await fetch(url, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Server error (${response.status}).`);
  }
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  return body;
}
const DEFAULT_WEIGHTS = { activity: 65, geography: 20, legal_type: 10, age: 5 };
const WEIGHT_LABELS = {
  activity: "Δραστηριότητα",
  geography: "Γεωγραφία",
  legal_type: "Νομική μορφή",
  age: "Ηλικία επιχείρησης",
};
const weights = { ...DEFAULT_WEIGHTS };
function weightsChanged() {
  return Object.keys(DEFAULT_WEIGHTS).some(
    (k) => weights[k] !== DEFAULT_WEIGHTS[k],
  );
}
function updateWeightSummary() {
  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  document.querySelectorAll(".weight-row").forEach((row) => {
    const k = row.querySelector("input").dataset.weight;
    row.querySelector("b").textContent =
      (total ? Math.round((weights[k] / total) * 100) : 0) + "%";
  });
  $("weights-total").textContent = !total
    ? "Τουλάχιστον ένα βάρος πρέπει να είναι πάνω από 0"
    : weightsChanged()
      ? "Προσαρμοσμένα βάρη · κανονικοποιούνται στο 100%"
      : "Προεπιλεγμένα βάρη";
  $("weights-total").classList.toggle("weights-custom", weightsChanged());
}
function renderWeights() {
  $("weights-sliders").innerHTML = Object.keys(DEFAULT_WEIGHTS)
    .map(
      (k) =>
        `<label class="weight-row"><span>${WEIGHT_LABELS[k]}</span><input type="range" min="0" max="100" step="5" value="${weights[k]}" data-weight="${k}" aria-label="${WEIGHT_LABELS[k]}" /><b></b></label>`,
    )
    .join("");
  updateWeightSummary();
}
$("weights-sliders").addEventListener("input", (event) => {
  const key = event.target.dataset.weight;
  if (!key) return;
  weights[key] = Number(event.target.value);
  updateWeightSummary();
});
$("weights-toggle").addEventListener("change", (event) => {
  $("weights-panel").classList.toggle("hidden", !event.target.checked);
  if (event.target.checked) renderWeights();
});
$("weights-reset").addEventListener("click", () => {
  Object.assign(weights, DEFAULT_WEIGHTS);
  renderWeights();
});
function updateOptionsBadge() {
  const active = [
    $("weights-toggle").checked && weightsChanged(),
    $("live-toggle").checked,
  ].filter(Boolean).length;
  $("options-badge").textContent = active ? `${active} ενεργ${active === 1 ? "ή" : "ές"}` : "";
  $("options-badge").classList.toggle("hidden", !active);
}
$("options-toggle").addEventListener("click", () => {
  const open = $("options-toggle").getAttribute("aria-expanded") !== "true";
  $("options-toggle").setAttribute("aria-expanded", String(open));
  $("options-panel").classList.toggle("hidden", !open);
});
$("weights-sliders").addEventListener("input", updateOptionsBadge);
$("weights-reset").addEventListener("click", updateOptionsBadge);
$("weights-toggle").addEventListener("change", updateOptionsBadge);
function updateLiveEstimate() {
  const limit = Number($("live-limit").value);
  const minutes = Math.ceil((limit * 60) / 8 / 60);
  $("live-estimate").textContent =
    `Έως ~${minutes} λεπτά για ${limit} επιχειρήσεις (όριο ΓΕΜΗ ~8 αιτήματα/λεπτό). Αποτελέσματα των τελευταίων 7 ημερών επαναχρησιμοποιούνται.`;
}
$("live-toggle").addEventListener("change", (event) => {
  $("live-panel").classList.toggle("hidden", !event.target.checked);
  updateLiveEstimate();
  updateOptionsBadge();
  renderLive();
});
$("live-limit").addEventListener("change", updateLiveEstimate);

// Live GEMI check of the top prospects (server-side; opt-in).
let live = null,
  liveTimer = null;
const LIVE_LABELS = {
  active: "Ενεργή",
  inactive: "Ανενεργή",
  not_found: "Δεν βρέθηκε",
  error: "Σφάλμα",
};
function liveBadge(vat) {
  const result = live?.results[vat];
  if (!result) return "";
  const text = result.status === "inactive" && result.description ? result.description : LIVE_LABELS[result.status];
  return ` <span class="live-badge live-${result.status}" title="ΓΕΜΗ · ${escapeHTML(result.checked_at || "")}">● ${escapeHTML(text)}</span>`;
}
function renderLive() {
  const show = Boolean(state) && ($("live-toggle").checked || live);
  $("live-bar").classList.toggle("hidden", !show);
  $("live-filter").classList.toggle("hidden", !live);
  if (!show) return;
  const running = live?.status === "running";
  $("live-start").classList.toggle("hidden", running);
  $("live-start").textContent = live ? "Νέος έλεγχος στο ΓΕΜΗ →" : "Έλεγχος στο ΓΕΜΗ →";
  $("live-stop").classList.toggle("hidden", !running);
  if (!live) {
    $("live-message").textContent = "Έλεγχος ενεργής κατάστασης των κορυφαίων προτάσεων στο ΓΕΜΗ.";
    return;
  }
  const counts = Object.values(live.results).reduce((acc, r) => ((acc[r.status] = (acc[r.status] || 0) + 1), acc), {});
  const summary = ["active", "inactive", "not_found", "error"]
    .filter((k) => counts[k])
    .map((k) => `${LIVE_LABELS[k]}: ${counts[k]}`)
    .join(" · ");
  $("live-message").textContent = live.message + (summary ? ` — ${summary}` : "");
  $("live-bar").classList.toggle("live-error", live.status === "error");
}
async function startLive() {
  if (!jobId || !state) return;
  const form = new FormData();
  form.append("api_key", $("live-key").value);
  form.append("limit", $("live-limit").value);
  try {
    live = await request(`/api/jobs/${jobId}/live-check`, {
      method: "POST",
      headers: { "X-ForceCRM": "local" },
      body: form,
    });
    renderLive();
    pollLive();
  } catch (err) {
    live = { status: "error", message: err.message, results: live?.results || {} };
    renderLive();
  }
}
async function pollLive() {
  clearTimeout(liveTimer);
  const current = jobId;
  try {
    const data = await request(`/api/jobs/${current}/live`);
    if (current !== jobId) return;
    live = data;
    renderLive();
    renderCandidates();
    if (!$("view-prospects").classList.contains("hidden")) drawMap();
    if (data.status === "running") liveTimer = setTimeout(pollLive, 3000);
  } catch (err) {
    live = { ...(live || { results: {} }), status: "error", message: err.message };
    renderLive();
  }
}
$("live-start").addEventListener("click", startLive);
$("live-stop").addEventListener("click", async () => {
  try {
    await request(`/api/jobs/${jobId}/live-check/stop`, { method: "POST", headers: { "X-ForceCRM": "local" } });
  } catch {}
  pollLive();
});
$("live-filter").addEventListener("input", () => {
  page = 0;
  renderCandidates();
  drawMap();
});
async function startAnalysis(demo) {
  if (busy) return;
  error("");
  const form = new FormData();
  form.append("demo", String(demo));
  if ($("weights-toggle").checked && weightsChanged()) {
    if (!Object.values(weights).some((v) => v > 0)) {
      error("Τουλάχιστον ένα βάρος πρέπει να είναι πάνω από 0.");
      return;
    }
    form.append("weights", JSON.stringify(weights));
  }
  if (!demo) {
    const file = $("portfolio").files[0];
    if (!file) {
      error("Επιλέξτε πρώτα ένα αρχείο .xlsx.");
      return;
    }
    form.append("file", file);
  }
  setBusy(true);
  clearTimeout(liveTimer);
  live = null;
  $("live-filter").value = "";
  $("notice").classList.add("hidden");
  // Existing results are cleared when a new job begins to avoid stale exports.
  state = null;
  $("download-all").classList.add("hidden");
  $("overview-results").classList.add("hidden");
  [
    "stat-customers",
    "stat-profiles",
    "stat-opportunities",
    "nav-count",
  ].forEach((id) => ($(id).textContent = "—"));
  ["profiles-list", "validation-content"].forEach((id) =>
    $(id).replaceChildren(),
  );
  $("prospects-content").classList.add("hidden");
  document
    .querySelectorAll(".empty-state")
    .forEach((el) => el.classList.remove("hidden"));
  try {
    const data = await request("/api/analyze", {
      method: "POST",
      headers: { "X-ForceCRM": "local" },
      body: form,
    });
    jobId = data.job_id;
    await poll();
  } catch (err) {
    error(err.message);
    setBusy(false);
  }
}
async function poll() {
  try {
    const job = await request("/api/jobs/" + jobId);
    $("progress-message").textContent = job.message;
    if (job.status === "error") {
      error(job.message);
      setBusy(false);
      return;
    }
    if (job.status === "complete") {
      state = job;
      render();
      setBusy(false);
      renderLive();
      if ($("live-toggle").checked) startLive();
      return;
    }
    setTimeout(poll, 1500);
  } catch (err) {
    error(err.message);
    setBusy(false);
  }
}
$("upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  startAnalysis(false);
});
$("demo-button").addEventListener("click", () => startAnalysis(true));
$("portfolio").addEventListener("change", () => {
  $("file-label").textContent =
    $("portfolio").files[0]?.name || "Επιλέξτε ή σύρετε ένα αρχείο Excel";
});
["dragenter", "dragover"].forEach((type) =>
  $("dropzone").addEventListener(type, () =>
    $("dropzone").classList.add("drag"),
  ),
);
["dragleave", "drop"].forEach((type) =>
  $("dropzone").addEventListener(type, () =>
    $("dropzone").classList.remove("drag"),
  ),
);
function summary(p) {
  return `<article class="card profile-summary"><span class="tag">${escapeHTML(p.profile_id)}</span><h3>${escapeHTML(p.title)}</h3><p>${p.significant_features.length} τεκμηριωμένα χαρακτηριστικά · ${percent(p.stability)} σταθερότητα</p><div class="metrics"><div><strong>${number(p.customer_count)}</strong><small>πελάτες</small></div><div><strong>${percent(p.revenue_share)}</strong><small>μερίδιο τζίρου</small></div></div></article>`;
}
function activityMix(p) {
  if (!p.activity_mix || !p.activity_mix.length) return "";
  return `<h2>Συχνότερες κύριες δραστηριότητες</h2><p class="caption">Περιγραφικά ποσοστά ΑΦΜ. Στατιστικά τεκμηριωμένη διαφορά υπάρχει μόνο όπου η δραστηριότητα εμφανίζεται και στον πίνακα χαρακτηριστικών.</p><div class="table-scroll"><table class="evidence-table"><thead><tr><th>Κύρια δραστηριότητα</th><th>Στο προφίλ</th><th>Στο σύνολο</th><th>Αναλογία</th></tr></thead><tbody>${p.activity_mix.map((a) => `<tr><td>${escapeHTML(a.label)}</td><td>${a.count}/${a.total} (${percent(a.profile_rate)})</td><td>${percent(a.universe_rate)}</td><td>${a.lift.toFixed(2)}×</td></tr>`).join("")}</tbody></table></div>`;
}

function portfolioTraits() {
  const rows = (state.report && state.report.portfolio_traits) || [];
  if (!state.report || !("portfolio_traits" in state.report)) return "";
  return `<article class="card profile-detail"><header><div><span class="tag">ΟΛΟ ΤΟ ΠΕΛΑΤΟΛΟΓΙΟ</span><h3>Κοινά χαρακτηριστικά πέρα από τη γεωγραφία</h3></div></header><p class="description">Τα προφίλ χωρίζουν τους πελάτες κυρίως ανά περιοχή. Εδώ φαίνονται δραστηριότητες, νομικές μορφές και άλλα επιχειρηματικά χαρακτηριστικά που υπερεκπροσωπούνται σε όλο το πελατολόγιο σε σχέση με το σύνολο της εστίασης.</p><p class="caption">Fisher ανά ΑΦΜ · διόρθωση Benjamini–Hochberg σε ${number(state.report.portfolio_traits_tested || 0)} ελέγχους · q ≤ 0,05, ≥ 5 ΑΦΜ, lift ≥ 1,2</p><div class="table-scroll"><table class="evidence-table"><thead><tr><th>Χαρακτηριστικό</th><th>Στο πελατολόγιο</th><th>Στο σύνολο</th><th>Lift</th><th>q-value</th></tr></thead><tbody>${rows.map((f) => `<tr><td>${escapeHTML(f.label)}</td><td>${f.count}/${f.total} (${percent(f.profile_rate)})</td><td>${percent(f.universe_rate)}</td><td>${f.lift.toFixed(2)}×</td><td>${f.q_value.toExponential(2)}</td></tr>`).join("") || '<tr><td colspan="5">Κανένα επιχειρηματικό χαρακτηριστικό δεν διαφέρει στατιστικά από την αγορά.</td></tr>'}</tbody></table></div></article>`;
}

function render() {
  const r = state.report;
  $("stat-customers").textContent = number(r.portfolio_customers);
  $("stat-profiles").textContent = r.profile_count;
  $("nav-count").textContent = r.profile_count;
  $("stat-opportunities").textContent = number(r.candidate_count);
  $("download-all").href = `/api/jobs/${jobId}/export/all.zip`;
  $("download-all").classList.remove("hidden");
  if (r.warnings.length) {
    $("notice").textContent = r.warnings.join("\n");
    $("notice").classList.remove("hidden");
  }
  $("profile-summary").innerHTML = state.profiles.map(summary).join("");
  $("overview-results").classList.remove("hidden");
  document
    .querySelectorAll(".empty-state")
    .forEach((el) => el.classList.add("hidden"));
  $("profiles-list").innerHTML = portfolioTraits() + state.profiles
    .map(
      (p) =>
        `<article class="card profile-detail"><header><div><span class="tag">${escapeHTML(p.profile_id)}</span><h3>${escapeHTML(p.title)}</h3></div><span class="tag">${p.ai_generated ? "AI-GENERATED · LOCAL LLM" : (state.report.demo ? "DEMO · TEMPLATE" : "ΕΠΑΛΗΘΕΥΜΕΝΟ ΠΡΟΤΥΠΟ")}</span></header><p class="description">${escapeHTML(p.description)}</p><p class="caption">${escapeHTML(p.description_status || "")}</p><div class="metrics"><div><strong>${number(p.customer_count)}</strong><small>ΑΦΜ · ${number(p.venue_count)} καταστήματα</small></div><div><strong>${percent(p.revenue_share)}</strong><small>μερίδιο τζίρου</small></div><div><strong>${percent(p.stability)}</strong><small>σταθερότητα bootstrap</small></div><div><strong>${percent(p.confidence_score)}</strong><small>σχετική εμπιστοσύνη προσαρμογής</small></div><div><strong>${p.quality === "strong" ? "Ισχυρή" : "Περιορισμένη"}</strong><small>ποιότητα τεκμηρίωσης</small></div></div>${activityMix(p)}<h2>Χαρακτηριστικά με αυξημένη παρουσία</h2><p class="caption">Μονάδα μέτρησης: μοναδικά ΑΦΜ · Fisher + συνολική διόρθωση Benjamini–Hochberg · διερευνητική ανάλυση</p><div class="table-scroll"><table class="evidence-table"><thead><tr><th>Χαρακτηριστικό</th><th>Στο προφίλ</th><th>Στο σύνολο</th><th>Lift</th><th>p-value</th><th>q-value</th></tr></thead><tbody>${p.significant_features.map((f) => `<tr><td>${escapeHTML(f.label)}</td><td>${f.count}/${f.total} (${percent(f.profile_rate)})</td><td>${percent(f.universe_rate)}</td><td>${f.lift.toFixed(2)}×</td><td>${f.p_value.toExponential(2)}</td><td>${f.q_value.toExponential(2)}</td></tr>`).join("") || '<tr><td colspan="6">Δεν βρέθηκαν χαρακτηριστικά με επαρκή στατιστική τεκμηρίωση.</td></tr>'}</tbody></table></div><details><summary>Προέλευση κάθε πρότασης και παράμετροι αναπαραγωγής</summary><pre>${escapeHTML(JSON.stringify({ claims: p.claims, llm: p.llm, stability: p.stability_detail, acceptance_cosine_distance: p.acceptance_cosine_distance }, null, 2))}</pre></details></article>`,
    )
    .join("");
  $("profile-filter").innerHTML =
    '<option value="">Όλα τα προφίλ</option>' +
    state.profiles
      .map(
        (p) =>
          `<option value="${escapeHTML(p.profile_id)}">${escapeHTML(p.profile_id)}</option>`,
      )
      .join("");
  $("prospects-content").classList.remove("hidden");
  page = 0;
  renderCandidates();
  renderValidation();
}
function filtered() {
  const liveFilter = $("live-filter").value;
  const profile = $("profile-filter").value,
    query = $("search").value.toLocaleLowerCase("el-GR");
  return state.candidates.filter(
    (c) =>
      (!profile || c.profile_id === profile) &&
      (!liveFilter ||
        (liveFilter === "active"
          ? live?.results[c.vat]?.status === "active"
          : live?.results[c.vat]?.status !== "inactive")) &&
      (!query ||
        [c.company_name_el, c.city, c.vat].some((v) =>
          String(v).toLocaleLowerCase("el-GR").includes(query),
        )),
  );
}
function renderCandidates() {
  if (!state) return;
  const rows = filtered();
  page = Math.min(page, Math.max(0, Math.ceil(rows.length / pageSize) - 1));
  $("filtered-count").textContent = number(rows.length);
  $("candidates-body").innerHTML =
    rows
      .slice(page * pageSize, (page + 1) * pageSize)
      .map(
        (c) =>
          `<tr><td>#${number(c.rank)}</td><td><strong>${escapeHTML(c.company_name_el)}</strong>${liveBadge(c.vat)}<small>${escapeHTML(c.city)} · ΑΦΜ ${escapeHTML(c.vat)}</small></td><td><span class="tag">${escapeHTML(c.profile_id)}</span></td><td class="score">${percent(c.score)}<span class="score-bar"><i style="width:${Math.max(0, Math.min(100, c.score * 100))}%"></i></span></td><td>${escapeHTML(c.rationale)}</td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="5">Δεν βρέθηκαν επιχειρήσεις για αυτά τα φίλτρα.</td></tr>';
  $("page-info").textContent = rows.length
    ? `${number(page * pageSize + 1)}–${number(Math.min((page + 1) * pageSize, rows.length))} από ${number(rows.length)}`
    : "0 αποτελέσματα";
  $("prev-page").disabled = page === 0;
  $("next-page").disabled = (page + 1) * pageSize >= rows.length;
}
["profile-filter", "search"].forEach((id) =>
  $(id).addEventListener("input", () => {
    page = 0;
    renderCandidates();
    drawMap();
  }),
);
$("prev-page").addEventListener("click", () => {
  page--;
  renderCandidates();
});
$("next-page").addEventListener("click", () => {
  page++;
  renderCandidates();
});
document.addEventListener("themechange", () => {
  if (!$("view-prospects")?.classList.contains("hidden")) drawMap();
});
const MAP_BOUNDS = { west: 19.2, east: 29.9, south: 34.7, north: 41.9 };
const PROFILE_COLORS = ["#256653", "#c29b51", "#6a80a1", "#8c7092", "#a06e50", "#7f9856"];
let greeceMap = null;
fetch("/greece-map.json")
  .then((response) => (response.ok ? response.json() : null))
  .then((data) => {
    greeceMap = data;
    if (state && !$("view-prospects").classList.contains("hidden")) drawMap();
  })
  .catch(() => {});
function mapProjection(w, h) {
  // Equirectangular with cos(latitude) correction, so Greece keeps its real proportions.
  const kx = Math.cos((((MAP_BOUNDS.north + MAP_BOUNDS.south) / 2) * Math.PI) / 180);
  const spanX = (MAP_BOUNDS.east - MAP_BOUNDS.west) * kx,
    spanY = MAP_BOUNDS.north - MAP_BOUNDS.south;
  const pad = 36,
    scale = Math.min((w - 2 * pad) / spanX, (h - 2 * pad) / spanY);
  const ox = (w - spanX * scale) / 2,
    oy = (h - spanY * scale) / 2;
  return (lon, lat) => [
    ox + (lon - MAP_BOUNDS.west) * kx * scale,
    oy + (MAP_BOUNDS.north - lat) * scale,
  ];
}
function drawRings(ctx, rings, project) {
  ctx.beginPath();
  rings.forEach((ring) => {
    ring.forEach(([lon, lat], i) => {
      const [x, y] = project(lon, lat);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.closePath();
  });
}
function drawMap() {
  if (!state) return;
  const canvas = $("map"),
    ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const project = mapProjection(w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = token("--map-sea");
  ctx.fillRect(0, 0, w, h);
  // Graticule
  ctx.strokeStyle = token("--map-grid");
  ctx.fillStyle = token("--map-label");
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.lineWidth = 1;
  for (let lon = 20; lon <= 29; lon++) {
    const [x] = project(lon, MAP_BOUNDS.south);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(lon + "°E", x + 3, h - 8);
  }
  for (let lat = 35; lat <= 41; lat++) {
    const [, y] = project(MAP_BOUNDS.west, lat);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillText(lat + "°N", 6, y - 4);
  }
  if (greeceMap) {
    ctx.lineJoin = "round";
    drawRings(ctx, greeceMap.neighbours, project);
    ctx.fillStyle = token("--map-neighbour");
    ctx.fill("evenodd");
    ctx.strokeStyle = token("--map-border");
    ctx.lineWidth = 1;
    ctx.stroke();
    drawRings(ctx, greeceMap.greece, project);
    ctx.fillStyle = token("--map-land");
    ctx.fill("evenodd");
    ctx.strokeStyle = token("--map-coast");
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
  const rows = filtered().slice(0, 2000);
  const profiles = new Set();
  rows.forEach((c) => {
    if (c.latitude === null || c.longitude === null) return;
    if (c.longitude < MAP_BOUNDS.west || c.longitude > MAP_BOUNDS.east || c.latitude < MAP_BOUNDS.south || c.latitude > MAP_BOUNDS.north)
      return;
    const [x, y] = project(c.longitude, c.latitude);
    const index = (Number(c.profile_id.split("-")[1]) - 1) % PROFILE_COLORS.length;
    profiles.add(c.profile_id);
    const status = live?.results[c.vat]?.status;
    ctx.globalAlpha = 0.75;
    ctx.fillStyle = PROFILE_COLORS[index];
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    if (status === "inactive") {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = token("--danger");
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  });
  ctx.globalAlpha = 1;
  // Legend
  const legend = [...profiles].sort();
  if (live && Object.values(live.results).some((r) => r.status === "inactive")) legend.push("inactive");
  ctx.font = "600 14px Segoe UI, sans-serif";
  legend.forEach((id, i) => {
    const y = 30 + i * 24,
      x = w - 190;
    if (id === "inactive") {
      ctx.strokeStyle = token("--danger");
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y - 5, 6, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.fillStyle = PROFILE_COLORS[(Number(id.split("-")[1]) - 1) % PROFILE_COLORS.length];
      ctx.beginPath();
      ctx.arc(x, y - 5, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = token("--ink");
    ctx.fillText(id === "inactive" ? "Ανενεργή (ΓΕΜΗ)" : id, x + 14, y);
  });
}
function renderValidation() {
  const r = state.report,
    v = r.validation;
  const metrics = v?.metrics || [];
  const e = r.enrichment;
  const coverage = e ? `<section class="card wide"><h2>Εμπλουτισμός καταστημάτων</h2><p>OpenStreetMap: ${number(e.osm_status_counts.matched || 0)} συνδέσεις σε ${number(e.venue_count)} καταστήματα (${percent(e.match_rate_all)}). Μεταξύ όσων έχουν GPS και διακριτικό όνομα: ${percent(e.match_rate_eligible)}. Οι αμφίβολες συνδέσεις μένουν ανοικτές.</p><p>ΕΛΣΤΑΤ: ${number(e.tourism_covered)} καταστήματα έχουν διαθέσιμες διανυκτερεύσεις ξενοδοχείων 2024 ανά κάτοικο του δήμου (απογραφή 2021).</p><a class="button outline" href="/api/jobs/${jobId}/export/enrichment.csv">↓ Στοιχεία ανά κατάστημα</a></section>` : "";
  const blockWeights = r.feature_configuration.block_weights;
  const customWeights = Boolean(r.feature_configuration.custom_weights);
  $("validation-content").innerHTML =
    `<div class="validation-grid">${coverage}<section class="card"><h2>Εσωτερικός έλεγχος ανάκτησης</h2><p>20% των ΑΦΜ αποκρύπτονται πριν από την εκπαίδευση. Όλα τα καταστήματά τους παραμένουν μαζί. Αυτός δεν είναι ο κρυφός έλεγχος του διοργανωτή.</p>${r.demo ? "<p><strong>Demo δεδομένα: οι παρακάτω τιμές δεν μετρούν πραγματική απόδοση πωλήσεων.</strong></p>" : ""}${customWeights ? `<p class="weights-warning">⚠ Η ανάλυση έγινε με προσαρμοσμένα βάρη. Η στήλη «Προεπιλεγμένα βάρη» δείχνει τον ίδιο έλεγχο με τα ελεγμένα βάρη.</p>` : ""}<table><thead><tr><th>Μέτρο</th><th>Μοντέλο</th>${customWeights ? "<th>Προεπιλεγμένα βάρη</th>" : ""}<th>Baseline</th><th>Διαφορά</th></tr></thead><tbody>${metrics.map((m) => `<tr><td>Recall@${m.k}</td><td>${percent(m.recall)}</td>${customWeights ? `<td>${m.default_weights_recall == null ? "—" : percent(m.default_weights_recall)}</td>` : ""}<td>${percent(m.baseline_recall)}</td><td>${(m.absolute_uplift * 100).toFixed(1)} π.μ.</td></tr>`).join("")}</tbody></table><p>Baseline: συχνότητα της κύριας δραστηριότητας στο πελατολόγιο εκπαίδευσης.</p></section><section class="card"><h2>Ποιότητα δεδομένων και ανάλυσης</h2><ul><li>${number(r.data_audit.venues)} καταστήματα · ${number(r.data_audit.businesses)} ΑΦΜ</li><li>Κάλυψη GPS: ${percent(r.data_audit.gps_coverage)}</li><li>Μέσο Adjusted Rand Index: ${r.bootstrap_ari.toFixed(3)}</li><li>${number(r.hypotheses_tested)} στατιστικές υποθέσεις εξετάστηκαν</li><li>${number(r.rejected_venues)} καταστήματα χωρίς αντιστοίχιση (NONE)</li><li>Διάρκεια: ${r.performance.seconds.toFixed(1)} δευτ. · RAM στο τέλος: ${r.performance.rss_mb_at_finish.toFixed(0)} MB</li></ul></section><section class="card wide"><h2>Πώς λειτουργεί — και πού χρειάζεται προσοχή</h2><p>Η ομοιότητα συνδυάζει δραστηριότητες (${percent(blockWeights.activity)}), γεωγραφία (${percent(blockWeights.geography)}), νομική μορφή (${percent(blockWeights.legal_type)}) και ηλικία (${percent(blockWeights.age)})${customWeights ? " — προσαρμοσμένα βάρη" : ""}. Οι κύριες δραστηριότητες έχουν μεγαλύτερο βάρος από τις συναφείς δευτερεύουσες. Ο τζίρος κατανέμεται στα καταστήματα κάθε ΑΦΜ, ώστε να μην διπλομετράται.</p><p>Η στατιστική τεκμηρίωση είναι διερευνητική: οι ομάδες έχουν σχηματιστεί από τα ίδια δεδομένα. Μια σταθερή ομάδα δεν συνεπάγεται ότι όλοι οι πελάτες της έχουν την ίδια πραγματική δραστηριότητα. Ένα ΑΦΜ με διαφορετικά καταστήματα μπορεί να συμμετέχει σε περισσότερα προφίλ.</p><p>Η εφαρμογή προτείνει επιχειρήσεις για ανθρώπινη αξιολόγηση. Δεν υπολογίζει πιστοληπτική ικανότητα, φερεγγυότητα ή πιθανότητα αγοράς. Οι περιγραφές με ένδειξη AI παράγονται από τοπικό μοντέλο με βάση τα στοιχεία του προφίλ.</p><div class="downloads">${["profiles.json", "assignments.csv", "lookalikes.csv", "report.json"].map((f) => `<a class="button outline" href="/api/jobs/${jobId}/export/${f}">↓ ${f}</a>`).join(" ")}</div><details><summary>Πλήρης αναφορά μεθοδολογίας και αναπαραγωγής</summary><pre>${escapeHTML(JSON.stringify(r, null, 2))}</pre></details></section></div>`;
}
request("/api/status")
  .then((s) => {
    $("stat-universe").textContent = number(s.data.businesses);
    $("stat-venues").textContent =
      number(s.data.venues) +
      " καταστήματα · " +
      percent(s.data.gps_coverage) +
      " με GPS";
    $("model-status").textContent = s.embedding_ready
      ? "Το embedding model είναι διαθέσιμο. Απαιτείται το Ollama με qwen3:1.7b."
      : "Για πλήρη ανάλυση: python run.py --setup-models · Το demo λειτουργεί χωρίς μοντέλα.";
  })
  .catch((err) => error(err.message));
request("/api/latest")
  .then((data) => {
    if (data.job_id && !busy) {
      jobId = data.job_id;
      setBusy(true);
      poll();
    }
  })
  .catch(() => {});
