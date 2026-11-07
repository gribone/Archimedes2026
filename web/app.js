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
  ["analyze-button", "demo-button", "portfolio"].forEach(
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
async function startAnalysis(demo) {
  if (busy) return;
  error("");
  const form = new FormData();
  form.append("demo", String(demo));
  if (!demo) {
    const file = $("portfolio").files[0];
    if (!file) {
      error("Επιλέξτε πρώτα ένα αρχείο .xlsx.");
      return;
    }
    form.append("file", file);
  }
  setBusy(true);
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
  $("profiles-list").innerHTML = state.profiles
    .map(
      (p) =>
        `<article class="card profile-detail"><header><div><span class="tag">${escapeHTML(p.profile_id)}</span><h3>${escapeHTML(p.title)}</h3></div><span class="tag">${p.ai_generated ? "AI-GENERATED · LOCAL LLM" : (state.report.demo ? "DEMO · TEMPLATE" : "ΕΠΑΛΗΘΕΥΜΕΝΟ ΠΡΟΤΥΠΟ")}</span></header><p class="description">${escapeHTML(p.description)}</p><div class="metrics"><div><strong>${number(p.customer_count)}</strong><small>ΑΦΜ · ${number(p.venue_count)} καταστήματα</small></div><div><strong>${percent(p.revenue_share)}</strong><small>μερίδιο τζίρου</small></div><div><strong>${percent(p.stability)}</strong><small>σταθερότητα bootstrap</small></div><div><strong>${percent(p.confidence_score)}</strong><small>σχετική εμπιστοσύνη προσαρμογής</small></div><div><strong>${p.quality === "strong" ? "Ισχυρή" : "Περιορισμένη"}</strong><small>ποιότητα τεκμηρίωσης</small></div></div><h2>Χαρακτηριστικά με αυξημένη παρουσία</h2><p class="caption">Μονάδα μέτρησης: μοναδικά ΑΦΜ · Fisher + συνολική διόρθωση Benjamini–Hochberg · διερευνητική ανάλυση</p><div class="table-scroll"><table class="evidence-table"><thead><tr><th>Χαρακτηριστικό</th><th>Στο προφίλ</th><th>Στο σύνολο</th><th>Lift</th><th>p-value</th><th>q-value</th></tr></thead><tbody>${p.significant_features.map((f) => `<tr><td>${escapeHTML(f.label)}</td><td>${f.count}/${f.total} (${percent(f.profile_rate)})</td><td>${percent(f.universe_rate)}</td><td>${f.lift.toFixed(2)}×</td><td>${f.p_value.toExponential(2)}</td><td>${f.q_value.toExponential(2)}</td></tr>`).join("") || '<tr><td colspan="6">Δεν βρέθηκαν χαρακτηριστικά με επαρκή στατιστική τεκμηρίωση.</td></tr>'}</tbody></table></div><details><summary>Προέλευση κάθε πρότασης και παράμετροι αναπαραγωγής</summary><pre>${escapeHTML(JSON.stringify({ claims: p.claims, llm: p.llm, stability: p.stability_detail, acceptance_cosine_distance: p.acceptance_cosine_distance }, null, 2))}</pre></details></article>`,
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
  const profile = $("profile-filter").value,
    query = $("search").value.toLocaleLowerCase("el-GR");
  return state.candidates.filter(
    (c) =>
      (!profile || c.profile_id === profile) &&
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
          `<tr><td>#${number(c.rank)}</td><td><strong>${escapeHTML(c.company_name_el)}</strong><small>${escapeHTML(c.city)} · ΑΦΜ ${escapeHTML(c.vat)}</small></td><td><span class="tag">${escapeHTML(c.profile_id)}</span></td><td class="score">${percent(c.score)}<span class="score-bar"><i style="width:${Math.max(0, Math.min(100, c.score * 100))}%"></i></span></td><td>${escapeHTML(c.rationale)}</td></tr>`,
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
function drawMap() {
  if (!state) return;
  const canvas = $("map"),
    ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fafcf8";
  ctx.fillRect(0, 0, w, h);
  ctx.font = "12px Segoe UI";
  ctx.strokeStyle = "#e4eadd";
  ctx.fillStyle = "#8c9a85";
  for (let lon = 19; lon <= 29; lon++) {
    const x = 45 + ((lon - 19) / 10) * (w - 80);
    ctx.beginPath();
    ctx.moveTo(x, 15);
    ctx.lineTo(x, h - 30);
    ctx.stroke();
    ctx.fillText(lon + "° E", x - 12, h - 8);
  }
  for (let lat = 35; lat <= 42; lat++) {
    const y = 15 + ((42 - lat) / 7) * (h - 45);
    ctx.beginPath();
    ctx.moveTo(45, y);
    ctx.lineTo(w - 35, y);
    ctx.stroke();
    ctx.fillText(lat + "° N", 3, y + 4);
  }
  const colors = [
    "#256653",
    "#c29b51",
    "#6a80a1",
    "#8c7092",
    "#a06e50",
    "#7f9856",
  ];
  filtered()
    .slice(0, 2000)
    .forEach((c) => {
      if (
        c.latitude === null ||
        c.longitude === null ||
        c.latitude < 35 ||
        c.latitude > 42 ||
        c.longitude < 19 ||
        c.longitude > 29
      )
        return;
      const x = 45 + ((c.longitude - 19) / 10) * (w - 80),
        y = 15 + ((42 - c.latitude) / 7) * (h - 45);
      ctx.globalAlpha = 0.65;
      ctx.fillStyle =
        colors[(Number(c.profile_id.split("-")[1]) - 1) % colors.length];
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  ctx.globalAlpha = 1;
}
function renderValidation() {
  const r = state.report,
    v = r.validation;
  const metrics = v?.metrics || [];
  const e = r.enrichment;
  const coverage = e ? `<section class="card wide"><h2>Εμπλουτισμός καταστημάτων</h2><p>OpenStreetMap: ${number(e.osm_status_counts.matched || 0)} συνδέσεις σε ${number(e.venue_count)} καταστήματα (${percent(e.match_rate_all)}). Μεταξύ όσων έχουν GPS και διακριτικό όνομα: ${percent(e.match_rate_eligible)}. Οι αμφίβολες συνδέσεις μένουν ανοικτές.</p><p>ΕΛΣΤΑΤ: ${number(e.tourism_covered)} καταστήματα έχουν διαθέσιμες διανυκτερεύσεις ξενοδοχείων 2024 ανά κάτοικο του δήμου (απογραφή 2021).</p><a class="button outline" href="/api/jobs/${jobId}/export/enrichment.csv">↓ Στοιχεία ανά κατάστημα</a></section>` : "";
  $("validation-content").innerHTML =
    `<div class="validation-grid">${coverage}<section class="card"><h2>Εσωτερικός έλεγχος ανάκτησης</h2><p>20% των ΑΦΜ αποκρύπτονται πριν από την εκπαίδευση. Όλα τα καταστήματά τους παραμένουν μαζί. Αυτός δεν είναι ο κρυφός έλεγχος του διοργανωτή.</p>${r.demo ? "<p><strong>Demo δεδομένα: οι παρακάτω τιμές δεν μετρούν πραγματική απόδοση πωλήσεων.</strong></p>" : ""}<table><thead><tr><th>Μέτρο</th><th>Μοντέλο</th><th>Baseline</th><th>Διαφορά</th></tr></thead><tbody>${metrics.map((m) => `<tr><td>Recall@${m.k}</td><td>${percent(m.recall)}</td><td>${percent(m.baseline_recall)}</td><td>${(m.absolute_uplift * 100).toFixed(1)} π.μ.</td></tr>`).join("")}</tbody></table><p>Baseline: συχνότητα της κύριας δραστηριότητας στο πελατολόγιο εκπαίδευσης.</p></section><section class="card"><h2>Ποιότητα δεδομένων και ανάλυσης</h2><ul><li>${number(r.data_audit.venues)} καταστήματα · ${number(r.data_audit.businesses)} ΑΦΜ</li><li>Κάλυψη GPS: ${percent(r.data_audit.gps_coverage)}</li><li>Μέσο Adjusted Rand Index: ${r.bootstrap_ari.toFixed(3)}</li><li>${number(r.hypotheses_tested)} στατιστικές υποθέσεις εξετάστηκαν</li><li>${number(r.rejected_venues)} καταστήματα χωρίς αντιστοίχιση (NONE)</li><li>Διάρκεια: ${r.performance.seconds.toFixed(1)} δευτ. · RAM στο τέλος: ${r.performance.rss_mb_at_finish.toFixed(0)} MB</li></ul></section><section class="card wide"><h2>Πώς λειτουργεί — και πού χρειάζεται προσοχή</h2><p>Η ομοιότητα συνδυάζει δραστηριότητες (65%), γεωγραφία (20%), νομική μορφή (10%) και ηλικία (5%). Οι κύριες δραστηριότητες έχουν μεγαλύτερο βάρος από τις συναφείς δευτερεύουσες. Ο τζίρος κατανέμεται στα καταστήματα κάθε ΑΦΜ, ώστε να μην διπλομετράται.</p><p>Η στατιστική τεκμηρίωση είναι διερευνητική: οι ομάδες έχουν σχηματιστεί από τα ίδια δεδομένα. Μια σταθερή ομάδα δεν συνεπάγεται ότι όλοι οι πελάτες της έχουν την ίδια πραγματική δραστηριότητα. Ένα ΑΦΜ με διαφορετικά καταστήματα μπορεί να συμμετέχει σε περισσότερα προφίλ.</p><p>Η εφαρμογή προτείνει επιχειρήσεις για ανθρώπινη αξιολόγηση. Δεν υπολογίζει πιστοληπτική ικανότητα, φερεγγυότητα ή πιθανότητα αγοράς. Οι περιγραφές με ένδειξη AI παράγονται από τοπικό μοντέλο με βάση τα στοιχεία του προφίλ.</p><div class="downloads">${["profiles.json", "assignments.csv", "lookalikes.csv", "report.json"].map((f) => `<a class="button outline" href="/api/jobs/${jobId}/export/${f}">↓ ${f}</a>`).join(" ")}</div><details><summary>Πλήρης αναφορά μεθοδολογίας και αναπαραγωγής</summary><pre>${escapeHTML(JSON.stringify(r, null, 2))}</pre></details></section></div>`;
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
      ? "Το embedding model είναι διαθέσιμο. Απαιτείται το Ollama με ministral-3:3b."
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
