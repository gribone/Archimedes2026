"use strict";
// "Φίλτρα δεδομένων" in the options panel: exclude activities (ΚΑΔ) or places, before or after the
// analysis. app.js reads window.bizFilters when an analysis starts.
(function () {
  const byId = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
    );
  // Case- and accent-insensitive matching for Greek (final ς treated as σ).
  const fold = (text) =>
    String(text || "")
      .normalize("NFD")
      .replace(/\p{M}/gu, "")
      .toLowerCase()
      .replace(/ς/g, "σ");
  const fmt = (n) => new Intl.NumberFormat("el-GR").format(n);
  const toggle = byId("filters-toggle");
  const panel = byId("filters-panel");
  let options = null;
  let loading = null;
  const chosen = { activities: new Map(), places: new Map() }; // value -> chip label

  const MODE_HELP = {
    post: "Τα προφίλ και οι στατιστικές μένουν ίδια. Οι εξαιρεμένες επιχειρήσεις απλώς δεν εμφανίζονται στις προτάσεις και στο αρχείο lookalikes.csv.",
    pre: "Τα φίλτρα εφαρμόζονται στα ακατέργαστα δεδομένα: η ανάλυση γίνεται σαν να μην υπήρχαν ποτέ οι εξαιρεμένες επιχειρήσεις.",
  };
  const mode = () => document.querySelector('input[name="filter-mode"]:checked').value;

  function load() {
    if (!loading)
      loading = fetch("/api/filters/options")
        .then((r) => {
          if (r.status === 401) location.replace("/login.html");
          if (!r.ok) throw new Error();
          return r.json();
        })
        .then((data) => {
          options = data;
          options.activities.forEach((a) => (a.key = fold(a.id + " " + a.label)));
          options.places = [
            ...data.prefectures.map((p) => ({
              value: "pref:" + p.name,
              label: "Π.Ε. " + p.name,
              detail: `${fmt(p.venues)} καταστήματα`,
              key: fold("πε περιφερειακη ενοτητα " + p.name),
            })),
            ...data.municipalities.map((m) => ({
              value: "mun:" + m.name,
              label: "Δήμος " + m.name,
              detail: `Π.Ε. ${m.prefecture} · ${fmt(m.venues)} καταστήματα`,
              key: fold("δημος " + m.name + " " + m.prefecture),
            })),
          ];
        })
        .catch(() => {
          loading = null;
          byId("filters-preview").textContent = "Δεν ήταν δυνατή η φόρτωση των επιλογών φίλτρων.";
        });
    return loading;
  }

  // One searchable picker: type → list of matches → click or Enter adds a chip.
  function picker(kind, inputId, listId, chipsId, search) {
    const input = byId(inputId),
      list = byId(listId),
      chips = byId(chipsId);
    let matches = [],
      active = -1;
    function close() {
      list.classList.add("hidden");
      input.setAttribute("aria-expanded", "false");
      active = -1;
    }
    function render() {
      const query = fold(input.value.trim());
      if (!options || !query) return close();
      matches = search(query).filter((m) => !chosen[kind].has(m.value)).slice(0, 40);
      list.innerHTML = matches.length
        ? matches
            .map(
              (m, i) =>
                `<li role="option" id="${listId}-${i}" data-i="${i}" aria-selected="${i === active}"><strong>${esc(m.label)}</strong><small>${esc(m.detail)}</small></li>`,
            )
            .join("")
        : '<li class="picker-empty">Δεν βρέθηκε κάτι.</li>';
      list.classList.remove("hidden");
      input.setAttribute("aria-expanded", "true");
      input.setAttribute("aria-activedescendant", active >= 0 ? `${listId}-${active}` : "");
    }
    function add(match) {
      chosen[kind].set(match.value, match.chip || match.label);
      input.value = "";
      close();
      drawChips();
      changed();
      input.focus();
    }
    function drawChips() {
      chips.innerHTML = [...chosen[kind]]
        .map(
          ([value, label]) =>
            `<span class="chip">${esc(label)}<button type="button" data-value="${esc(value)}" aria-label="Αφαίρεση: ${esc(label)}">✕</button></span>`,
        )
        .join("");
    }
    input.addEventListener("input", () => {
      active = -1;
      load().then(render);
    });
    input.addEventListener("focus", () => load());
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!matches.length) return;
        active = (active + (event.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
        render();
      } else if (event.key === "Enter") {
        event.preventDefault(); // never submits the upload form
        if (matches.length && !list.classList.contains("hidden")) add(matches[Math.max(active, 0)]);
      } else if (event.key === "Escape") close();
    });
    input.addEventListener("blur", () => setTimeout(close, 150));
    list.addEventListener("mousedown", (event) => {
      const item = event.target.closest("li[data-i]");
      if (item) {
        event.preventDefault();
        add(matches[Number(item.dataset.i)]);
      }
    });
    chips.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      chosen[kind].delete(button.dataset.value);
      drawChips();
      changed();
    });
    return { drawChips, input };
  }

  const activityPicker = picker("activities", "activity-search", "activity-options", "activity-chips", (query) => {
    const found = options.activities
      .filter((a) => a.key.includes(query))
      .map((a) => ({
        value: a.id,
        label: `${a.id} · ${a.label}`,
        chip: `${a.id} · ${a.label.length > 42 ? a.label.slice(0, 40) + "…" : a.label}`,
        detail: `${fmt(a.primary)} καταστήματα με κύριο ΚΑΔ · ${fmt(a.declared)} συνολικά`,
      }));
    // A typed code prefix (e.g. 5630) excludes the whole family of codes.
    if (/^\d{2,7}$/.test(query) && options.activities.some((a) => a.id.startsWith(query)))
      found.unshift({
        value: query,
        label: `Όλοι οι ΚΑΔ που ξεκινούν από ${query}`,
        chip: `ΚΑΔ ${query}…`,
        detail: `${options.activities.filter((a) => a.id.startsWith(query)).length} κωδικοί`,
      });
    return found;
  });
  const placePicker = picker("places", "place-search", "place-options", "place-chips", (query) =>
    options.places.filter((p) => p.key.includes(query)),
  );

  function payload() {
    if (!toggle.checked) return null;
    const places = [...chosen.places.keys()];
    const result = {
      mode: mode(),
      activity_scope: byId("activity-any").checked ? "any" : "primary",
      activities: [...chosen.activities.keys()],
      prefectures: places.filter((p) => p.startsWith("pref:")).map((p) => p.slice(5)),
      municipalities: places.filter((p) => p.startsWith("mun:")).map((p) => p.slice(4)),
    };
    return result.activities.length || places.length ? result : null;
  }

  let timer = null;
  function changed() {
    const isPre = mode() === "pre";
    byId("filter-mode-help").textContent = MODE_HELP[mode()];
    byId("filter-pre-warning").classList.toggle("hidden", !(isPre && payload()));
    if (typeof updateOptionsBadge === "function") updateOptionsBadge();
    clearTimeout(timer);
    const body = payload();
    if (!body) {
      byId("filters-preview").textContent = "Δεν έχει επιλεγεί κάποια εξαίρεση.";
      return;
    }
    byId("filters-preview").textContent = "Υπολογισμός…";
    timer = setTimeout(async () => {
      try {
        const response = await fetch("/api/filters/preview", {
          method: "POST",
          headers: { "X-ForceCRM": "local", "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error();
        const p = await response.json();
        if (JSON.stringify(payload()) !== JSON.stringify(body)) return; // a newer change is pending
        byId("filters-preview").textContent =
          `Εξαιρούνται ${fmt(p.excluded_venues)} από ${fmt(p.total_venues)} καταστήματα ` +
          `(${fmt(p.excluded_businesses)} επιχειρήσεις χωρίς άλλο κατάστημα).`;
      } catch {
        byId("filters-preview").textContent = "Δεν ήταν δυνατός ο υπολογισμός.";
      }
    }, 250);
  }

  toggle.addEventListener("change", () => {
    panel.classList.toggle("hidden", !toggle.checked);
    if (toggle.checked) load();
    changed();
  });
  document.querySelectorAll('input[name="filter-mode"]').forEach((radio) => radio.addEventListener("change", changed));
  byId("activity-any").addEventListener("change", changed);
  byId("filters-clear").addEventListener("click", () => {
    chosen.activities.clear();
    chosen.places.clear();
    byId("activity-any").checked = false;
    document.querySelector('input[name="filter-mode"][value="post"]').checked = true;
    activityPicker.drawChips();
    placePicker.drawChips();
    changed();
  });
  byId("filter-mode-help").textContent = MODE_HELP.post;

  window.bizFilters = {
    payload,
    active: () => Boolean(payload()),
    setDisabled(value) {
      panel.querySelectorAll("input, button").forEach((el) => (el.disabled = value));
      toggle.disabled = value;
    },
  };
})();
