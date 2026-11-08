"use strict";
// Login session, first-login guided tour and the help chat. Loaded after app.js; it uses app.js's
// global showView() but keeps its own helpers inside this closure.
(function () {
  const byId = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
    );
  const post = (url, body, json = false) =>
    fetch(url, {
      method: "POST",
      headers: json
        ? { "X-ForceCRM": "local", "Content-Type": "application/json" }
        : { "X-ForceCRM": "local" },
      body: json ? JSON.stringify(body) : body,
    });
  let user = null;
  const goView = (view) => {
    if (typeof showView === "function") showView(view);
  };

  // ---------------------------------------------------------------- user menu
  const menuButton = byId("user-button");
  const dropdown = byId("user-dropdown");
  function setMenu(open) {
    dropdown.classList.toggle("hidden", !open);
    menuButton.setAttribute("aria-expanded", String(open));
  }
  menuButton.addEventListener("click", (event) => {
    event.stopPropagation();
    setMenu(dropdown.classList.contains("hidden"));
  });
  document.addEventListener("click", (event) => {
    if (!dropdown.contains(event.target)) setMenu(false);
  });
  byId("logout-button").addEventListener("click", async () => {
    await post("/api/auth/logout").catch(() => {});
    location.replace("/login.html");
  });
  byId("menu-tour").addEventListener("click", () => {
    setMenu(false);
    tour.start();
  });
  byId("tour-button").addEventListener("click", () => tour.start());

  function initials(name) {
    const parts = name.trim().split(/\s+/).filter(Boolean);
    const letters = parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2);
    return letters.toLocaleUpperCase("el-GR");
  }

  // ---------------------------------------------------------------- guided tour
  const STEPS = [
    {
      title: "Καλώς ήρθατε στο Bizcover!",
      body: "Σε 1 λεπτό θα δείτε πώς το Bizcover βρίσκει ποιοι είναι οι καλύτεροι πελάτες σας και ποιες επιχειρήσεις τους μοιάζουν. Μπορείτε να ξαναδείτε την ξενάγηση από το κουμπί «?» πάνω δεξιά.",
    },
    {
      target: ".sidebar nav",
      view: "overview",
      title: "Το μενού",
      body: "Τέσσερις ενότητες: Επισκόπηση (ξεκινάτε εδώ), Προφίλ πελατών, Νέες ευκαιρίες και Τεκμηρίωση. Οι τρεις τελευταίες γεμίζουν μετά την ανάλυση.",
    },
    {
      target: "#dropzone",
      view: "overview",
      title: "1. Ανεβάστε τους πελάτες σας",
      body: "Σύρετε εδώ ένα αρχείο Excel με δύο στήλες: ΑΦΜ και τζίρο. Αν δεν έχετε έτοιμο αρχείο, πατήστε «Λήψη προτύπου» ακριβώς από κάτω.",
    },
    {
      target: "#options-toggle",
      view: "overview",
      developer: true,
      title: "Περισσότερες επιλογές (προαιρετικά)",
      body: "Για προχωρημένους: αλλαγή των βαρών ομοιότητας και έλεγχος στο ΓΕΜΗ αν οι προτάσεις είναι ακόμη ενεργές. Οι προεπιλογές είναι ήδη ελεγμένες, οπότε μπορείτε να το αφήσετε όπως είναι.",
    },
    {
      target: "#analyze-button",
      view: "overview",
      title: "2. Πατήστε «Ανάλυση»",
      body: "Η ανάλυση γίνεται εξ ολοκλήρου στον υπολογιστή σας και διαρκεί λίγα λεπτά. Θα βλέπετε την πρόοδο σε αυτή τη σελίδα.",
    },
    {
      target: "#demo-button",
      view: "overview",
      title: "Θέλετε απλώς να δοκιμάσετε;",
      body: "Η «Δοκιμή με demo δεδομένα» τρέχει την εφαρμογή με ένα δείγμα, χωρίς δικό σας αρχείο.",
    },
    {
      target: ".stats",
      view: "overview",
      title: "Τα βασικά νούμερα",
      body: "Μετά την ανάλυση, εδώ βλέπετε πόσους πελάτες αναλύσαμε, πόσα προφίλ βρέθηκαν και πόσες νέες ευκαιρίες προτείνονται.",
    },
    {
      target: '.nav[data-view="profiles"]',
      view: "profiles",
      title: "3. Προφίλ πελατών",
      body: "Ομάδες πελατών με κοινά, μετρήσιμα χαρακτηριστικά. Κάθε προφίλ έχει περιγραφή στα ελληνικά από τοπικό AI (με σήμανση), μερίδιο τζίρου, σταθερότητα και στατιστική τεκμηρίωση.",
    },
    {
      target: '.nav[data-view="prospects"]',
      view: "prospects",
      title: "4. Νέες ευκαιρίες",
      body: "Επιχειρήσεις που δεν είναι ακόμη πελάτες σας αλλά μοιάζουν με ένα προφίλ, σε χάρτη και πίνακα με σκορ και αιτιολόγηση «Γιατί προτείνεται». Είναι προτάσεις για δική σας αξιολόγηση, όχι εκτίμηση φερεγγυότητας.",
    },
    {
      target: '.nav[data-view="validation"]',
      view: "validation",
      title: "5. Τεκμηρίωση",
      body: "Πόσο καλά δουλεύει η μέθοδος (Recall σε κρυμμένους πελάτες), η ποιότητα των δεδομένων, οι περιορισμοί και η λήψη όλων των αρχείων.",
    },
    {
      target: "#user-button",
      view: "overview",
      title: "Ο λογαριασμός σας",
      body: "Από εδώ μπορείτε να ξαναδείτε την ξενάγηση ή να αποσυνδεθείτε. Δίπλα, το ☾ αλλάζει σε σκοτεινό θέμα.",
    },
    {
      target: "#chat-launcher",
      view: "overview",
      title: "Και κάτι ακόμα: ο βοηθός σας",
      body: "Έχετε απορία; Ρωτήστε τον βοηθό στα ελληνικά. Τρέχει στο ίδιο τοπικό μοντέλο AI και, αν δεν μπορεί να βοηθήσει, σας φέρνει σε επαφή με την υποστήριξη.",
      last: true,
    },
  ];

  const tour = (() => {
    let index = 0,
      root = null,
      spot = null,
      pop = null,
      target = null,
      previousFocus = null;
    const visible = (el) => {
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    };
    const steps = () => STEPS.filter((s) => !s.developer || document.body.classList.contains("is-developer")); // a hidden target (e.g. narrow screen) falls back to a centred card
    function build() {
      root = document.createElement("div");
      root.className = "tour";
      root.innerHTML =
        '<div class="tour-spot" aria-hidden="true"></div>' +
        '<div class="tour-pop" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-body">' +
        '<p class="tour-count" id="tour-count"></p><h2 id="tour-title"></h2><p id="tour-body"></p>' +
        '<div class="tour-dots" id="tour-dots" aria-hidden="true"></div>' +
        '<div class="tour-actions"><button type="button" class="text-button" id="tour-skip">Παράλειψη</button>' +
        '<span><button type="button" class="button outline" id="tour-back">← Πίσω</button>' +
        '<button type="button" class="button tour-next" id="tour-next">Επόμενο →</button></span></div></div>';
      document.body.appendChild(root);
      spot = root.querySelector(".tour-spot");
      pop = root.querySelector(".tour-pop");
      byId("tour-skip").addEventListener("click", () => finish(false));
      byId("tour-back").addEventListener("click", () => show(index - 1));
      byId("tour-next").addEventListener("click", () => show(index + 1));
      root.addEventListener("keydown", (event) => {
        if (event.key === "Escape") finish(false);
        if (event.key === "ArrowRight") show(index + 1);
        if (event.key === "ArrowLeft") show(index - 1);
        if (event.key === "Tab") {
          // Keep keyboard focus inside the dialog.
          const items = [...pop.querySelectorAll("button:not([disabled])")];
          const first = items[0],
            lastItem = items[items.length - 1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            lastItem.focus();
          } else if (!event.shiftKey && document.activeElement === lastItem) {
            event.preventDefault();
            first.focus();
          }
        }
      });
    }
    function place() {
      if (!root) return;
      const margin = 16;
      if (!target || !visible(target)) {
        spot.style.cssText = "top:50%;left:50%;width:0;height:0";
        pop.style.top = Math.max(margin, (innerHeight - pop.offsetHeight) / 2) + "px";
        pop.style.left = Math.max(margin, (innerWidth - pop.offsetWidth) / 2) + "px";
        return;
      }
      const r = target.getBoundingClientRect();
      const pad = 8;
      spot.style.cssText = `top:${r.top - pad}px;left:${r.left - pad}px;width:${r.width + pad * 2}px;height:${r.height + pad * 2}px`;
      const w = pop.offsetWidth,
        h = pop.offsetHeight;
      let top, left;
      if (innerWidth - r.right > w + margin * 2) {
        // Room on the right (e.g. sidebar items).
        left = r.right + pad + margin;
        top = r.top + r.height / 2 - h / 2;
      } else if (innerHeight - r.bottom > h + margin * 2) {
        top = r.bottom + pad + margin;
        left = r.left + r.width / 2 - w / 2;
      } else if (r.top > h + margin * 2) {
        top = r.top - pad - margin - h;
        left = r.left + r.width / 2 - w / 2;
      } else {
        left = r.left - pad - margin - w;
        top = r.top + r.height / 2 - h / 2;
      }
      pop.style.top = Math.min(Math.max(margin, top), innerHeight - h - margin) + "px";
      pop.style.left = Math.min(Math.max(margin, left), innerWidth - w - margin) + "px";
    }
    function show(i) {
      const list = steps();
      if (i < 0) return;
      if (i >= list.length) return finish(true);
      index = i;
      const step = list[i];
      if (step.view) goView(step.view);
      if (step.target === "#chat-launcher") chat.showLauncher();
      target = step.target ? document.querySelector(step.target) : null;
      if (target && !visible(target)) target = null;
      byId("tour-count").textContent = `Βήμα ${i + 1} από ${list.length}`;
      byId("tour-title").textContent = step.title;
      byId("tour-body").textContent = step.body;
      byId("tour-dots").innerHTML = list.map((_, j) => `<i class="${j === i ? "on" : ""}"></i>`).join("");
      byId("tour-back").disabled = i === 0;
      byId("tour-next").textContent = step.last ? "Ας ξεκινήσουμε ✓" : "Επόμενο →";
      root.classList.toggle("tour-centered", !target);
      if (target) target.scrollIntoView({ block: "center", behavior: "instant" });
      place();
      byId("tour-next").focus();
    }
    async function finish(completed) {
      if (!root) return;
      removeEventListener("resize", place);
      removeEventListener("scroll", place, true);
      root.remove();
      root = null;
      goView("overview");
      if (user && !user.tutorial_done) {
        user.tutorial_done = true;
        const form = new FormData();
        form.append("done", "true");
        post("/api/auth/tutorial", form).catch(() => {});
      }
      if (completed) chat.open(true);
      else previousFocus?.focus?.();
    }
    return {
      start() {
        if (root) return;
        previousFocus = document.activeElement;
        chat.close();
        build();
        addEventListener("resize", place);
        addEventListener("scroll", place, true);
        show(0);
      },
    };
  })();

  // ---------------------------------------------------------------- help chat
  const SUGGESTIONS = [
    "Πώς ανεβάζω το πελατολόγιό μου;",
    "Τι σημαίνει «σταθερότητα»;",
    "Πώς διαβάζω τις νέες ευκαιρίες;",
    "Θέλω να μιλήσω με άνθρωπο",
  ];
  const chat = (() => {
    const history = [];
    let panel, list, input, sendButton, launcher, greeted = false, streaming = false, contact = null;
    function build() {
      launcher = document.createElement("button");
      launcher.type = "button";
      launcher.id = "chat-launcher";
      launcher.className = "chat-launcher";
      launcher.setAttribute("aria-controls", "chat-panel");
      launcher.setAttribute("aria-expanded", "false");
      launcher.innerHTML = '<span aria-hidden="true">💬</span><span>Βοηθός</span>';
      launcher.addEventListener("click", () => (panel.classList.contains("hidden") ? open() : close()));
      panel = document.createElement("section");
      panel.id = "chat-panel";
      panel.className = "chat-panel hidden";
      panel.setAttribute("aria-label", "Βοηθός Bizcover");
      panel.innerHTML =
        '<header class="chat-head"><div><strong>Βοηθός Bizcover</strong>' +
        '<small><span class="chat-ai">AI</span> Τοπικό μοντέλο · οι απαντήσεις μπορεί να έχουν λάθη</small></div>' +
        '<button type="button" class="chat-close" aria-label="Κλείσιμο βοηθού">✕</button></header>' +
        '<div class="chat-list" role="log" aria-live="polite"></div>' +
        '<div class="chat-suggestions"></div>' +
        '<form class="chat-form"><label class="sr-only" for="chat-input">Μήνυμα</label>' +
        '<textarea id="chat-input" rows="1" maxlength="1000" placeholder="Γράψτε την ερώτησή σας…"></textarea>' +
        '<button type="submit" class="chat-send" aria-label="Αποστολή">↑</button></form>' +
        '<footer class="chat-foot"><button type="button" class="text-button chat-support">Επικοινωνία με υποστήριξη</button>' +
        "<span>Δεν κρίνει φερεγγυότητα επιχειρήσεων</span></footer>";
      document.body.append(panel, launcher);
      list = panel.querySelector(".chat-list");
      input = panel.querySelector("textarea");
      sendButton = panel.querySelector(".chat-send");
      panel.querySelector(".chat-close").addEventListener("click", close);
      panel.querySelector(".chat-support").addEventListener("click", () => supportCard("Ο χρήστης ζήτησε υποστήριξη"));
      panel.querySelector(".chat-form").addEventListener("submit", (event) => {
        event.preventDefault();
        send(input.value);
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          send(input.value);
        }
      });
      input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
      });
      panel.addEventListener("keydown", (event) => {
        if (event.key === "Escape") close();
      });
      const chips = panel.querySelector(".chat-suggestions");
      SUGGESTIONS.forEach((text) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.textContent = text;
        chip.addEventListener("click", () => send(text));
        chips.appendChild(chip);
      });
    }
    // Minimal, safe formatting of model output: escape first, then **bold**, `code` and *italics*.
    const format = (text) =>
      esc(text)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    function bubble(role, html, label) {
      const item = document.createElement("div");
      item.className = "chat-msg " + role;
      item.innerHTML = (label ? `<span class="chat-label">${esc(label)}</span>` : "") + `<div class="chat-text">${html}</div>`;
      list.appendChild(item);
      list.scrollTop = list.scrollHeight;
      return item;
    }
    function greet(afterTour) {
      if (greeted) return;
      greeted = true;
      // No name in the greeting: Greek names would need the vocative case.
      bubble(
        "bot",
        (afterTour ? "Τέλος η ξενάγηση! " : "Γεια σας! ") +
          "Είμαι ο βοηθός του Bizcover. Ρωτήστε με πώς να χρησιμοποιήσετε την εφαρμογή ή τι σημαίνουν τα αποτελέσματα. Αν χρειαστείτε άνθρωπο, θα σας φέρω σε επαφή με την υποστήριξη.",
        "Βοηθός · σταθερό μήνυμα",
      );
    }
    async function send(raw) {
      const message = String(raw || "").trim();
      if (!message || streaming) return;
      input.value = "";
      input.style.height = "auto";
      panel.querySelector(".chat-suggestions").classList.add("hidden");
      bubble("user", esc(message));
      streaming = true;
      sendButton.disabled = true;
      const reply = bubble("bot", '<span class="chat-typing" aria-label="Ο βοηθός γράφει"><i></i><i></i><i></i></span>', "Βοηθός");
      const textBox = reply.querySelector(".chat-text");
      const label = reply.querySelector(".chat-label");
      let text = "",
        escalate = null;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "X-ForceCRM": "local", "Content-Type": "application/json" },
          body: JSON.stringify({ message, history: history.slice(-6) }),
        });
        if (response.status === 401) return location.replace("/login.html");
        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let newline;
          while ((newline = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            if (!line) continue;
            const event = JSON.parse(line);
            if (event.type === "meta") {
              label.textContent = event.ai
                ? `AI · τοπικό μοντέλο ${event.model || ""}`.trim()
                : "Βοηθός · σταθερή απάντηση";
              if (event.escalate) escalate = event.reason || "Παραπομπή στην υποστήριξη";
            } else if (event.type === "token") {
              text += event.text;
              textBox.innerHTML = format(text);
              list.scrollTop = list.scrollHeight;
            }
          }
        }
      } catch (err) {
        text = text || "Δεν μπόρεσα να απαντήσω αυτή τη στιγμή.";
        textBox.innerHTML = format(text);
        escalate = escalate || "Σφάλμα σύνδεσης με τον βοηθό";
      } finally {
        streaming = false;
        sendButton.disabled = false;
      }
      history.push({ role: "user", content: message }, { role: "assistant", content: text });
      if (escalate) supportCard(escalate);
      input.focus();
    }
    async function supportCard(reason) {
      if (!contact) {
        try {
          contact = (await (await fetch("/api/support")).json()).contact;
        } catch {
          contact = { name: "Υποστήριξη Bizcover", email: "", phone: "", hours: "" };
        }
      }
      const card = bubble(
        "support",
        `<strong>Χρειάζεστε άνθρωπο; Η ομάδα υποστήριξης είναι εδώ.</strong>` +
          `<ul><li>✉ ${esc(contact.email)}</li><li>☎ ${esc(contact.phone)}</li><li>🕘 ${esc(contact.hours)}</li></ul>` +
          `<button type="button" class="button outline chat-ticket">Αποστολή της συνομιλίας στην υποστήριξη</button>` +
          `<small>Το αίτημα αποθηκεύεται τοπικά (demo).</small>`,
        contact.name,
      );
      const button = card.querySelector(".chat-ticket");
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const response = await post("/api/support/tickets", { reason, transcript: history }, true);
          if (!response.ok) throw new Error();
          const { ticket_id } = await response.json();
          button.replaceWith(Object.assign(document.createElement("p"), {
            className: "chat-ticket-ok",
            textContent: `✓ Δημιουργήθηκε αίτημα #${ticket_id}. Θα επικοινωνήσουμε μαζί σας.`,
          }));
        } catch {
          button.disabled = false;
          button.textContent = "Αποτυχία· δοκιμάστε ξανά";
        }
      });
    }
    function open(afterTour = false) {
      showLauncher();
      panel.classList.remove("hidden");
      launcher.setAttribute("aria-expanded", "true");
      launcher.classList.add("open");
      greet(afterTour);
      input.focus();
    }
    function close() {
      if (!panel || panel.classList.contains("hidden")) return;
      panel.classList.add("hidden");
      launcher.setAttribute("aria-expanded", "false");
      launcher.classList.remove("open");
      launcher.focus();
    }
    function showLauncher() {
      launcher.classList.remove("hidden");
    }
    build();
    return { open, close, showLauncher };
  })();

  // ---------------------------------------------------------------- start
  fetch("/api/auth/me")
    .then((r) => {
      if (r.status === 401) {
        location.replace("/login.html");
        return null;
      }
      return r.json();
    })
    .then((me) => {
      if (!me) return;
      user = me.user;
      // Technical details (.dev-only) are shown only to the «Developer» account.
      document.body.classList.toggle("is-developer", user.role === "developer");
      byId("user-name").textContent = user.username;
      menuButton.textContent = initials(user.username);
      menuButton.title = user.username;
      if (!me.auth) byId("logout-button").classList.add("hidden");
      if (me.auth && !user.tutorial_done) tour.start();
    })
    .catch(() => {});
})();
