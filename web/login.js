"use strict";
(function () {
  const form = document.getElementById("login-form");
  const errorBox = document.getElementById("login-error");
  const submit = document.getElementById("login-submit");
  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.classList.toggle("hidden", !message);
  };
  // Already signed in (or login turned off): go straight to the app.
  fetch("/api/auth/me")
    .then((r) => (r.ok ? r.json() : null))
    .then((me) => {
      if (me) location.replace("/");
    })
    .catch(() => {});
  // One-click demo sign-in (no account): shown only when the server enables it.
  const demoBox = document.getElementById("demo-login");
  const demoButtons = [document.getElementById("demo-login-button"), document.getElementById("developer-login-button")];
  fetch("/api/auth/demo")
    .then((r) => (r.ok ? r.json() : null))
    .then((info) => {
      if (info && info.enabled) demoBox.classList.remove("hidden");
    })
    .catch(() => {});
  // One-click accounts: «Επίδειξη» (salesperson's view) or «Developer» (also the technical details).
  demoButtons.forEach((button) =>
    button.addEventListener("click", async () => {
      showError("");
      demoButtons.forEach((b) => (b.disabled = true));
      try {
        const body = new FormData();
        body.set("role", button.dataset.role);
        const response = await fetch("/api/auth/demo", { method: "POST", headers: { "X-ForceCRM": "local" }, body });
        const reply = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(reply.detail || `Σφάλμα διακομιστή (${response.status}).`);
        location.replace("/");
      } catch (err) {
        showError(err.message);
        demoButtons.forEach((b) => (b.disabled = false));
      }
    }),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = form.username.value.trim();
    const password = form.password.value;
    if (!username || !password) {
      showError("Συμπληρώστε όνομα χρήστη και κωδικό.");
      (username ? form.password : form.username).focus();
      return;
    }
    showError("");
    submit.disabled = true;
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "X-ForceCRM": "local" },
        body: new FormData(form),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Σφάλμα διακομιστή (${response.status}).`);
      location.replace("/");
    } catch (err) {
      showError(err.message);
      form.password.select();
      submit.disabled = false;
    }
  });
})();
