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
