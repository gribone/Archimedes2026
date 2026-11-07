"use strict";
// Applies the saved theme before first paint; with no saved choice the OS setting wins.
(function () {
  const KEY = "forcecrm-theme";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  let saved = null;
  try {
    saved = localStorage.getItem(KEY);
  } catch (e) {}
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  const current = () => root.dataset.theme || (media.matches ? "dark" : "light");
  const announce = () =>
    document.dispatchEvent(new CustomEvent("themechange", { detail: current() }));
  media.addEventListener("change", () => {
    if (!root.dataset.theme) announce();
  });
  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const sync = () => {
      const dark = current() === "dark";
      button.textContent = dark ? "☀" : "☾";
      button.setAttribute("aria-label", dark ? "Φωτεινό θέμα" : "Σκοτεινό θέμα");
      button.title = button.getAttribute("aria-label");
    };
    sync();
    document.addEventListener("themechange", sync);
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem(KEY, next);
      } catch (e) {}
      announce();
    });
  });
})();
