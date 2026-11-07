(() => {
  const system = matchMedia("(prefers-color-scheme: dark)");
  let choice;
  try { choice = localStorage.getItem("forcecrm-theme"); } catch {}
  function apply() {
    const dark = choice === "dark" || (choice !== "light" && system.matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.getElementById("theme-toggle")?.setAttribute("aria-pressed", String(dark));
    document.dispatchEvent(new Event("themechange"));
  }
  apply();
  system.addEventListener("change", apply);
  document.addEventListener("DOMContentLoaded", () => {
    apply();
    document.getElementById("theme-toggle").addEventListener("click", () => {
      choice = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("forcecrm-theme", choice); } catch {}
      apply();
    });
  });
})();
