/**
 * Non-Blocking 500px Right-Anchored Slide-Over Drawer (`src/static/js/components/drawer.js`)
 * Supports 3 sub-tabs: 'overview' | 'geo' | 'registry'.
 */

export function switchDrawerSubTab(tabKey) {
  document.querySelectorAll("[data-drawer-tab]").forEach((btn) => {
    const active = btn.getAttribute("data-drawer-tab") === tabKey;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });

  document.querySelectorAll("[data-drawer-pane]").forEach((pane) => {
    const active = pane.getAttribute("data-drawer-pane") === tabKey;
    pane.classList.toggle("active", active);
  });
}

