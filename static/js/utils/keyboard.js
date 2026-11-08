/**
 * Greek Keyboard Focus Shield & Global Shortcut Controller (`src/static/js/utils/keyboard.js`)
 * Prevents single-key shortcuts from intercepting Greek/English text input while
 * supporting J/K, ArrowUp/ArrowDown, Enter, Esc, Ctrl+K, M, T, E, D.
 */

export function isTypingInInput(e) {
  const tag = e.target?.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    e.target?.isContentEditable
  );
}

export function registerKeyboardShortcuts(handlers) {
  window.addEventListener("keydown", (e) => {
    // 1. Ctrl+K / Cmd+K always focuses search bar
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      handlers.onFocusSearch?.();
      return;
    }

    // 2. Escape closes active modal/drawer or blurs input
    if (e.key === "Escape") {
      if (isTypingInInput(e)) {
        e.target.blur();
      }
      handlers.onEscape?.(e);
      return;
    }

    // 3. Shield all single-key shortcuts when typing in an input
    if (isTypingInInput(e)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const key = e.key;
    if (key === "ArrowDown" || key === "j" || key === "J" || key === "ξ" || key === "Ξ") {
      e.preventDefault();
      handlers.onNextRow?.();
    } else if (key === "ArrowUp" || key === "k" || key === "K" || key === "κ" || key === "Κ") {
      e.preventDefault();
      handlers.onPrevRow?.();
    } else if (key === "Enter") {
      handlers.onOpenSelectedRow?.();
    } else if (key === "m" || key === "M" || key === "μ" || key === "Μ") {
      handlers.onToggleMapFocus?.();
    } else if (key === "t" || key === "T" || key === "τ" || key === "Τ") {
      handlers.onToggleTableFocus?.();
    } else if (key === "d" || key === "D" || key === "δ" || key === "Δ") {
      handlers.onCycleDensity?.();
    } else if (key === "e" || key === "E" || key === "ε" || key === "Ε") {
      handlers.onExcludeActive?.();
    }
  });
}

