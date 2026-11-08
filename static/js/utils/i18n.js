/**
 * Bilingual i18n Utility (`src/static/js/utils/i18n.js`)
 * Supports instantaneous EL <-> EN switching with Intl formatting and CLDR plurals.
 */

let currentDictionary = {};
let fallbackDictionary = {};
let currentLang = localStorage.getItem("archimedes_lang") || "el";

const pluralRulesCache = {
  en: new Intl.PluralRules("en-US"),
  el: new Intl.PluralRules("el-GR"),
};

export const formatters = {
  currency: (val, lang = currentLang) =>
    new Intl.NumberFormat(lang === "el" ? "el-GR" : "en-US", {
      style: "currency",
      currency: "EUR",
      maximumFractionDigits: 0,
    }).format(val || 0),

  compactCurrency: (val, lang = currentLang) =>
    new Intl.NumberFormat(lang === "el" ? "el-GR" : "en-US", {
      style: "currency",
      currency: "EUR",
      notation: "compact",
      maximumFractionDigits: 2,
    }).format(val || 0),

  percent: (val, decimals = 1, lang = currentLang) =>
    new Intl.NumberFormat(lang === "el" ? "el-GR" : "en-US", {
      style: "percent",
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(val || 0),

  integer: (val, lang = currentLang) =>
    new Intl.NumberFormat(lang === "el" ? "el-GR" : "en-US").format(Math.round(val || 0)),
};

export async function initI18n(lang = currentLang) {
  currentLang = lang === "en" ? "en" : "el";
  localStorage.setItem("archimedes_lang", currentLang);
  document.documentElement.lang = currentLang;

  try {
    if (Object.keys(fallbackDictionary).length === 0) {
      const enRes = await fetch("/static/locales/en.json");
      if (enRes.ok) fallbackDictionary = await enRes.json();
    }
    if (currentLang === "en") {
      currentDictionary = fallbackDictionary;
    } else {
      const elRes = await fetch(`/static/locales/${currentLang}.json`);
      currentDictionary = elRes.ok ? await elRes.json() : fallbackDictionary;
    }
  } catch (err) {
    console.warn("i18n dictionary load warning:", err);
  }

  applyTranslations();
  return currentLang;
}

export function getLang() {
  return currentLang;
}

export function t(keyPath, params = {}) {
  const keys = keyPath.split(".");
  let val = keys.reduce((obj, k) => obj?.[k], currentDictionary);

  if (val === undefined) {
    val = keys.reduce((obj, k) => obj?.[k], fallbackDictionary) ?? keyPath;
  }

  if (typeof val !== "string") return keyPath;

  return val.replace(/\{(\w+)\}/g, (_, k) =>
    params[k] !== undefined ? params[k] : `{${k}}`
  );
}

export function tp(keyPath, count, params = {}) {
  const pr = pluralRulesCache[currentLang] || pluralRulesCache.en;
  const category = pr.select(count);
  return t(`${keyPath}.${category}`, { count: formatters.integer(count), ...params });
}

export function applyTranslations(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    const params = el.dataset.i18nParams ? JSON.parse(el.dataset.i18nParams) : {};
    el.textContent = t(key, params);
  });

  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.getAttribute("data-i18n-placeholder"));
  });

  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.getAttribute("data-i18n-title"));
  });

  root.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
    el.setAttribute("aria-label", t(el.getAttribute("data-i18n-aria-label")));
  });
}

