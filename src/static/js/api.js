/**
 * API Client (`src/static/js/api.js`)
 * Typed fetch wrappers for Archimedes2026 backend endpoints.
 */

export async function fetchBootstrap() {
  const res = await fetch("/api/bootstrap");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function postRun(params) {
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params }),
  });
  return res.json();
}

export async function fetchCompanyDetails(companyId) {
  const res = await fetch(`/api/company/${companyId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function postVerifyCompany(companyId, force = true) {
  const res = await fetch("/api/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ company_id: companyId, force }),
  });
  return res.json();
}

export async function postVerifyBatch(count = 25) {
  const res = await fetch("/api/verify-batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ count }),
  });
  return res.json();
}

export async function postExcludeCompany(companyId, reason) {
  const res = await fetch("/api/exclude", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ company_id: companyId, reason }),
  });
  return res.json();
}

export async function postUnexcludeCompany(companyId) {
  const res = await fetch("/api/unexclude", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ company_id: companyId }),
  });
  return res.json();
}
