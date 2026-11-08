# Project: Archimedes2026 Production Refactoring and Scaling

## Architecture
Archimedes2026 is an enterprise B2B Ideal Customer Profile (ICP) Discovery and Lookalike Recommendation Engine tailored for the Greek business ecosystem.
The system comprises:
1. **Data Ingestion Layer** (`src/ingest.py`): Parses company registry, kad codes, geographical coordinates, and client data.
2. **Feature Store Layer** (`src/features/`): Computes behavioral expansion velocity, cyclical seasonality, brand heuristics, municipality typologies, multi-tier spatial trees, and algebraic/neural KAD representations.
3. **Mathematical Engine** (`src/icp_engine.py` / `src/config.py`): Discovers ICP clusters via weighted spherical K-Means, computes empirical Bayes categorical log-lifts, spatial Medoid geodesics, and hybrid ensemble cosine scores.
4. **Backend Service & API Layer** (`src/server.py`, `src/service.py`, `src/verifier.py`, `app.py`): High-performance Starlette ASGI application, asynchronous background worker threads, automated verification cache, and JSON serialization.
5. **Frontend Cockpit** (`src/static/`): Dark executive single-page dashboard with interactive Leaflet map, Chart.js analytics, dynamic filters, inspection drawer, and 100% offline vendored assets.
6. **Evaluation & Verification Harness** (`evaluate_holdout.py`, `tests/`): Automated regression test suite and 100-fold cross-validation benchmarking against holdout client subsets.

---

## Baseline Deliverables Snapshot
- `output/profiles.json`: SHA-256 `c27312cb2bd7d449144eae6ae836ab01a50e9932673ddb01e846f20cb6ccde1f` (369 lines, 2 ICP profiles)
- `output/assignments.csv`: SHA-256 `049af7fad63302ac6e2cc666d1c4419207a951c6dcab31634d2042c054ca4baa` (301 lines: 1 header + 300 client companies)
- `output/lookalikes.csv`: SHA-256 `0c5ac09a30c9aa9c562636b9a5262335bfe000be42fa3932f47b55650438aaa8` (2001 lines: 1 header + 2000 lookalike candidates)

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Baseline Deliverables Snapshot | Record SHA-256 digests and row counts of profiles.json, assignments.csv, lookalikes.csv | M1 | R1 |
| 2 | Automated Pytest Suite Config | Add pytest>=8.0.0 and httpx>=0.27.0 to pyproject.toml with testpaths and addopts | M1 | R1 |
| 3 | Ingest Unit Tests (`test_ingest.py`) | Test VAT zero-padding (9 digits), KAD canonical primary selection, and client VAT branch isolation | M1 | R1 |
| 4 | Engine Unit Tests (`test_engine.py`) | Test weighted K-Means determinism, silhouette calculation, K=1 edge cases, and Bootstrap ARI bounds [0, 1] | M1 | R1 |
| 5 | Scoring Unit Tests (`test_scoring.py`) | Test similarity scores in [0, 1], NO_FIT thresholding, and client branch exclusion | M1 | R1 |
| 6 | API Unit Tests (`test_api.py`) | Test /api/health, /api/bootstrap, /api/company/1, and /api/run via starlette TestClient | M1 | R1 |
| 7 | Ingestion Feature Extraction Relocation | Move _compute_post_launch_expansion, _compute_seasonality_features, _compute_brand_features, _compute_municipality_typology to src/features/engineering.py | M2 | R3 |
| 8 | Dynamic Ingestion Size Checks | Remove hardcoded size assertions (24424 companies, 300 client VATs) in src/ingest.py | M2 | R3 |
| 9 | Fast Contiguous 2D NumPy Feature Store | Replace slow [r.tolist()] / np.vstack with flat 2D arrays (.npz) with Parquet fallback | M2 | R3 |
| 10 | Transit Hub Coordinates Externalization | Move GREEK_TRANSIT_FOOT_TRAFFIC_HUBS (38+ tuples) to data/greek_transit_hubs.json | M2 | R3 |
| 11 | Scoring Kernel Unification | Unify score_companies_against_icps and empirical Bayes log-lifts between icp_engine and evaluate_holdout | M3 | R2 |
| 12 | Scoring Kernel Signature Cleanup | Remove unused top_reasons_per_company_icp empty list from score_companies_against_icps | M3 | R2 |
| 13 | Trait Weights Externalization | Move 14 hardcoded trait weights into PipelineConfig.trait_weights | M3 | R2 |
| 14 | Portable Parquet/JSON Cache | Replace pickle.dump/load in save/load_sha256_cached_result with Parquet and JSON | M3 | R2 |
| 15 | Evaluation Harness Dataclass Replace | Replace manual dictionary reconstruction in make_icp_engine_ranker with dataclasses.replace | M3 | R2 |
| 16 | Non-blocking ASGI Handlers | Wrap synchronous CPU operations (run, company, export) and network calls in asyncio.to_thread | M4 | R4 |
| 17 | Remove Dead Streamlit Endpoints | Remove /_stcore/health and /_stcore/host-config; implement dedicated /api/health handler | M4 | R4 |
| 18 | Dynamic Virtualenv Resolution | Fix hardcoded python3.12 site-packages path in app.py and src/server.py | M4 | R4 |
| 19 | Batch Company Exclusions | Collect all closed VATs in verify_top_leads, save once, and run single pipeline recomputation | M4 | R4 |
| 20 | High-performance JSON Serialization | Replace recursive _clean() with OrjsonResponse / Polars native JSON | M4 | R4 |
| 21 | Verifier Service Optimization | Remove dummy request to businessportal.gr, batch cache writes, and use in-memory IngestedDataset | M4 | R4 |
| 22 | Frontend ES Module Modularization | Decompose src/static/app.js into 7 ES modules (state, api, map, charts, tables, drawer, app) | M5 | R5 |
| 23 | Offline Vendor Bundling | Bundle Leaflet, Chart.js, Lucide, Tailwind locally in src/static/vendor/ for 100% offline UI | M5 | R5 |
| 24 | High-Density Theme & Interactive Binding | Retain dark executive theme; bind interactive handlers to window for inline onclick attributes | M5 | R5 |
| 25 | Packaging CLI Entry Point Fix | Fix archimedes2026 entry point in pyproject.toml to point to app:main | M6 | R6 |
| 26 | Multi-Version CI/CD Workflow | Add .github/workflows/ci.yml running linting and tests across Python 3.12 and 3.13 | M6 | R6 |
| 27 | Deliverables Generation Validation | Verify profiles.json, assignments.csv, lookalikes.csv generate cleanly matching schema and baseline | M6 | R6 |
| 28 | Holdout Benchmark Target Verification | Run evaluate_holdout.py and ensure Recall@500 >= 14.83% and Recall@2000 >= 31.87% | M6 | R6 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Test Safety Harness & Baseline Snapshot | Features 1-6 (tests/, pyproject.toml) | none | DONE (50/50 tests passing; gate passed) |
| M2 | Feature Store & Ingest Pipeline Modularization | Features 7-10 (src/features/, src/ingest.py, data/greek_transit_hubs.json) | M1 | DONE (71/71 tests passing; gate passed) |
| M3 | Mathematical Engine & Scoring Unification | Features 11-15 (src/icp_engine.py, evaluate_holdout.py, src/config.py) | M1, M2 | IN_PROGRESS (Gate verification) |
| M4 | Backend API, Service Layer & Verification | Features 16-21 (src/server.py, src/service.py, src/verifier.py, app.py) | M1, M2, M3 | PLANNED |
| M5 | Frontend Modularization & Offline UI | Features 22-24 (src/static/js/, src/static/vendor/, index.html) | M1, M4 | PLANNED |
| M6 | Integration, Release Tooling, CI/CD & Final Verification | Features 25-28 (pyproject.toml, .github/workflows/ci.yml, deliverables, evaluate_holdout.py) | M1, M2, M3, M4, M5 | PLANNED |

---

## Interface Contracts

### M1 ↔ Codebase
- `pytest tests/`: All tests in `test_ingest.py`, `test_engine.py`, `test_scoring.py`, and `test_api.py` must pass with exit code 0.
- `starlette.testclient.TestClient(app)`: Responses for `/api/health`, `/api/bootstrap`, `/api/company/1` (mocked or valid fallback), and `/api/run` must return HTTP 200.

### M2 ↔ M3 / M4 (`src/features/` & `src/ingest.py`)
- `compute_post_launch_expansion(companies, company_activities) -> pl.DataFrame`
- `compute_seasonality_features(companies) -> pl.DataFrame`
- `compute_brand_features(companies) -> pl.DataFrame`
- `compute_municipality_typology(companies) -> pl.DataFrame`
- `load_greek_transit_hubs(path) -> list[tuple[float, float, str]]`
- `save_feature_store_npz(fs, path, model_col) -> None`
- `load_feature_store_npz(path, model_col) -> dict[str, Any] | None`

### M3 ↔ M4 / Benchmark (`src/icp_engine.py` & `evaluate_holdout.py`)
- `score_companies_against_icps(ec_sorted, fs, profiles, client_latlon_rad, client_weights, cfg) -> np.ndarray` (shape: `(N, K)`, values in `[0.0, 1.0]`)
- `PipelineConfig.trait_weights: tuple[tuple[str, float], ...]`
- `save_sha256_cached_result(result, cfg) -> Path` (writes Parquet + JSON)
- `load_sha256_cached_result(clients_sha256, cfg) -> ICPEngineResult | None`
- `make_icp_engine_ranker(fs, cfg, fast_cv_mode) -> RankerFn` (uses `dataclasses.replace(cfg, ...)`)

### M4 ↔ M5 (`src/server.py` ↔ Frontend)
- `GET /api/health` -> `{"status": "ok"}`
- `GET /api/bootstrap` -> `{"status": "ok", "generation": int, "data": {...}}`
- `POST /api/run` -> `{"status": "ok", "generation": int, "data": {...}}`
- `GET /api/company/{company_id}` -> company JSON object
- `POST /api/leads/verify-top` -> `{"status": "ok", "checked": int, "closed_detected": int, "closed_companies": [...], "payload": ...}`

---

## Code Layout
```
Archimedes2026/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── greek_transit_hubs.json
│   ├── vwHackCompany.csv
│   ├── vwHackCompanyActivity.csv
│   ├── ForceCRM_Hackathon_Pelatologio.xlsx
│   ├── cached_universe_features.parquet
│   ├── optimal_weights.json
│   └── ...
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── icp_engine.py
│   ├── service.py
│   ├── server.py
│   ├── verifier.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── store.py
│   │   ├── engineering.py
│   │   ├── embeddings.py
│   │   ├── spatial.py
│   │   ├── enrichment.py
│   │   ├── structured.py
│   │   ├── transit.py
│   │   └── text.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       ├── logo.svg
│       ├── vendor/
│       │   ├── leaflet/
│       │   ├── chartjs/
│       │   ├── lucide/
│       │   └── tailwind/
│       └── js/
│           ├── state.js
│           ├── api.js
│           ├── map.js
│           ├── charts.js
│           ├── tables.js
│           ├── drawer.js
│           └── app.js
├── tests/
│   ├── test_ingest.py
│   ├── test_engine.py
│   ├── test_scoring.py
│   └── test_api.py
├── output/
│   ├── profiles.json
│   ├── assignments.csv
│   ├── lookalikes.csv
│   └── cache/
├── app.py
├── evaluate_holdout.py
├── pyproject.toml
└── PROJECT.md
```
