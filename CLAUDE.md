# ForceCRM Hackathon 2026 — Challenge Context

Source docs (Greek): `../ForceCRM_Hackathon_2026_Technical_Brief.pdf` (authoritative); pitch deck `../ForceCRM_Hackathon_2026.pptx`. Both are outside the repo, in `/Users/gribgen/workspace/FORCECRM/`.
Repo: github.com/gribone/Archimedes2026. Original data lives outside the repo in `../Data/` (never commit it).

## Current project state (as of 2026-11-07)

### Branches (GitHub default = `main`)
| Branch | What it is |
|---|---|
| `main` | **The submission.** = `feature/forcecrm-local-ai` + setup/docs fix commit `95878d7`. FastAPI + plain JS UI, `forcecrm/` package. |
| `feature/forcecrm-local-ai` | Original local-ai solution (`661cfc2`), without the fixes. Keep, don't merge. |
| `feat/icp-engine-and-ui` | Teammate's alternative (Streamlit, `src/`, uv). Weaker; **not to be merged**. Teammate still pushes UI changes to it. |
| `feat/stage1-ingest-baseline` | First commit of the ICP branch only (ingest + baseline). History only. |
The user wants exactly these 4 branches on GitHub; delete any temporary branches after merging.

### Running `main`
- Needs **Python 3.12** (3.13 resolves; 3.11/3.14 fail on pinned wheels; `run.py` checks this). This Mac's `python3` is 3.14 → use `python3.12` or `$(uv python find 3.12)`.
- Copy the 3 CSVs (`../Data/data/vwHack*.csv`) into the **repo root** next to `run.py`; the Excel is uploaded in the browser (or passed to the CLI). `*.csv`/`*.xlsx` are gitignored.
- `python3.12 run.py --first-run` → creates `.venv`, installs deps, downloads `intfloat/multilingual-e5-small` to `models/e5-small`, `ollama pull ministral-3:3b`, serves http://127.0.0.1:8000. Verified on a clean clone on this Mac.
- CLI: `.venv/bin/python -m scripts.analyze --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --output outputs/real`; holdout audit: `.venv/bin/python -m scripts.audit_ranking`; tests: `.venv/bin/python -m pytest -q` (18 pass).
- Ollama installed via Homebrew (`ollama serve`); model `ministral-3:3b` pulled. Mac has only 8 GB RAM.
- scipy 1.15.3's macOS wheel fails to load on this macOS (dyld `__thread_bss` error) → pinned scipy 1.18.1 / numpy 2.5.4.

### Measured results (reproduced, not just claimed)
| | Recall@500 | Recall@2000 |
|---|---:|---:|
| `main`, mean of 5 splits | 18.0% | 50.3% |
| `main`, default seeded split | 15/60 (25.0%) | 33/60 (55.0%) |
| `feat/icp-engine-and-ui`, 5-fold | 14.0% | 29.7% |
| Primary-ΚΑΔ baseline | ~1.3–1.7% | ~7–8% |
`main` finds 3 profiles (tourist municipalities incl. Heraklion/Cyclades; Kassandra-Halkidiki + Thessaloniki; Attica/ΙΚΕ), ~16 s per Greek description on this Mac.

### Open problems on `main` (priority order)
1. PDF model card (`docs/ForceCRM-Model-Data-Card.pdf`) still says 13/60; `scripts/build_card.py` only finds Windows/Linux fonts, so it can't rebuild on macOS. The Markdown card is fixed.
2. ~~Greek descriptions~~ — fixed 2026-11-07 in `28f8f1b`: `narrative.py` now renders each fact as a Greek template, Ministral rewrites one sentence at a time (chat + few-shot), deterministic checks (numbers, %, names, banned wording/grammar patterns) reject bad rewrites, failures fall back to the template. Typically 3–5 of 6 sentences are LLM-written; ~30–50 s/profile on this Mac. Still no human fluency rating; a stronger Greek LLM (Krikri-8B, EuroLLM-9B) remains an option.
3. ~~Stability/confidence not in text~~ — fixed: every description ends with a computed quality/stability(mean+p10)/confidence sentence.
4. Profiles are almost purely geographic; ΚΑΔ/activity semantics barely appear (A1 risk for judging).
5. Untested: browser flow (upload → profiles → export), Windows/Linux clean install, human fluency ratings.
6. ~~AI-tools declaration~~ — fixed: README and Markdown model card list Claude Code too (PDF card still stale, see 1).
7. Known limits the README admits: uncalibrated confidence; exploratory stats (same data clusters and tests); LLM timings in docs are from the original author's Windows laptop.

### Local leftovers
- `data/` in the repo = symlinks to `../Data` created for the ICP branch (untracked on `main`; safe to delete).
- `.venv/` in the repo is the ICP branch's uv env (1.5 GB); `main`'s `run.py` would reuse the `.venv` path, so delete it before running `main` here.
- Two git worktrees (`local-ai`, `stage1`) are registered in a temporary scratchpad from the review session; once that folder is gone, run `git worktree prune`.

## Goal
"Who will be the next customer?" — a localhost web app (any stack) for Greek food-service businesses (ΚΑΔ 56).
User uploads a B2B company's customer list (Excel: ΑΦΜ + revenue). The app:
1. Discovers Ideal Customer Profiles (ICPs) — groups of customers with shared, measurable traits.
2. Describes each ICP in Greek with a **local** LLM.
3. Ranks **lookalikes**: universe businesses that are not customers but resemble an ICP, with score + reason.

Terms: *Universe* = all businesses in the dataset. *Lookalike* = non-customer similar to an ICP.

## Hard constraints
- **All AI runs locally** on a 16 GB RAM laptop (CPU or consumer GPU): LLM 3–9B in 4-bit (Ollama / llama.cpp), embeddings ≤ ~600M params (sentence-transformers).
- **No external AI API/service at runtime.** External *data* sources are allowed (declare origin + license; Google Places only with own key within free credits).
- European or open-source models preferred.
- AI coding assistants allowed but must be declared (tool + purpose).
- Runs with **one command** in a clean environment, with install instructions.

## Data (`../Data/`, relative to repo root)
CSV: UTF-8 (with BOM), separator `;`, **decimal comma**, empty values = `NULL`.
All active businesses with main ΚΑΔ 56, incorporated 2020–2024: 24,424 records, 96% with GPS.

`../Data/data/vwHackCompany.csv` — one row per shop (24,424)
- `company_id` (join key to activity), `vat` (ΑΦΜ, string with leading zeros; join key to customer list)
- `company_name_el` (sole proprietorships without trade name: «Επιχείρηση Εστίασης»)
- `legal_type` (ΑΤΟΜΙΚΗ, ΕΕ, ΙΚΕ, ΟΕ, ΑΕ, ΕΠΕ …), `is_branch` (1 = branch)
- `street`, `street_number`, `zip_code`, `city`, `municipality_name`, `prefecture_name`
- `incorporation_date` (for branches: start of operation), `latitude`, `longitude`
- **Branches:** one ΑΦΜ can map to several rows (HQ + branches), each with its own `company_id`, address, GPS.

`../Data/data/vwHackCompanyActivity.csv` — 161,964 rows
- `company_id`, `activity_id` (8-digit ΚΑΔ 2025 / NACE 2.1, no dots), `activity_type` (Κύρια = primary / Δευτερεύουσα = secondary / Βοηθητική = auxiliary), `date_from`

`../Data/data/vwHackActivities.csv` — 3,737 rows: `activity_id`, `description` (Greek)

`../Data/ForceCRM_Hackathon_Pelatologio.xlsx` — 300 customers, columns `ΑΦΜ` (text), `Τζίρος (€)` (indicative revenue).
Each ΑΦΜ maps to one row of vwHackCompany. **20% of the real customers were withheld** for evaluation.

⚠️ ΚΑΔ codes are self-declared: many secondary codes don't reflect real activity. The solution must handle this noise.

Sources/licenses (must be cited in any presentation): ΓΕΜΗ Open Data; GPS geocoded with © OpenStreetMap contributors (ODbL). Data is for the competition only.

## System requirements
- **A1 Representation:** combine activity, geography, legal type, age. Weight primary vs secondary activity differently. Use **embeddings of ΚΑΔ descriptions**, not just codes.
- **A2 ICP discovery:** 1–6 profiles; may answer "one ICP" if the list is homogeneous. Document how revenue is used (weighting, high-value selection, etc.).
- **A3 Consistency:** same logic/metric for forming profiles and assigning businesses. Every assignment has a confidence score; "fits no profile" is a valid outcome. Businesses with many ΚΑΔ must not be penalised by the metric.
- **A4 Profile evidence:** defining traits shown over-represented statistically — Fisher exact / hypergeometric **with multiple-testing correction**, or shrinkage. Rare traits must not pass by chance.
- **A5 Stability:** bootstrap or small parameter perturbations (e.g. Adjusted Rand Index); report stability **per profile**.
- **A6 Local LLM description:** Greek text built only from profile data, every claim maps to a measurable fact; labelled as AI-generated; shows profile quality and confidence; reproducible (fixed generation params).
- **A7 Lookalikes:** per profile, rank non-customers with score + short justification.
- **A8 Web app:** localhost; upload Excel; view profiles (description + stats) and lookalikes; export output files.

## Models (indicative; any local model OK)
LLMs: EuroLLM-9B-Instruct (EU, Apache 2.0, Greek), Salamandra-7B-Instruct (BSC, Apache 2.0), Apertus-8B-Instruct (CH, Apache 2.0), Ministral-3-3B-Instruct (Mistral, Apache 2.0), Llama-Krikri-8B-Instruct (GR, ILSP/Athena, Llama 3.1 License, Greek-specialised), Ministral-8B-Instruct-2410 (Mistral Research License — non-commercial), Qwen3 4B/8B (CN, Apache 2.0), Gemma 3 4B/12B (US, Gemma Terms).
Embeddings: jina-embeddings-v5-text-nano (DE, verify license + Greek), granite-embedding multilingual r2 97M/311M (IBM, Apache 2.0), Qwen3-Embedding-0.6B (Apache 2.0), BGE-M3 (MIT, dense+sparse), multilingual-e5 base/large (MIT, baseline), EmbeddingGemma-300M (Gemma Terms).

Justify every model used on four axes:
1. Greek quality **measured on challenge data**, compared with ≥2 alternatives.
2. License — commercial use? Check the specific repo's license file, not the model family's.
3. Origin / digital sovereignty — who built it, where, with what data.
4. Runtime cost — memory and time per business / per profile on the laptop.

## Evaluation
**Hold-out test:** app runs on the given list → lookalike ranking over the universe → **Recall@500 and Recall@2000** of the hidden 20% customers. `lookalikes.csv` must contain **≥ 2,000 suggestions in total**.
**Baseline to beat:** businesses with the same main ΚΑΔ as customers, ordered by that ΚΑΔ's frequency in the customer list.

Judging criteria: completeness (end-to-end, local, covers A1–A8); model choice (4 axes, with measurements); hold-out Recall@k vs baseline; understandability (can a salesperson understand and trust it); responsible AI (AI-text labelling, groundedness, model/data card, AI Act self-assessment); presentation (clarity, demo, business value); extra points.

## Extra points
- **X1 Geographic ICP:** profiles based on shop location (GPS).
- **X2 Per-shop enrichment:** match to Google Places or OpenStreetMap by name/address/GPS; report match rate and how ambiguous matches were handled.
- **X3 Inventiveness:** other open data (e.g. data.gov.gr) that makes ICPs smarter — insight valued over number of sources.

## EU AI Act (key points)
- B2B business clustering is **not** high-risk (Annex III).
- **Trap:** creditworthiness scoring of natural persons is high-risk, and sole proprietorships (ΑΤΟΜΙΚΗ) are natural persons → the system must **never produce a solvency/credit score**.
- Transparency (Art. 50, applies from 2 Aug 2026): AI-generated content must be labelled.
- Annex III high-risk obligations postponed to 2 Dec 2027 (Reg. (EU) 2026/1744, Digital Omnibus on AI).
- AI literacy (Art. 4): builders and users must understand what it does and where it fails.
- Self-assessment answers: What does the system decide/suggest? Who is affected? What does the user see about how results arose? Where can it err and how do we detect it?

## Deliverables
- Code repo with install + **one-command** run instructions.
- Localhost web app (A8).
- Output files in fixed format (auto-graded):
  - `profiles.json` — profile_id, title, AI description, customer count, revenue share, stability, confidence, traits with lift and p-value
  - `assignments.csv` — company_id, profile_id, score
  - `lookalikes.csv` — profile_id, company_id, rank, score, justification
- Model & data card (1–2 pages): models, rationale, data + external sources with licenses, limitations, known errors.
- AI Act self-assessment (1 page).
- Declaration of AI tools used in development.
- 5-minute presentation with live demo.
