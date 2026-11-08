# ForceCRM Hackathon 2026 — Challenge Context

Source docs (Greek): `../ForceCRM_Hackathon_2026_Technical_Brief.pdf` (authoritative); pitch deck `../ForceCRM_Hackathon_2026.pptx`. Both are outside the repo, in `/Users/gribgen/workspace/FORCECRM/`.
Repo: github.com/gribone/Archimedes2026. Original data lives outside the repo in `../Data/` (never commit it).

## Current project state (as of 2026-11-08)

### Branches (GitHub default = `main`)
| Branch | What it is |
|---|---|
| `main` | **The submission.** FastAPI + plain-JS UI, `forcecrm/` package. = `feature/forcecrm-local-ai` + later commits: setup/docs fix `95878d7`, Greek narration rework `28f8f1b`, PDF card rebuild `e9218ef`, architecture doc `f027c1b`, portfolio traits / activity mix `dd18c72`, task board `030a3ca`, the merge of Dimitris's `feature/ui-titles-live-gemi` (grounded AI titles, weights panel, opt-in live ΓΕΜΗ check, offline Greece map, dark mode), compliance fixes `3a5278a`, then Dimitris's stacked task branches fast-forwarded on 2026-11-08: T2 SQLite store `fc760ba`, T3 stage pipeline `f28c260`, UI collapsibles `3e55d80`, T16 demo login + guided tour + local help chat `33fdebd`. All merged branches were deleted from GitHub. `git log --oneline` is authoritative. |
| `feature/forcecrm-local-ai` | Original local-ai solution (`661cfc2`), without the fixes, plus Stratos's dark-mode commit `e780450`, which duplicates the dark mode already merged into `main`. Keep, don't merge. |
| `feat/icp-engine-and-ui` | Teammate's alternative (Streamlit, `src/`, uv). Weaker; **not to be merged**. Teammate still pushes UI changes to it. |
| `feat/stage1-ingest-baseline` | First commit of the ICP branch only (ingest + baseline). History only. |
The user wants exactly these 4 branches on GitHub; delete any temporary branches after merging.

### Start here
- **App name and brand:** the web app is **Bizcover** (user, 2026-11-08). Assets are in `web/brand/` (icon, logo, icon + logo). Colours are the SVGs' codes, set as CSS tokens in `web/style.css`: blue `#2563EB`, dark blue `#1E3A8A`, sky `#38BDF8`, navy `#0F172A`, rose `#E11D48` / `#FB7185`. The wordmark has white «cover», so the full logo only sits on a blue background; elsewhere it is HTML text. The `X-ForceCRM` request header is an internal security header, so keep it. «ForceCRM» stays only where it names the challenge.
- **Work queue:** [`TODO.md`](TODO.md), the shared multi-agent task board. Claim a task there before starting, and follow its lanes and "done" checklist.
- **How the code works:** `docs/ARCHITECTURE.md`. It covers the pipeline diagram, the feature weights, the ranking formula, why profiles are geographic (§3) and determinism (§4).
- **Other docs:** user-facing run instructions and the methodology are in `README.md`. Model choice and measurements are in `docs/MODEL_DATA_CARD.md` and the PDF `docs/ForceCRM-Model-Data-Card.pdf`. Requirement-by-requirement verification is in `docs/COMPETITION_AUDIT.md`, and the AI Act self-assessment in `docs/RESPONSIBLE_AI.md`.
- **Hard rules:**
  - The LLM never affects profiles, scores or ranking; it only writes description text.
  - Never produce anything resembling a credit or solvency score (AI Act trap, see below).
  - The only outbound runtime call is the **opt-in** live ΓΕΜΗ check (`forcecrm/live.py`): prospect ΑΦΜ only, the user's own key. Keep it opt-in and declared (README, `docs/ENRICHMENT.md`, card, `docs/RESPONSIBLE_AI.md`).
  - Profile titles come from `forcecrm/titles.py`: the LLM picks phrases, stored in `activity_families`. Don't confuse this with the per-ΚΑΔ `activity_mix` from `evidence.py`.
  - Login and help chat (`forcecrm/auth.py`, `forcecrm/assistant.py`, `web/login.*`, `web/onboarding.js`): demo accounts only, all local. The assistant uses `DEFAULT_LLM`, answers only from `assistant.HELP`, and must stay out of profiles/scores/ranking. Update `HELP` when the UI changes. Support hand-off is rule-based (`HUMAN_TERMS`, `PROBLEM_TERMS`, repeated question), not left to the model.
  - One-click demo sign-in: `POST /api/auth/demo` (`auth.demo_login`). It uses the built-in user «Επίδειξη»: unusable password marker, tour already done. `FORCECRM_DEMO_LOGIN=0` disables it. Use it for the live demo.
  - Never commit data or outputs: `*.csv`, `*.xlsx`, `outputs/`, `models/` and `.cache/` are gitignored.

### Environment on this Mac (Apple A18 Pro, 8 GB RAM, macOS 27)
- **Python:** needs **3.12** (3.13 resolves; 3.11 and 3.14 fail on the pinned wheels). `run.py` checks the **venv's** interpreter, not the launcher's: launched with 3.14 it builds `.venv` from `python3.12`/`python3.13` on PATH or a uv-managed 3.12 (task/T2-sqlite, 2026-11-07). The system `python3` is 3.14; there is no `python3.12` on PATH. The uv-managed interpreter is `~/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12`. ⚠️ **Inside the repo, `uv python find 3.12` returns the teammate branch's `.venv`, which is the wrong environment.** Run it from outside the repo, or use the path above.
- **Ollama:** installed via Homebrew. Start it with `ollama serve`, or let `run.py` start it. Pulled models: `ministral-3:3b` (default), plus `qwen3:1.7b` and `gemma3:1b` for the benchmark.
- **scipy:** 1.15.3's macOS wheel fails to load here (dyld `__thread_bss` error), so scipy 1.18.1 and numpy 2.5.4 are pinned.
- **Durable one-command setup in the repo** (not done yet; ask the user first, because step 1 deletes the teammate's 1.5 GB uv env, which `uv sync` on that branch can recreate):
  1. `rm -rf .venv`. `run.py` hardcodes `.venv`, and the existing one is the ICP branch's.
  2. `cp ../Data/data/vwHack*.csv .`
  3. `~/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 run.py --first-run`. This creates `.venv`, installs `requirements-ai.txt`, downloads E5-small to `models/e5-small`, pulls `ministral-3:3b` and serves http://127.0.0.1:8000.
- **Working environment used on 2026-11-07 (temporary, may disappear on reboot):** a clean clone with venv, CSVs and models at `/private/tmp/claude-501/-Users-gribgen-workspace-FORCECRM/26b027b9-790e-4caa-874f-51f86226274c/scratchpad/clean`.
  - Contents: `.venv` (py3.12, `requirements-ai.txt` + reportlab), `models/e5-small`, `models/e5-base`, `models/multilingual-minilm`, and the three CSVs.
  - To run the repo's code against it, set `C=<that path>`, then:
    `PYTHONPATH=$PWD HF_HUB_OFFLINE=1 $C/.venv/bin/python -X utf8 -m scripts.analyze --data $C --portfolio ../Data/ForceCRM_Hackathon_Pelatologio.xlsx --output outputs/real`.
  - The same pattern works for the other scripts and for `-m pytest -q`.
  - If the folder is gone, use the durable setup above.

### Commands (inside a provisioned `.venv`; prefix `PYTHONPATH=$PWD` and `--data <dir with CSVs>` when using another env)
| Task | Command | Notes |
|---|---|---|
| Web app | `python3.12 run.py` | http://127.0.0.1:8000 |
| Full analysis | `.venv/bin/python -m scripts.analyze --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --output outputs/real` | ~1.6–2.5 min cold (mostly the 3 LLM titles + descriptions); ~2 s when every stage is cached (`.cache/stages/`). `--from-stage <s>` recomputes `s` and everything after it, `--until-stage <s>` stops there, `--no-cache` skips the disk cache. `--model` defaults to `DEFAULT_LLM` |
| Validate exports | `.venv/bin/python -m scripts.validate_exports --output outputs/real --portfolio <xlsx>` | Must print `PASS` |
| Tests | `.venv/bin/python -m pytest -q` | 75 pass (`tests/conftest.py` sets `FORCECRM_AUTH=0`; `tests/test_auth.py` turns the login back on) |
| End-to-end (running server) | `.venv/bin/python -m scripts.e2e_server --url http://127.0.0.1:8000 --portfolio <xlsx> [--mode full\|weights\|restore]` | Drives the same HTTP calls as the browser, signing in first as `e2e-test` (delete that user from `.cache/forcecrm.sqlite` before a demo). The server reads CSVs and `models/e5-small` from the repo root |
| 5-split recall audit | `.venv/bin/python -m scripts.audit_ranking --with-tourism` | Use `--with-tourism` to match the shipped pipeline; without it you get the older numbers |
| Representation experiment | `.venv/bin/python -m scripts.experiment_representation --portfolio <xlsx> [--sweep]` | Activity centring vs finer geography |
| Benchmarks | `scripts.benchmark embeddings` / `embedding-recall --portfolio <xlsx>` / `llms --profiles outputs/real/profiles.json` | Need `models/e5-base`, `models/multilingual-minilm`, `qwen3:1.7b`, `gemma3:1b` |
| AI Act PDF (1 page) | `.venv/bin/python -m scripts.build_ai_act` | Renders `docs/RESPONSIBLE_AI.md` to `docs/ForceCRM-AI-Act-Self-Assessment.pdf`; needs reportlab. Rebuild after editing the Markdown, and keep it to 1 page |
| T5 co-declaration experiment | `.venv/bin/python -m scripts.experiment_codeclaration --portfolio <xlsx>` | Results in `docs/ARCHITECTURE.md` §3 |
| PDF model card | `.venv/bin/python -m scripts.build_card` | Needs `pip install reportlab`, `outputs/real`, `outputs/embedding-comparison.json` and `outputs/llm-comparison.json` |

### Measured results (reproduced on this Mac)
| | Recall@500 | Recall@2000 |
|---|---:|---:|
| `main`, mean of 5 splits, with tourism signal (= shipped pipeline, secondary-ΚΑΔ rule of 2026-11-08) | 21.7% | 51.0% |
| `main`, mean of 5 splits, without tourism (`audit_ranking` default, `docs/ranking-audit.json`) | 18.0% | 50.3% |
| `main`, default seeded split (seed 2026, in `report.json`) | 15/60 (25.0%) | 33/60 (55.0%) |
| `feat/icp-engine-and-ui`, 5-fold | 14.0% | 29.7% |
| Primary-ΚΑΔ baseline | ~1.3–1.7% | ~7–8% |

- **Profiles:** `main` finds 3: ICP-01 Kassandra-Halkidiki + Thessaloniki (73), ICP-02 Attica/ΙΚΕ (100), ICP-03 tourist islands, i.e. Cyclades, Heraklion, Chania, Dodecanese (127). Stability is 92% / 93% / 94%, with worst 10% at 90% / 83% / 88%. Numbering changed with the 2026-11-08 secondary-ΚΑΔ rule.
- **Secondary ΚΑΔ (challenge owner's guidance, 2026-11-08):** the primary dominates, related secondaries take part (incl. non-food at 5%), and trust falls as 5/n above 5 declared and is zero above 20. Recall-neutral, stability up. Policy: `features.SECONDARY_DEFAULT`. See `docs/ARCHITECTURE.md` §2.3.
- **Portfolio-wide business traits:** ΙΚΕ 110/300 (2.4×) and school canteens 19/300 (2.4×, q = 0.02).
- **LLM benchmark** (share of sentences passing the grounding checks / seconds per profile): Ministral 3 3B 61% / 50 s; Qwen3 1.7B 100% / 32 s; Gemma 3 1B 94% / 17 s. **Decision (user, 2026-11-07): ship Ministral 3 3B** for its European provenance. It overrode Dimitris's switch to Qwen during the merge. Revisit when Krikri-8B or EuroLLM-9B are benchmarked (T4). `DEFAULT_LLM` in `forcecrm/narrative.py` is the single source of truth, and `run.py` pulls whatever it names.
- **Embeddings:** five models compared on the code-family proxy (hit@5) and on the real task (5-split recall@500/@2000 with only the embedding swapped):

  | Model | hit@5 | Recall@500 / @2000 |
  |---|---:|---:|
  | E5-small (shipped) | 93.2% | 21.3% / 51.3% |
  | E5-base | 93.2% | 21.0% / 50.0% |
  | MiniLM | 89.0% | 20.0% / 50.3% |
  | granite r2 311M | 94.5% | 21.3% / 49.7% |
  | BGE-M3 | 95.9% | 20.0% / 50.3% |

  None beats E5-small on the task. BGE-M3 publishes only `pytorch_model.bin`: `scripts.download_models` filters for safetensors, so fetch that file with `hf_hub_download`. The benchmark models live in the temporary env's `models/`.
- **Determinism:** the same inputs give byte-identical `lookalikes.csv` and `assignments.csv`, verified across separate runs.

### Working conventions for agents
- **Ranking or features:** after any change to `features.py`, `clustering.py`, `affinity.py`, `pipeline.rank_candidates` or `forcecrm/stages/`, run `audit_ranking --with-tourism`. Compare against 21.7% / 51.0%; with 60 held-out VATs per split, ±2–3 points is noise. Check that `lookalikes.csv` is unchanged when you didn't intend a ranking change.
- **Any change:** run `pytest`, then a full `scripts.analyze` and `scripts.validate_exports` (must `PASS`).
- **Greek text:** wording rules live in `narrative.py`: `BANNED`, `BAD_PATTERNS`, `check_sentence`, few-shot `EXAMPLES`.
  - Every number in a sentence must come from its source fact.
  - Computed sentences (activity mix, quality/stability/confidence) are templates with `author: computed template`.
  - `validate_exports` requires `description == ' '.join(claims[].sentence)`.
- **When numbers change, update all of these together:**
  - `README.md` (Model comparison and Methodology)
  - `docs/MODEL_DATA_CARD.md`
  - the PDF (rebuild with `scripts.build_card`)
  - `docs/COMPETITION_AUDIT.md`
  - `docs/ARCHITECTURE.md`
  - this file
- **Commits:** commit or push only when the user asks; they usually want `main` pushed. End commit messages with the `Co-Authored-By` line from the system prompt. Keep the AI-tools declaration current: README "AI coding assistant disclosure", the Markdown card, and the `build_card.py` text.
- **Long runs:** the Mac has 8 GB RAM, so don't run two LLM-heavy jobs at once.

### Open problems on `main` (priority order)
1. **Visual browser check pending.** `scripts/e2e_server.py` passed end to end against the real server on 2026-11-07 (full run 24/24, custom weights 25/25, restore after restart 10/10), and with the login on on 2026-11-08 (30/30, Dimitris). The T2/T3/T16 merge was **not re-tested** after merging into `main`; the user said to skip tests. Run `pytest` and `scripts.e2e_server` before the demo. It covered upload, analysis, every export, `all.zip`, security rejections and the live check without a key. Nobody has yet *looked* at the rendered pages: portfolio-traits card, activity mix, titles, per-profile rank, map, options panel, dark mode. The live ΓΕΜΗ check with a real key is also untested.
2. **Greek fluency** has no native-speaker rating yet (the benchmark JSON has null rating fields). Activity names appear in ΓΕΜΗ capitals inside «».
3. **Clean install untested on Windows/Linux.** The durable repo setup on this Mac hasn't been done (see Environment).
4. **LLM default:** Ministral passes the checks least often (61%) and is the slowest. Qwen3 or Gemma are faster and pass more, but they aren't European. It's the user's call.
5. **Known limits the docs admit:** confidence is uncalibrated; the statistics are exploratory (the same data forms and tests the clusters); profiles on this portfolio are geographic by nature (`docs/ARCHITECTURE.md` §3).
6. **Cosmetic:** the PDF card's page header says "TECHNICAL BRIEF" (the original author's label).
7. **ΓΕΜΗ live check:** confirm the API host (`opendata-api.businessportal.gr` vs `opendata.businessportal.gr`; override with `GEMI_API_URL`) and the ODC-BY 1.0 licence on the official portal. So far the licence is confirmed only by a third-party source.

Done 2026-11-07:
- Greek narration rework, with quality/stability/confidence stated in the text
- AI-tools declaration includes Claude Code
- PDF card rebuilt on macOS from fresh benchmarks
- Architecture doc with a determinism analysis
- Geographic-profile investigation, plus portfolio-wide traits and a per-profile activity mix

### Local leftovers
- **`data/`** in the repo: symlinks to `../Data` created for the ICP branch (untracked; safe to delete).
- **`output/`** (singular) in the repo: old exports from the ICP branch (untracked; not `main`'s `outputs/`).
- **`.venv/`** in the repo: the ICP branch's uv env (1.5 GB). See the durable setup above before replacing it.
- **Worktrees:** two git worktrees (`local-ai`, `stage1`) are registered in a temporary scratchpad from the review session. Once that folder is gone, run `git worktree prune`.

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
