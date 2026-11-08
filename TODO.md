# TODO: shared task board (multi-agent)

Goal: maximise the score on the brief's judging criteria (`../ForceCRM_Hackathon_2026_Technical_Brief.pdf` §8.2, deck slide 6):
- completeness
- model choice (4 axes, with measurements)
- results (Recall@500/2000 vs baseline)
- understandability for a salesperson
- responsible AI
- presentation
- extra points X1–X3

Read [CLAUDE.md](CLAUDE.md) (environment, commands, conventions) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## How to use this board

1. **Claim** a task: set its `Status` to `claimed: <agent/person>, <date>, branch <name>` and commit only that line. Then re-read the board, in case someone else claimed the same task first.
2. **Work on a local branch** `task/<ID>-short-name`, or a git worktree. The user wants **exactly 4 branches on GitHub**. Push a task branch only if needed for review, and delete it from GitHub after merging into `main`.
3. **Stay in your lane.** Each task lists the files it owns. If you must touch another lane's files, coordinate through a note on that task.
4. **Done means** the task's acceptance criteria hold, plus:
   - `pytest` passes
   - a full `scripts.analyze` run plus `scripts.validate_exports` prints `PASS`
   - for any ranking or feature change, `scripts.audit_ranking --with-tourism` results are reported against **21.7% / 51.0%** (5-split mean)
   - the docs are synced (list in CLAUDE.md, "When numbers change…")
5. Set `Status: done (<commit>)` and add one line of results. **Commit or push only when the user asks.**
6. Tasks marked 🧑‍⚖️ need a **user decision** before implementation: write the options with measurements, then ask.

Legend: **P0** = needed to submit or could cost auto-graded points · **P1** = strong scoring impact · **P2** = nice to have.

## Lanes and dependencies

| Lane | Tasks | Owns these files | Depends on |
|---|---|---|---|
| A: UI | T1, T13 | `web/*`, the UI parts of `forcecrm/server.py` | T2 API, if the UI reads from the DB |
| B: Data/DB | T2 | `forcecrm/db.py` (new), `forcecrm/data.py`, server data endpoints | n/a |
| C: Pipeline | T3 | `forcecrm/pipeline.py`, `forcecrm/stages/` (new) | Coordinate with T2 (stage artifacts may live in SQLite) |
| D: Models | T4, T9 | `forcecrm/narrative.py` (`DEFAULT_LLM`), `scripts/benchmark.py`, model docs | n/a |
| E: Engine | T5, T10, T11 | `forcecrm/features.py`, `clustering.py`, `affinity.py`, `evidence.py` | Run after T3 merges, or agree the stage interface with C |
| F: Deliverables | T6, T7, T8, T12, T14, T15 | `docs/*`, `scripts/validate_exports.py`, `README.md` | Final numbers from D and E |

Suggested order: T6 (quick, P0) and T7 in parallel with T1/T2/T3, then T4 and T5, then T8/T12/T14 last, once the numbers are final.

## User's tasks

### T1 · Fix the UI: functional login; remove or fix non-functional buttons · P1 · Lane A
- **Status:** login part done and merged into `main` on 2026-11-08 (Dimitris, branch `task/T16-onboarding`, deleted). It also has a one-click demo sign-in with no account and no tour pop-up (2026-11-08): demo accounts (new username = new account, existing username checks its scrypt password), HttpOnly/SameSite=Strict session cookie, 401 on the API without a session, logout, first-login guided tour and a local help assistant with support hand-off. `scripts.e2e_server` 30/30 with login on. The button audit below is still open. Since the merge, `main` also has dark mode, the "Περισσότερες επιλογές" panel (weights and live ΓΕΜΗ check) and an offline Greece map. Include them in the click-through audit. UI made friendlier on 2026-11-08: plain-Greek labels, certainty badges instead of raw p/q-values, collapsible profiles, hover explanations, plain-language recall sentences.
- **Context:** there is **no login today**, neither on `main` (FastAPI + `web/`) nor on the teammate's Streamlit branch. The app binds to 127.0.0.1 and has origin/host checks only.
- **Decide first 🧑‍⚖️ (ask the user):**
  - Is login a real requirement (multi-user, per-user runs) or demo polish?
  - Recommended minimum: a single local account. The password hash (stdlib `hashlib.scrypt`) is stored in SQLite from T2. Use an HttpOnly, SameSite=Strict session cookie; a first-run screen sets the password, and there is a logout button.
  - No external identity provider; everything must stay local.
- **Work:**
  - Audit every button and link in `web/index.html` / `web/app.js`. The nav, upload, demo, template download, pagination and export buttons all have handlers in code, but the whole flow has **never been clicked in a browser**.
  - Verify downloads (including `all.zip`) also work for the **latest run restored on startup** (`/api/latest`), not only for jobs started in the current server process.
  - Remove anything that does nothing.
  - Check Greek labels and decimal formatting.
  - Check that it works at 1280 px and on a laptop screen.
- **Acceptance:**
  - Login, logout and protected API routes work; unauthenticated API calls get 401.
  - No dead controls remain.
  - A recorded manual test covers upload → profiles (including the portfolio-traits card and activity mix) → prospects → map → validation → each export.
  - Add API tests in `tests/` for auth and the export endpoints.

### T2 · Embedded database (SQLite, serverless) for data fetching · P1 · Lane B
- **Status:** done, merged into `main` on 2026-11-08 (Dimitris, with Claude Code; branch deleted). `forcecrm/db.py`: cold import 1.5 s, warm load 0.6 s, 20 MB. `lookalikes.csv`/`assignments.csv`/`enrichment.csv` byte-identical to `main`; `profiles.json` differs only in `llm.seconds`. `validate_exports` PASS; 56 tests. Prospects table pages via `/api/runs/{id}/lookalikes` (job payload 11.1 MB → 0.4 MB). Live ΓΕΜΗ cache moved into SQLite. `users` table created for T1 (schema only).
- **Context:** every run re-reads three CSVs (24,424 venues, 161,964 activity rows) with pandas, and the UI receives the whole result. "Serverless" here means SQLite: a single file, standard library, no daemon, which is fine for the "one command" rule.
- **Work:**
  - Create `forcecrm/db.py`. On first start it imports the CSVs into `.cache/forcecrm.sqlite` (gitignored), keyed by the CSVs' SHA-256 (already computed in `report.json`). Re-import only when the hashes change.
  - Tables:
    - `companies`
    - `activities`
    - `activity_descriptions`
    - `runs` (id, portfolio hash, created, params, status)
    - `profiles`
    - `profile_traits`
    - `assignments`
    - `lookalikes` (indexed by run, profile, rank)
    - `users` (for T1)
  - Add paginated, filterable API endpoints: lookalikes by profile, search by name or municipality, and company detail. The UI should fetch pages instead of the full preview.
  - Keep the file exports (`profiles.json`, `assignments.csv`, `lookalikes.csv`) **byte-identical** to the current ones. They are auto-graded.
- **Acceptance:**
  - A cold start imports in under 60 s on the 8 GB Mac; warm starts take under 2 s.
  - Exports are unchanged (compare SHA-256 against a pre-change run).
  - Tests cover import idempotency and pagination.
  - `docs/ARCHITECTURE.md` is updated.

### T3 · Re-architect as an explicit, layered, deterministic pipeline · P1 · Lane C
- **Status:** done, merged into `main` on 2026-11-08 (Dimitris, with Claude Code; branch deleted). `forcecrm/stages/`: 10 stages (`load, enrich, represent, discover, stability, evidence, profiles, rank, narrate, evaluate`), Merkle cache keys over CSV/portfolio/model/library hashes and stage source code, artifacts in `.cache/stages/`; `narrate` cached per profile on its facts + Ollama digest. `rank` now runs before `narrate`. `lookalikes.csv`, `assignments.csv`, `enrichment.csv` and `profiles.json` (descriptions included) identical to the pre-refactor run; `validate_exports` PASS; e2e full 24/24, weights 25/25; 63 tests. Timing on the Linux laptop: cold 235 s → code-changed rerun with cached text 19 s → fully cached 2.1 s. CLI `--from-stage/--until-stage/--no-cache`; stage diagram in `docs/ARCHITECTURE.md` §1/§2.0. The suggested `ingest/validate` are one stage (`load`) and `assign` is part of `discover`, since assignment is the second half of the same fit. Follow-ups: the UI could show prospects before the text is ready (lane A); `audit_ranking`/`experiment_representation` still build features themselves.
- **Context:** `pipeline.run_analysis` is one large function. The judges value clarity ("understandability", presentation), and the brief's requirements map naturally to stages.
- **Work:**
  - Split it into explicit stages, each with typed inputs and outputs, a version string, and a content hash of its inputs. Suggested stages:
    1. `ingest`
    2. `validate`
    3. `enrich`
    4. `represent` (A1)
    5. `discover` (A2)
    6. `assign` (A3)
    7. `evidence` (A4)
    8. `stability` (A5)
    9. `narrate` (A6)
    10. `rank` (A7)
    11. `evaluate`
    12. `export` (A8)
  - Cache each stage's artifact by its hash (in SQLite from T2 or `.cache/`), so re-runs only recompute what changed. Log per-stage timing to `report.json`.
  - Keep "deterministic-ish" explicit:
    - Stages 1–8 and 10–12 must stay bit-identical for the same inputs.
    - Only stage 9 (LLM text) may vary across machines, and it is already fact-checked.
  - Add a stage diagram to `docs/ARCHITECTURE.md` and a `--from-stage` / `--until-stage` CLI option.
- **Acceptance:**
  - `lookalikes.csv` and `assignments.csv` are byte-identical to `main` before the refactor.
  - All tests pass, plus a test that two runs produce identical stage hashes.
  - A run with only the narration changed skips stages 1–8.

### T4 · Land on the LLM to ship 🧑‍⚖️ · P1 · Lane D
- **Status:** decided for now (user, 2026-11-07): **ship Ministral 3 3B**, which overrides the Qwen switch in Dimitris's merged branch. The work below (measuring Krikri and EuroLLM) remains open. Re-ask the user if a Greek-specialised European model wins.
- **Data so far** (Apple A18 Pro, 8 GB; share of sentences passing the grounding checks / seconds per profile / RAM):

  | Model | Pass rate | Time per profile | RAM |
  |---|---:|---:|---:|
  | Ministral 3 3B (FR, Apache, current default) | 61% | 50 s | 2.3 GB |
  | Qwen3 1.7B (CN) | 100% | 32 s | 1.1 GB |
  | Gemma 3 1B (US, Gemma terms) | 94% | 17 s | 0.8 GB |

  Not yet measured: the brief's European, Greek-strong options, **Llama-Krikri-8B** (Greek, ILSP) and **EuroLLM-9B** (EU, Apache), plus Salamandra-7B and Apertus-8B. At Q4 they need about 5–6 GB, which is tight on this 8 GB Mac but fine on the brief's 16 GB laptop.
- **Work:**
  - Add Krikri-8B and EuroLLM-9B (Q4 GGUF via Ollama) to `scripts/benchmark.py llms`. Report pass rate, latency and RAM.
  - Check each model's **exact repo licence file** (axis 2: Krikri is under the Llama 3.1 licence, which allows commercial use with conditions) and its origin (axis 3).
  - Get a **native-speaker blind fluency rating, 1–5** (see T9).
  - Present a recommendation and let the user decide. Switching is one line: `DEFAULT_LLM` in `forcecrm/narrative.py`, plus the `ollama pull` in `run.py` and the docs.
- **Acceptance:**
  - The decision is recorded in CLAUDE.md, README, `MODEL_DATA_CARD.md` and the PDF card (rebuild).
  - The benchmark JSON covers at least 3 LLMs, including at least 1 European Greek-capable 7–9B model.

### T5 · Make profiles KAD-based instead of location-based · P1 · Lane E · 🧑‍⚖️ trade-off
- **Status:** partly done (2026-11-08). (1) Secondary ΚΑΔ weighted per the challenge owner's guidance (shipped, recall-neutral, stability up). (2) Co-declaration experiment (`scripts/experiment_codeclaration.py`): a 50/50 E5 + co-declaration mix is recall-neutral (21.0% / 52.3%) and makes profiles more activity-based, but merges 3 profiles into 2. **Awaiting the user's decision** whether to ship it (ARCHITECTURE §3).
- **Context (read first):** `docs/ARCHITECTURE.md` §3 and `scripts/experiment_representation.py`.
  - Today's profiles are regions, because the E5 embeddings of all ΚΑΔ 56 descriptions are nearly identical (mean cosine 0.91). Activity carries 65% of the weight but only 8% of the separation between profiles.
  - Centring the activity embeddings made the clusters activity-driven (81% of separation), but 5-split recall fell from 21.3% / 51.3% to 18.0% / 44.7%, and the significant traits stayed geographic.
- **Ideas to try** (measure each with `audit_ranking --with-tourism` plus the experiment script):
  1. **Decouple profile formation from ranking geography.**
     - Form profiles in an activity-only space.
     - Keep geography and tourism in the ranking's market-affinity term (`affinity.py`), which already carries the location signal.
     - A3 says forming and assigning must use the same logic and metric. That still holds if assignment uses the same activity space, while the ranking adds affinity as a documented second factor.
  2. **A better ΚΑΔ representation.**
     - The teammate's branch (`origin/feat/icp-engine-and-ui`, `src/features.py`) uses **PPMI co-occurrence of ΚΑΔ declared together** (from 161,964 company-ΚΑΔ links), reduced with SVD and fused with neural embeddings.
     - Co-declaration patterns separate café-bar vs taverna vs canteen far better than description text.
     - Port the idea (not the branch; it must not be merged). Their claimed recall is 14.0% / 30.7%, but that's for their whole pipeline, not this component.
  3. **Use secondary ΚΑΔ outside 56** (e.g. retail, bakery, catering, accommodation). These are what actually differ between food businesses. Today only 56xx secondaries are kept.
  4. **Two-level profiles:** activity profile × region facet. A profile reads "school canteens (Attica 40%, Thessaloniki 20%)", and the region becomes a filter rather than the profile's identity.
- **Acceptance:**
  - Each profile's top significant traits are activity-based (primary or secondary ΚΑΔ, or OSM amenity).
  - Recall is reported against the current 21.7% / 51.0% reference.
  - **If recall drops by more than noise (~3 points), stop and ask the user** whether the trade-off is acceptable. Recall is also judged.
  - Update ARCHITECTURE §3, README, card and audit.

## Additional tasks from the brief, deck and judging criteria

### T6 · Check the export schema against the brief (auto-graded) · P0 · Lane F
- **Status:** partly done (2026-11-07). `profiles.json` now also has `confidence` (same value as `confidence_score`), and the validator asserts the brief's keys. `lookalikes.csv` keeps exactly the 5 columns, with `rationale` for «αιτιολόγηση». **Still needed: the organisers' answer on exact column and key names.**
- **Risk:** the brief §10 names the files and fields and says they're graded automatically.
  - `lookalikes.csv` should be `profile_id, company_id, rank, score, αιτιολόγηση` (justification). Ours is `rationale`.
  - `profiles.json` should have "εμπιστοσύνη" (confidence). Ours is `confidence_score`. It should also have "χαρακτηριστικά με lift και p-value"; ours is `significant_features` (✓).
- **Work:**
  - Ask the user to check with the organisers which exact column and key names the grader expects.
  - Until then, add **aliases** without removing anything:
    - `confidence` next to `confidence_score`
    - `traits` (or keep it) next to `significant_features`
    - a `justification` column, *if* the grader tolerates extra columns
  - Otherwise rename consistently in the code, the validator and the docs.
- **Acceptance:** `scripts/validate_exports.py` asserts the agreed schema exactly; README's export table matches.

### T7 · End-to-end browser test plus a durable one-command install · P0 · Lanes A + F
- **Status:** API-level end to end done (`scripts/e2e_server.py`, all checks pass on macOS). Still open: a visual click-through, a clean install on Windows and Linux, and the durable setup on this Mac.
- **Work:**
  - Do the durable setup on this Mac (CLAUDE.md "Durable one-command setup"; ask the user before deleting the ICP `.venv`).
  - Verify `run.py --first-run` on a **fresh clone** on Windows and on Linux (or in a Linux container). The brief requires one command in a clean environment.
  - Click through the whole UI (shared with T1).
- **Acceptance:** recorded success on 2 OSes, with times; README troubleshooting is updated with any issues found.

### T8 · 5-minute presentation and live-demo script · P0 · Lane F
- **Status:** deck drafted (2026-11-07/08, Claude artifact «ForceCRM · Ποιος θα είναι ο επόμενος πελάτης;», 10 slides in Greek with speaker notes, demo slide includes the one-click demo sign-in). Still open: team names on the cover, share it with the team, a rehearsal within 5 minutes, and the optional backup slide with screenshots.
- **Work:**
  - Build a deck for the "Παρουσίαση" criterion: clarity, demo, business value. Suggested slides:
    1. the problem
    2. live demo
    3. how it works (pipeline diagram)
    4. results vs baseline (Recall@500/2000, 5-split)
    5. model choice on the 4 axes, with measurements
    6. responsible AI (AI labels, groundedness, no credit score, AI Act)
    7. extra points
    8. limitations
  - Cite the data sources on a slide: **ΓΕΜΗ Open Data; GPS © OpenStreetMap contributors (ODbL)**. The brief requires this in every presentation.
  - Write a demo script with a **pre-computed run as fallback**: a full run takes about 1.6–2.5 min on 8 GB, so warm up Ollama before going on stage.
- **Acceptance:** the deck exists, the demo was rehearsed within 5 minutes, and the fallback path was tested.

### T9 · Human Greek fluency rating · P1 · Lane D
- **Status:** open
- **Work:**
  - A native speaker blind-rates the descriptions (fluency and coherence, 1–5) for each LLM in `outputs/llm-comparison.json`. The fields `human_greek_fluency_1_to_5` and `human_coherence_1_to_5` are currently null.
  - Also review the lookalike justifications. They currently mix Greek labels with dot decimals ("Ομοιότητα 93.3%"); use comma decimals.
- **Acceptance:** the ratings are filled in, cited in the card (model-choice axis 1: "Greek quality measured on challenge data"), and feed T4.

### T10 · Embedding comparison with brief-listed models · P1 · Lane D/E
- **Status:** done (2026-11-07). Added granite r2 311M and BGE-M3 (both on the brief's list), plus a task-level metric (`scripts.benchmark embedding-recall`). None beats E5-small on 5-split recall, and the card and README are updated. Optional: Qwen3-Embedding-0.6B or jina-v5-nano.
- **Context:** we compared E5-small, E5-base and MiniLM. MiniLM is not on the brief's list. The brief asks for at least 2 alternatives on the 4 axes and lists:
  - granite-embedding multilingual r2 (IBM, Apache)
  - BGE-M3 (MIT)
  - Qwen3-Embedding-0.6B
  - jina-v5-nano (verify the licence and Greek support)
  - EmbeddingGemma
- **Work:**
  - Add granite-r2 (311M) and BGE-M3 to `scripts/benchmark.py embeddings`.
  - Add a task closer to the goal than code-family hit@5. For example, measure holdout recall when only the embedding changes (with the T5 representation).
  - Check each repo's licence file.
- **Acceptance:** the card's embedding table has at least 4 models, including at least 2 from the brief's list, with licence and origin.

### T11 · Recall improvements without overfitting · P1 · Lane E
- **Status:** open
- **Work ideas:**
  - Tune `affinity.py` signal weights and the shrinkage prior with nested CV (do not tune on the same 5 splits you report).
  - Add OSM-amenity and secondary-ΚΑΔ affinity signals.
  - Try a per-profile quota in the top 2,000 so small profiles aren't drowned out.
  - Try a learning-to-rank over the existing features using only training labels.
- **Acceptance:** improvement is shown on held-out splits that tuning never saw; the report gives mean ± sd; determinism is preserved.

### T12 · Extra points X3: open data that explains the portfolio · P2 · Lane F (+E)
- **Status:** open
- **Insight to build on:** customers over-index on **school canteens** (2.4×, q = 0.02) and on ΙΚΕ companies.
  - Add school counts or pupils per municipality from data.gov.gr / the Ministry of Education open data, as a market-context feature and trait (like the ELSTAT hotel-nights band).
  - Optional: population density (ELSTAT census 2021), or university and hospital locations.
- The brief rewards **insight over number of sources**: show that the new source makes a profile more explainable or improves recall.
- **Acceptance:** the source and licence are declared in `docs/ENRICHMENT.md` and the card; the effect is measured; a snapshot is bundled in `open_data/` (no runtime network calls).

### T13 · Salesperson-friendly prospect view · P1 · Lane A (+E for wording)
- **Status:** done (2026-11-08): plain-Greek justifications, per-profile rank, a company card (click a prospect's name) and a CSV export of the filtered list (`/api/runs/{id}/lookalikes.csv`, Excel-friendly).
- **Work:**
  - A7 says lookalikes are ranked **per profile**. We export one global rank with the nearest profile. Add a per-profile rank in the UI, and optionally as an extra export column if T6 allows it.
  - Rewrite justifications in plain Greek, e.g. "Καφετέρια ΙΚΕ στον Δήμο Κασσάνδρας, όπως 30 πελάτες σας του προφίλ ICP-02".
  - Add a click-through to the company card: name, address, ΚΑΔ, OSM link if matched.
  - Add a CSV export of the filtered view.
- **Acceptance:** a non-technical reader can say why each of the top 10 prospects was suggested.

### T14 · Responsible-AI and card polish · P1 · Lane F
- **Status:** mostly done (2026-11-08). `docs/RESPONSIBLE_AI.md` covers the 4 questions, Annex III 5(b), Art. 50, Art. 4 and Reg. 2026/1744; a one-page PDF is built by `scripts/build_ai_act.py`; the PDF card header is fixed. Still open: confirm the ΓΕΜΗ API host and licence on the official portal.
- **Work:**
  - Check `docs/RESPONSIBLE_AI.md` against the brief §9:
    - answers the 4 questions explicitly
    - covers Art. 50 labelling (in force from 2 Aug 2026)
    - covers the ΑΤΟΜΙΚΗ = natural-person trap (no solvency score anywhere, including the UI wording)
    - covers Art. 4 AI literacy
    - mentions Annex III and the 2 Dec 2027 date (Reg. 2026/1744)
    - fits on **1 page**: export a PDF
  - PDF model card: fix the header ("TECHNICAL BRIEF" → "MODEL & DATA CARD") and add the portfolio-traits / geography finding.
  - AI-tools declaration: add the teammate's tools if they used any. Dimitris's merged commit was co-authored by Claude Code, which is already declared.
  - ΓΕΜΗ live check: confirm the API host and the ODC-BY 1.0 licence on the official portal (`docs/ENRICHMENT.md`, "Still to check").
- **Acceptance:** both PDFs fit the page limits (card 1–2 pages, AI Act 1 page) and their numbers match the code.

### T15 · Confidence calibration · P2 · Lane E
- **Status:** open
- **Context:** `confidence_score` and lookalike `score` are uncalibrated, and the docs admit it.
- **Work:** on holdout splits, fit an isotonic or Platt mapping from score to "probability of being a hidden customer within the top-k". Report a reliability plot. Never call it a purchase or credit probability.
- **Acceptance:** calibration curve in the card; UI wording reviewed for T14 compliance.

## Done (for context)

All on 2026-11-07, on `main`:
- Greek narration rework: per-sentence, fact-checked, quality/stability/confidence stated in the text.
- AI-tools declaration includes Claude Code.
- PDF card rebuilt on macOS from fresh benchmarks.
- `docs/ARCHITECTURE.md`, including the determinism analysis.
- Geographic-profile investigation, plus portfolio-wide business traits and per-profile activity mix.
- CLAUDE.md agent handoff and `AGENTS.md`.
