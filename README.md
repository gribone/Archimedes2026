# ForceCRM · Local Customer Intelligence

A localhost application for discovering 1–6 ideal customer profiles in Greek HORECA businesses, explaining them in Greek, and ranking new prospects. FastAPI serves a dependency-free Greek browser interface; NumPy/scikit-learn perform analysis; E5 embeds Greek activity descriptions; Ollama runs Ministral locally.

The supplied portfolio yields 3 profiles, 24,424 venue assignments and 23,645 distinct prospective VATs. On the seeded internal split of 240 training and 60 held-out customers, ForceCRM retrieves 15/60 (25.0%) at 500 and 33/60 (55.0%) at 2,000; the primary-code baseline retrieves 1/60 and 5/60. Averaged over five paired splits, recall is 18.0% at 500 and 50.3% at 2,000 versus 1.3% and 8.0% for the baseline ([details](docs/COMPETITION_AUDIT.md)). The organizer scores its own hidden VAT list separately.

## Run locally

### Prerequisites

- **Python 3.12** (3.13 also resolves; 3.11 and 3.14 are not supported by the pinned wheels). Python must include pip.
- [Ollama](https://ollama.com/download) 0.13.1 or newer (macOS: `brew install ollama` also works).
- About 8 GB free disk space; 16 GB RAM recommended (8 GB works, more slowly).

### 1. Add the competition data

The data files are not in the repository. Copy the three supplied CSVs into the repository root, next to `run.py`:

```text
Archimedes2026/
├── run.py
├── vwHackCompany.csv
├── vwHackCompanyActivity.csv
├── vwHackActivities.csv
└── ForceCRM_Hackathon_Pelatologio.xlsx   (optional here; you upload it in the browser)
```

`.gitignore` excludes `*.csv` and `*.xlsx`, so these files are never committed.

On first start the CSVs are imported into a local SQLite file, `.cache/forcecrm.sqlite` (about 1.5 s, 20 MB, gitignored). It is re-imported automatically whenever a CSV changes. Analysis runs and ΓΕΜΗ statuses are stored there too, so the prospects list pages and filters without sending the whole result to the browser. Nothing to install: SQLite ships with Python. See [docs/ARCHITECTURE.md §2.12](docs/ARCHITECTURE.md).

### 2. One-command launch

macOS / Linux:

```bash
python3.12 run.py --first-run
```

Windows:

```powershell
py -3.12 run.py --first-run
```

The launcher checks the Python version, creates `.venv`, installs dependencies, downloads both models, starts Ollama if it is not already running, and serves **http://127.0.0.1:8000**. Later launches use `python3.12 run.py` (`py -3.12 run.py` on Windows) and work offline.

To prepare the models separately and launch later:

```bash
python3.12 run.py --setup-models
python3.12 run.py
```

This downloads `intfloat/multilingual-e5-small` to `models/e5-small`, records its exact revision, and runs `ollama pull ministral-3:3b` (Q4_K_M, approximately 3 GB). If Ollama is not installed, the setup command explains how to finish. Ollama must remain running. No API key is needed. You can disconnect the internet after provisioning. On startup, the browser restores the latest complete local export from `outputs/`.

### Try the interface without downloading AI models

```bash
python3.12 run.py --demo
```

Open the URL and click **Δοκιμή με demo δεδομένα**. This deliberately uses a synthetic portfolio, character TF-IDF/SVD features, and deterministic Greek templates. The real registry still supplies all venues. Demo results are visibly marked and **do not satisfy the semantic embedding/LLM requirements for submission**. Demo mode never silently substitutes for a failed real analysis.

### Sign in, guided tour and help assistant

- **Demo login.** Opening the app shows a sign-in page. Any new username creates a local account on the spot with the password typed; an existing username must use its own password. Passwords are stored as salted `scrypt` hashes in the local SQLite store (`.cache/forcecrm.sqlite`), and the session is an HttpOnly, SameSite=Strict cookie (7 days). There is no external identity provider. Without a session, the API answers 401 and the page redirects to sign-in. Set `FORCECRM_AUTH=0` to turn the login off (the test suite does this).
- **One-click demo sign-in.** The sign-in page also offers «Είσοδος επίδειξης, χωρίς λογαριασμό», for live presentations. It signs into a built-in local user «Επίδειξη» that has no usable password, so the normal form can never open it, and that has the guided tour already done (replay it with «?»). If someone has already registered a real account with that name, a numbered name is used, so a demo sign-in never opens a personal account. Set `FORCECRM_DEMO_LOGIN=0` to hide and disable it.
- **Guided tour.** On a user's first sign-in, an overlay walks through the screens (menu, upload, options, analysis, demo, figures, profiles, prospects, documentation, account). The **?** button in the top bar, or *Επανάληψη ξενάγησης* in the account menu, replays it.
- **Help assistant (Βοηθός).** Available to every user from the bottom-right button. It runs on the **same local Ollama model** as the descriptions (no extra model, no external call) and streams its answer. It answers only from a fixed Greek help text and a read-only summary of the current analysis, and never changes profiles, scores or ranking. Every answer is labelled as AI or as a fixed reply.
  - Questions about solvency or creditworthiness get a fixed refusal without calling the model.
  - **Hand-off to support** doesn't rely on the model: asking for a person gets a fixed reply and the support card; reporting a problem, or repeating the same question, adds the card after the model's answer; the model can also flag a question it can't answer from the help text. The card shows (placeholder) contact details and can save the conversation as a local support ticket (`support_tickets` table).
  - While an analysis is running, the assistant waits instead of loading the model a second time, so the laptop never runs two LLM jobs at once.

### Upload your portfolio

The supplied workbook is `ForceCRM_Hackathon_Pelatologio.xlsx` (300 rows, `ΑΦΜ` and `Τζίρος (€)`). Upload it through the interface. Generic `vat` and `revenue` columns (Greek `ΑΦΜ` and `Έσοδα` also accepted) work too. A blank template can be downloaded in the interface. VATs are preserved as nine-character strings, including leading zeros. Decimal numbers and Greek decimal strings such as `1.234,50` are accepted. Supply one row per VAT; duplicate or unknown VATs, negative/nonfinite revenue, and portfolios with fewer than five customers are rejected with an explanation.

The interface provides profiles, grounded descriptions, statistical tables, stability, a searchable/paginated prospect list, a prospects map over an offline Greece outline, internal holdout results, and downloads. Recommendations are sales research leads, not automatic outreach or credit decisions.

**Περισσότερες επιλογές** (more options) panel:
- **Adjustable similarity weights** (activity / geography / legal form / age). The validated defaults are 65 / 20 / 10 / 5. A custom run shows a warning and re-runs the internal holdout with the default weights, so the accuracy trade-off is visible. Submission exports should use the defaults.
- **Live ΓΕΜΗ status check (opt-in, online).** Checks whether the top N prospects (1–200) are still active in the ΓΕΜΗ Open Data API. It needs your **own** ΓΕΜΗ `api_key`, entered in the form or set as `GEMI_API_KEY`.
  - It runs server-side and sends **only prospect ΑΦΜ**, never the customer list or revenue.
  - It is rate-limited (`GEMI_REQUESTS_PER_MINUTE`, default 8), and results are cached for 7 days in the local SQLite store (`.cache/forcecrm.sqlite`, table `gemi_status`).
  - It writes `live-status.csv` with the source attribution.
  - This is the only feature that contacts an external service. It fetches data, not AI, which the brief allows. It is off unless you start it; everything else works offline.

## Exports and CLI

Each analysis writes `outputs/<run-id>/`:

| File | Fields / purpose |
|---|---|
| `profiles.json` | Array containing `profile_id`, `title`, `description`, `customer_count`, `venue_count`, `revenue_share`, `stability` (+ `stability_detail`), `confidence_score` and its alias `confidence` (the brief's «εμπιστοσύνη»), `quality`, `significant_features` (lift, p-value, q-value), `activity_mix` (top primary activities vs market), `ai_generated`, `description_status`, per-sentence `claims` (source fact + `author`) and `llm` parameters |
| `assignments.csv` | `company_id,profile_id,score`; every venue; rejected matches use literal `NONE` |
| `lookalikes.csv` | `profile_id,company_id,rank,score,rationale`. Globally ranked, one best venue per non-customer VAT. `rationale` is the brief's «αιτιολόγηση», written in plain Greek, e.g. «Κοινά με το ICP-02: Δήμος ΚΑΣΣΑΝΔΡΑΣ · Π.Ε. ΧΑΛΚΙΔΙΚΗΣ · ΙΚΕ. Ομοιότητα 93,3%, συγκέντρωση πελατών 2,54× του αναμενόμενου.» The UI also shows each prospect's rank within its profile (A7). |
| `report.json` | Source hashes, portfolio hash, model/config identity, diagnostics, `portfolio_traits` (business traits over-represented across the whole portfolio), latency, internal evaluation, caveats |

CSV outputs use comma delimiters, dot decimals, UTF-8 BOM, and standard CSV quoting. Input CSVs use semicolons and comma decimals. Rankings include all eligible prospects, therefore exceed 2,000 unless fewer exist. `rank` is **global**; filtering a profile retains that global rank. The nearest profile is reported for every candidate; those outside its radius are explicitly marked **διερευνητική πρόταση** in the rationale, while assignments correctly remain `NONE`. This avoids pretending there are 2,000 confident matches when there are not.

The analysis runs as named stages (`load → enrich → represent → discover → stability → evidence → profiles → rank → narrate → evaluate`), each cached in `.cache/stages/` under a hash of its inputs and code. A repeat run on the same inputs takes about 2 s instead of several minutes; a new customer list reuses the universe-side stages; only changed profiles go back to the LLM. `report.json` lists each stage's status and time under `stages`. See [docs/ARCHITECTURE.md §2.0](docs/ARCHITECTURE.md).

```bash
.venv/bin/python -m scripts.analyze --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --output outputs/real
.venv/bin/python -m scripts.analyze --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --from-stage narrate   # regenerate the AI text only
.venv/bin/python -m scripts.analyze --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --until-stage rank     # stop after ranking (no LLM, no export)
.venv/bin/python -m scripts.analyze --demo --output outputs/demo
.venv/bin/python -m scripts.validate_exports --output outputs/real
.venv/bin/python -m scripts.e2e_server --url http://127.0.0.1:8000 --portfolio ForceCRM_Hackathon_Pelatologio.xlsx   # app must be running
.venv/bin/python -m scripts.audit_ranking --with-tourism   # 5-split recall, matches the shipped pipeline
.venv/bin/python -m scripts.experiment_representation --portfolio ForceCRM_Hackathon_Pelatologio.xlsx
.venv/bin/python -m pytest -q
```

Windows: replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

## Methodology

See [Architecture](docs/ARCHITECTURE.md) for the full pipeline diagram, the ranking formula, why the profiles on this portfolio are mostly geographic (with the experiment that tested alternatives), and why recommendations are deterministic (same inputs give byte-identical `lookalikes.csv`).

1. **Business representation.** Normalize semantic activity vectors, then concatenate weighted blocks: activity 65%, geography 20%, legal form 10%, enterprise age 5%. E5 uses the required `query: ` prefix. Primary descriptions have 85% of the activity contribution; up to three semantically related secondary foodservice codes share 15%. Duplicates are removed, auxiliary/unknown types ignored. A missing secondary block copies the primary representation, so merely registering more codes never incurs a count penalty. More codes can still change business meaning; semantic filtering is a heuristic, not proof that an activity is active.
2. **Spatial features.** Fixed geographic radial basis anchors across Greece (130 km scale) plus prefecture. GPS missingness is encoded explicitly. Age uses the fixed reference date `2026-11-01`; no moving wall-clock features. This supplies geographic ICP features without external calls or map tiles. [Venue and market enrichment](docs/ENRICHMENT.md) runs offline using the bundled OSM and ELSTAT snapshots.
3. **Revenue and branches.** Use square-root revenue weights capped at the portfolio's 95th percentile, normalized and divided by venue count per VAT. Raw revenue, also divided across venues, determines profile revenue shares. Shares sum to one even when a chain spans profiles. Customer counts are unique within each profile, so their sum can exceed the portfolio count when a VAT has venues in multiple profiles.
4. **Profile discovery.** Deterministic weighted spherical k-means, 1–6 candidates, six seeded starts. A split needs at least five VATs per cluster and cosine silhouette minus `0.025 × (k−1)` greater than `0.18`. Otherwise return one profile. This is a conservative selection heuristic that should be tuned only on development splits.
5. **Assignment and rejection.** Fit and assign with the same cosine distance. A profile accepts venues within its customer-distance 95th percentile plus 0.015 (minimum radius 0.015). Outliers are `NONE`; cluster membership used to estimate profiles still includes all portfolio venues. Assignment score is cosine similarity (clipped to [0,1]); profile confidence is mean `exp(−distance/radius)`, a relative fit measure, **not a calibrated probability**. A held-out confidence calibration would be an improvement.
6. **Statistical evidence.** One-sided Fisher exact tests compare VAT-level presence against remaining universe VATs. Benjamini–Hochberg is applied across all profile/trait tests together. Require q≤0.05, ≥5 supporting VATs, lift≥1.2. Lift compares profile prevalence with universe prevalence. Statistical tables are **exploratory**: clustering and trait discovery use the same data, so adjusted p-values do not establish independent confirmatory evidence. Because profiles on this portfolio split mainly by place, the same test is also run on the whole portfolio against the universe, restricted to business traits (activity, legal form, incorporation year, branch, OSM amenity). This gives `portfolio_traits` in `report.json`, e.g. ΙΚΕ 2.4× and school canteens 2.4×. Each profile also reports a descriptive `activity_mix`: its top three primary activities against market shares.
7. **Stability.** Twenty seeded VAT bootstraps, refitting fixed k, Hungarian matching and per-profile VAT-membership Jaccard (mean and 10th percentile). Global venue-level ARI is also reported. This measures membership stability conditional on k, not stability of k itself. A single homogeneous profile naturally has high membership stability.
8. **Grounded Greek narration.** Each profile fact (size, revenue share and up to four defining traits) is first rendered as a correct Greek sentence by a template. Trait selection avoids restating the same place twice and always keeps the strongest activity or legal-form trait. Ministral then rewrites each sentence separately (Ollama chat, fixed system prompt, five fictional few-shot examples, JSON schema). Every rewrite must pass deterministic checks. Numbers and percentages must be a subset of the source fact, and the defining count must survive. Place, activity and legal-form names must survive accent-insensitively. Hedging and misleading wording observed on challenge data is rejected (for example «άτομα», «τόπος καταγωγής», «κυριαρχεί», «περίπου», wrong case or gender). Duplicates and markdown are also rejected. A sentence that fails twice keeps its verified template wording. Each claim records `author` (`local LLM` or `computed template`) and its source field. `description_status` states how many sentences the LLM wrote. Every description ends with two computed sentences: the profile's most common primary activities against the market, and profile quality, bootstrap stability (mean and 10th percentile) and fit confidence, explicitly not a purchase probability (A6). Settings: temperature 0, seed 42, top-k 1, context 2048. The model digest, prompt hash and rejected drafts are retained. On an 8 GB Apple-silicon Mac this takes 25–50 s per profile (6–9 LLM calls, plus one title call).
9. **Retrieval and evaluation.** Exclude all venues belonging to any customer VAT. Rank each remaining VAT by its best venue's concentration-adjusted cosine score, deterministic company-ID tie-break. Internal evaluation removes 20% of VATs **before fitting profiles or revenue weights**, then reports Recall@500/2000 versus the primary-code-frequency baseline. Feature embeddings are unsupervised and use the known universe. This is an internal development split, not the organizer's hidden evaluation; do not optimize repeatedly against it.

## Model comparison

On an Apple A18 Pro with 8 GiB RAM, all three tested LLMs produced correct Greek descriptions for all three profiles. The share of fact sentences each model wrote that passed every grounding check (the rest keep their verified template) was 61% for Ministral 3 3B, 100% for Qwen3 1.7B and 94% for Gemma 3 1B. Mean time per profile was 50.0 s, 32.2 s and 16.8 s. Ministral remains the default for its European provenance and Apache-licensed distribution. Fluency still requires independent human review. Five embedding models were compared, including two from the brief's list (IBM granite multilingual r2 311M and BGE-M3). On the real task, 5-split holdout recall with only the embedding swapped (`scripts.benchmark embedding-recall`), none beat E5-small (21.3% / 51.3%). BGE-M3 scored best on the code-family proxy (95.9% vs 93.2%) but costs about 10× the time and 4.7× the disk. The printable [Model & Data Card](docs/ForceCRM-Model-Data-Card.pdf) records the evaluation scope and costs; rebuild it with `.venv/bin/python -m scripts.build_card` after `pip install reportlab` (Windows, macOS and Linux fonts are supported). Reproduce the comparisons with the following commands.

```bash
ollama pull qwen3:1.7b
ollama pull gemma3:1b
.venv/bin/python -m scripts.benchmark llms --profiles outputs/real/profiles.json --output outputs/llm-comparison.json
.venv/bin/python -m scripts.download_models --model intfloat/multilingual-e5-base --destination models/e5-base
.venv/bin/python -m scripts.download_models --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --destination models/multilingual-minilm
.venv/bin/python -m scripts.download_models --model ibm-granite/granite-embedding-311m-multilingual-r2 --destination models/granite-311m-r2
.venv/bin/python -m scripts.download_models --model BAAI/bge-m3 --destination models/bge-m3   # its weights are pytorch_model.bin; fetch that file too (see CLAUDE.md)
.venv/bin/python -m scripts.benchmark embeddings --embedding-paths models/e5-small models/e5-base models/multilingual-minilm models/granite-311m-r2 models/bge-m3 --output outputs/embedding-comparison.json
.venv/bin/python -m scripts.benchmark embedding-recall --portfolio ForceCRM_Hackathon_Pelatologio.xlsx --embedding-paths models/e5-small models/e5-base models/multilingual-minilm models/granite-311m-r2 models/bge-m3 --output outputs/embedding-recall.json
```

For MiniLM comparisons the benchmark uses unprefixed text; E5 uses `query:`. The embedding benchmark measures recovery of Greek activity-code families, a weak-label proxy. Add human-labeled Greek relevance judgments before presenting it as semantic quality. For LLMs, evaluate schema success, original generation latency and blind human fluency/coherence ratings (null until reviewed). Verify each alternative's actual license before deployment; Gemma uses custom terms. Missing models produce explicit unavailable entries. Do not invent missing scores or fill human ratings automatically. Hardware timing excludes downloads; application RAM reporting is process RSS at finish, not total system peak including Ollama.

## Google Colab (temporary experimentation)

Open [notebooks/ForceCRM_Colab.ipynb](notebooks/ForceCRM_Colab.ipynb) in Colab and follow [the step-by-step guide](docs/COLAB.md). It uploads your source/data ZIP into the Colab VM, downloads weights, starts Ollama inside that VM, and runs analysis directly in notebook cells. **Colab is cloud computation, so this does not satisfy the final local-laptop requirement.** No external AI inference API is used, but uploaded competition/customer data resides on Google's infrastructure. Only upload data you are permitted to put there. The final submission is this localhost app.

## Data, responsible AI and disclosure

Sources: **ΓΕΜΗ (GEMI) Open Data** and **Geocoding powered by OpenStreetMap contributors (ODbL)**. Bundled enrichment: OpenStreetMap via Geofabrik (ODbL), ELSTAT hotel nights and census, and the Natural Earth map outline (public domain). Optional live check: ΓΕΜΗ Open Data API (ODC-BY 1.0, © Κεντρική Υπηρεσία ΓΕΜΗ / ΚΕΕΕ; see [ENRICHMENT.md](docs/ENRICHMENT.md)). Supplied data is restricted to this competition; do not publish the CSVs or customer portfolios. `.gitignore` excludes them, models, caches and outputs; inspect staged files before publishing.

[Responsible AI self-assessment](docs/RESPONSIBLE_AI.md) records scope, impacts, visible evidence and failure modes. All model-composed text is labeled. No credit/solvency scores, protected characteristics or automated contact are generated. The backend binds to loopback, checks Host, requires a signed-in session for the API (local demo accounts), requires a same-origin custom mutation header, uses no CORS wildcard, disables external embedding fetches during analysis, and restricts Ollama to literal `127.0.0.1` with proxy inheritance disabled. The browser loads no CDN assets or map tiles. The only outbound runtime call is the opt-in ΓΕΜΗ live check described above.

AI coding assistant disclosure (development only, not part of application inference):
- **OpenAI Codex**: architecture, implementation, documentation, tests and debugging.
- **Claude Code (Anthropic)**, used by several team members: the SQLite store, the stage pipeline, and the login, onboarding tour and help chat; also code review against the brief, holdout-recall verification, clean-install/setup fixes, the per-sentence Greek narration with grounding checks and the stated profile quality/stability/confidence, the demo login, guided tour and help assistant, and documentation updates.

## Troubleshooting

- `Python 3.12 or 3.13 is required`: the launcher itself may be any Python 3; it builds `.venv` with `python3.12`/`python3.13` from PATH, or a uv-managed 3.12 if `uv` is installed. If neither exists, install Python 3.12 (python.org, `brew install python@3.12`, or `uv python install 3.12`). If an older `.venv` was created with another version, delete `.venv` and relaunch.
- `python` opens the Microsoft Store: install Python 3.12 and enable PATH, or use `py -3.12`.
- `FileNotFoundError` for a `vwHack*.csv` file: copy the three CSVs into the repository root (step 1).
- Missing embeddings: `python run.py --setup-models`; model downloads are explicit, never automatic during analysis.
- Ollama connection refused: start the Ollama app, or run `ollama serve` in another terminal. Ensure `ollama list` contains the model named by `DEFAULT_LLM` in `forcecrm/narrative.py` (currently `ministral-3:3b`).
- First analysis slow: activity embeddings are computed once per exact model/text fingerprint. The LLM runs only per profile, not for 24,424 businesses.
- Occupied port: `python run.py --port 8001`.
- Update data: keep filenames/schema unchanged. Changed source/model files invalidate embedding hashes. A fresh run exports separate results.
- Exports remain in `outputs/`. After restart, the browser restores the latest complete local analysis.

See [competition verification](docs/COMPETITION_AUDIT.md) for paired evaluation and the requirement audit.
