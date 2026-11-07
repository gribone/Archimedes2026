# ForceCRM · Local Customer Intelligence

A localhost application for discovering 1–6 ideal customer profiles in Greek HORECA businesses, explaining them in Greek, and ranking new prospects. FastAPI serves a dependency-free Greek browser interface; NumPy/scikit-learn perform analysis; E5 embeds Greek activity descriptions; Ollama runs Ministral locally.

**This workspace is provisioned.** From this directory, run `.\.venv\Scripts\python.exe run.py` and open http://127.0.0.1:8000. The launcher finds the project-local Ollama executable under `.tools/ollama/`. The grading files for the 300-customer workbook are in `outputs/real/`. On startup, the browser restores the latest complete local export.

The supplied portfolio yielded 3 profiles, 24,424 venue assignments and 23,645 distinct prospective VATs. On a seeded internal split of 240 training and 60 held-out customers, ForceCRM retrieved 9/60 at 500 and 20/60 at 2,000. The primary-code baseline retrieved 1/60 and 5/60. The organizer scores its own hidden VAT list separately.

## Run locally

Prerequisites: **Python 3.11 or 3.12**, current [Ollama](https://ollama.com/download) (at least 0.13.1), roughly 8 GB free disk space, and ideally 16 GB RAM. Python must be installed with pip. Place the three supplied CSVs in this directory. They are already present in this workspace.

One-command launch from a clean Python environment:

```powershell
python run.py --first-run
```

The launcher creates `.venv`, installs dependencies, downloads both models and starts **http://127.0.0.1:8000**. With the Windows Python Launcher, replace `python` with `py -3.12`. Subsequent launches use `python run.py` and work offline.

To prepare the models separately and launch later, use these commands instead:

```powershell
python run.py --setup-models
python run.py
```

This downloads `intfloat/multilingual-e5-small` to `models/e5-small`, records its exact revision, and runs `ollama pull ministral-3:3b` (Q4_K_M, approximately 3 GB). If Ollama is not installed, the setup command explains how to finish. Ollama must remain running. No API key is needed. You can disconnect the internet after provisioning. On the current workspace, the created environment can also be used directly:

```powershell
.\.venv\Scripts\python.exe run.py --demo
```

### Try the interface without downloading AI models

```powershell
python run.py --demo
```

Open the URL and click **Δοκιμή με demo δεδομένα**. This deliberately uses a synthetic portfolio, character TF-IDF/SVD features, and deterministic Greek templates. The real registry still supplies all venues. Demo results are visibly marked and **do not satisfy the semantic embedding/LLM requirements for submission**. Demo mode never silently substitutes for a failed real analysis.

### Upload your portfolio

The supplied workbook is `ForceCRM_Hackathon_Pelatologio.xlsx` (300 rows, `ΑΦΜ` and `Τζίρος (€)`). Upload it through the interface. Generic `vat` and `revenue` columns (Greek `ΑΦΜ` and `Έσοδα` also accepted) work too. A blank template can be downloaded in the interface. VATs are preserved as nine-character strings, including leading zeros. Decimal numbers and Greek decimal strings such as `1.234,50` are accepted. Supply one row per VAT; duplicate or unknown VATs, negative/nonfinite revenue, and portfolios with fewer than five customers are rejected with an explanation.

The interface provides profiles, grounded descriptions, statistical tables, stability, a searchable/paginated prospect list, an offline GPS scatter plot, internal holdout results, and downloads. Recommendations are sales research leads, not automatic outreach or credit decisions.

## Exports and CLI

Each analysis writes `outputs/<run-id>/`:

| File | Fields / purpose |
|---|---|
| `profiles.json` | Array containing `profile_id`, `title`, `description`, `customer_count`, `revenue_share`, `stability`, `confidence_score`, `significant_features`, AI labels and sentence evidence |
| `assignments.csv` | `company_id,profile_id,score`; every venue; rejected matches use literal `NONE` |
| `lookalikes.csv` | `profile_id,company_id,rank,score,rationale`; globally ranked, one best venue per noncustomer VAT |
| `report.json` | Source hashes, portfolio hash, model/config identity, diagnostics, latency, internal evaluation, caveats |

CSV outputs use comma delimiters, dot decimals, UTF-8 BOM, and standard CSV quoting. Input CSVs use semicolons and comma decimals. Rankings include all eligible prospects, therefore exceed 2,000 unless fewer exist. `rank` is **global**; filtering a profile retains that global rank. The nearest profile is reported for every candidate; those outside its radius are explicitly marked **διερευνητική πρόταση** in the rationale, while assignments correctly remain `NONE`. This avoids pretending there are 2,000 confident matches when there are not.

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze --portfolio customers.xlsx --output outputs/real
.\.venv\Scripts\python.exe -m scripts.analyze --demo --output outputs/demo
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS: replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Methodology

1. **Business representation.** Normalize semantic activity vectors, then concatenate weighted blocks: activity 65%, geography 20%, legal form 10%, enterprise age 5%. E5 uses the required `query: ` prefix. Primary descriptions have 85% of the activity contribution; up to three semantically related secondary foodservice codes share 15%. Duplicates are removed, auxiliary/unknown types ignored. A missing secondary block copies the primary representation, so merely registering more codes never incurs a count penalty. More codes can still change business meaning; semantic filtering is a heuristic, not proof that an activity is active.
2. **Spatial features.** Fixed geographic radial basis anchors across Greece (130 km scale) plus prefecture. GPS missingness is encoded explicitly. Age uses the fixed reference date `2026-11-01`; no moving wall-clock features. This supplies geographic ICP features without external calls or map tiles. Enrichment EP2/EP3 is not implemented or claimed.
3. **Revenue and branches.** Use square-root revenue weights capped at the portfolio's 95th percentile, normalized and divided by venue count per VAT. Raw revenue, also divided across venues, determines profile revenue shares. Shares sum to one even when a chain spans profiles. Customer counts are unique within each profile, so their sum can exceed the portfolio count when a VAT has venues in multiple profiles.
4. **Profile discovery.** Deterministic weighted spherical k-means, 1–6 candidates, six seeded starts. A split needs at least five VATs per cluster and cosine silhouette minus `0.025 × (k−1)` greater than `0.18`. Otherwise return one profile. This is a conservative selection heuristic that should be tuned only on development splits.
5. **Assignment and rejection.** Fit and assign with the same cosine distance. A profile accepts venues within its customer-distance 95th percentile plus 0.015 (minimum radius 0.015). Outliers are `NONE`; cluster membership used to estimate profiles still includes all portfolio venues. Export score is cosine similarity (clipped to [0,1]); profile confidence is mean `exp(−distance/radius)`, a relative fit measure, **not a calibrated probability**. A held-out confidence calibration would be an improvement.
6. **Statistical evidence.** One-sided Fisher exact tests compare VAT-level presence against remaining universe VATs. Benjamini–Hochberg is applied across all profile/trait tests together. Require q≤0.05, ≥5 supporting VATs, lift≥1.2. Lift compares profile prevalence with universe prevalence. Statistical tables are **exploratory**: clustering and trait discovery use the same data, so adjusted p-values do not establish independent confirmatory evidence.
7. **Stability.** Twenty seeded VAT bootstraps, refitting fixed k, Hungarian matching and per-profile VAT-membership Jaccard (mean and 10th percentile). Global venue-level ARI is also reported. This measures membership stability conditional on k, not stability of k itself. A single homogeneous profile naturally has high membership stability.
8. **Grounded Greek narration.** Ministral selects/orders evidence IDs and one of two Greek sentence variants. A validator rejects unsupported IDs, repeated facts, malformed JSON, and omitted required facts. Arbitrary model prose is never displayed. This constrained drafting approach sacrifices free-form language for strict sentence-level grounding. Invalid or unavailable LLM output stops real analysis with an explicit error. Temperature 0, seed 42, top-k 1, context 2048; model digest and raw output retained. Hardware kernels can still differ; cached digest+prompt results are exactly replayable.
9. **Retrieval and evaluation.** Exclude all venues belonging to any customer VAT. Rank each remaining VAT by its best venue's nearest-profile similarity, deterministic company-ID tie-break. Internal evaluation removes 20% of VATs **before fitting profiles or revenue weights**, then reports Recall@500/2000 versus the primary-code-frequency baseline. Feature embeddings are unsupervised and use the known universe. This is an internal development split, not the organizer's hidden evaluation; do not optimize repeatedly against it.

## Model comparison

On this laptop, all three tested LLMs produced grounded drafts for all three profiles. Mean generation times were 176.4 seconds for Ministral 3 3B, 30.1 seconds for Qwen3 1.7B and 19.3 seconds for Gemma 3 1B. Ministral remains the default for its European provenance and Apache-licensed distribution. Fluency still requires independent human review. E5-small tied E5-base on Greek activity-family hit@5 at 93.2% and took 59.5 ms per description versus 195.5 ms. The printable [Model & Data Card](docs/ForceCRM-Model-Data-Card.pdf) records the evaluation scope and costs. Reproduce the comparisons with the following commands. On this prepared Windows machine the Ollama CLI is also available at `.tools/ollama/ollama.exe`.

```powershell
ollama pull qwen3:1.7b
ollama pull gemma3:1b
.\.venv\Scripts\python.exe -m scripts.benchmark llms --profiles outputs/real/profiles.json --output outputs/llm-comparison.json
.\.venv\Scripts\python.exe -m scripts.download_models --model intfloat/multilingual-e5-base --destination models/e5-base
.\.venv\Scripts\python.exe -m scripts.download_models --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --destination models/multilingual-minilm
.\.venv\Scripts\python.exe -m scripts.benchmark embeddings --output outputs/embedding-comparison.json
```

For MiniLM comparisons the benchmark uses unprefixed text; E5 uses `query:`. The embedding benchmark measures recovery of Greek activity-code families, a weak-label proxy. Add human-labeled Greek relevance judgments before presenting it as semantic quality. For LLMs, evaluate schema success, original generation latency and blind human fluency/coherence ratings (null until reviewed). Verify each alternative's actual license before deployment; Gemma uses custom terms. Missing models produce explicit unavailable entries. Do not invent missing scores or fill human ratings automatically. Hardware timing excludes downloads; application RAM reporting is process RSS at finish, not total system peak including Ollama.

## Google Colab (temporary experimentation)

Open [notebooks/ForceCRM_Colab.ipynb](notebooks/ForceCRM_Colab.ipynb) in Colab and follow [the step-by-step guide](docs/COLAB.md). It uploads your source/data ZIP into the Colab VM, downloads weights, starts Ollama inside that VM, and runs analysis directly in notebook cells. **Colab is cloud computation, so this does not satisfy the final local-laptop requirement.** No external AI inference API is used, but uploaded competition/customer data resides on Google's infrastructure. Only upload data you are permitted to put there. The final submission is this localhost app.

## Data, responsible AI and disclosure

Sources: **GEMI Open Data** and **Geocoding powered by OpenStreetMap contributors (ODbL)**. Supplied data is restricted to this competition; do not publish the CSVs or customer portfolios. `.gitignore` excludes them, models, caches and outputs. A local Git repository is initialized. Data, model weights and outputs are ignored; inspect staged files before publishing.

[Responsible AI self-assessment](docs/RESPONSIBLE_AI.md) records scope, impacts, visible evidence and failure modes. All model-composed text is labeled. No credit/solvency scores, protected characteristics or automated contact are generated. The backend binds to loopback, checks Host, requires a same-origin custom mutation header, uses no CORS wildcard, disables external embedding fetches during analysis, and restricts Ollama to literal `127.0.0.1` with proxy inheritance disabled. The browser loads no CDN assets or map tiles.

AI coding assistant disclosure: **OpenAI Codex** was used for architecture, implementation, documentation, tests and debugging. It is a development tool, not part of application inference.

## Troubleshooting

- `python` opens the Microsoft Store: install Python 3.12 and enable PATH, or use `py -3.12`.
- Missing embeddings: `python run.py --setup-models`; model downloads are explicit, never automatic during analysis.
- Ollama connection refused: start the Ollama app, or run `ollama serve` in another terminal. Ensure `ollama list` contains `ministral-3:3b`.
- First analysis slow: activity embeddings are computed once per exact model/text fingerprint. The LLM runs only per profile, not for 24,424 businesses.
- Occupied port: `python run.py --port 8001`.
- Update data: keep filenames/schema unchanged. Changed source/model files invalidate embedding hashes. A fresh run exports separate results.
- Exports remain in `outputs/`. After restart, the browser restores the latest complete local analysis.
