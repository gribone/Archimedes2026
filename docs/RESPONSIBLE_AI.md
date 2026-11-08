# EU AI Act self-assessment (1 page)

ForceCRM · local B2B prospect research for Greek food-service businesses · 8 November 2026. This is an engineering self-assessment following the competition brief §9; it is not a legal opinion.

**Classification.** Clustering businesses for B2B sales is **not an Annex III high-risk use** of Regulation (EU) 2024/1689.
- **The trap:** Annex III point 5(b) makes creditworthiness evaluation or credit scoring of **natural persons** high-risk, and sole proprietorships (ΑΤΟΜΙΚΗ) are natural persons. ForceCRM therefore **never produces a solvency, credit or payment-risk score**.
- Its scores measure only similarity to the user's existing customers. Revenue is used only to weight the user's own customers, never to score prospects.
- Repurposing the system for credit, eligibility or pricing decisions would require a new assessment.
- Per the brief, Annex III obligations were postponed to **2 December 2027** by **Regulation (EU) 2026/1744 (Digital Omnibus on AI)**.

## 1. What does the system decide or suggest?
- **Suggests only, decides nothing.** It groups the uploaded customers into 1–6 profiles and ranks non-customer businesses by similarity, each with a short justification.
- **A person acts.** A salesperson decides whether to research or contact a business. Nothing is contacted, priced or rejected automatically.

## 2. Who is affected?
- **The users:** sales staff and the uploading company.
- **The prospects** listed. Sole proprietors among them may be identifiable natural persons.
- **Data minimisation:**
  - The upload is limited to ΑΦΜ and revenue, and everything runs on the laptop.
  - The optional live ΓΕΜΗ check sends only prospect ΑΦΜ (public-registry identifiers) with the user's own key, never the customer list or revenue.
  - Run folders and caches stay local and can be deleted by the user.
  - Demo accounts (username, scrypt password hash) and support tickets are stored only in the local SQLite file.

## 3. What does the user see about how a result arose?
- **Transparency (Art. 50, applicable from 2 August 2026):** every AI-written text is labelled.
  - In the UI, each AI text carries an «AI · ΤΟΠΙΚΟ ΜΟΝΤΕΛΟ» tag.
  - In `profiles.json`: `ai_generated`, `description_status` and `title_source`. Each sentence also records its `author` (local LLM or verified template) and its source fact.
  - The in-app help assistant labels each answer "AI · τοπικό μοντέλο" or "σταθερή απάντηση", and its header warns that answers can contain mistakes.
- **Evidence behind every profile:** counts, revenue share, traits with lift, p-value and q-value, bootstrap stability, fit confidence and the activity mix against the market.
- **Prospect justifications** name the shared traits and the similarity figures. Prospects outside a profile's acceptance radius are marked «Διερευνητική πρόταση».
- **Reproducibility:** generation settings are fixed (temperature 0, seed 42), and model identity, prompt hash and rejected drafts are recorded.

## 4. Where can it be wrong, and how do we detect it?
| Risk | Detection / mitigation |
|---|---|
| Self-declared ΚΑΔ that aren't really practised | Primary activity ≥ 80% of the activity vector; only related secondaries kept, trusted less the more a business declares, ignored above 20; a human verifies before contact |
| Invented or wrong text | One sentence per fact; numbers, %, names and wording checked; a failing sentence keeps its verified template |
| Traits passing by chance | VAT-level Fisher tests, global Benjamini–Hochberg correction, ≥ 5 ΑΦΜ, lift ≥ 1.2; labelled exploratory |
| Unstable profiles | Per-profile bootstrap Jaccard (mean and worst 10%), ARI; "limited quality" label |
| Branch leakage or double counting | Counts, holdout and exclusion all per ΑΦΜ; revenue split across branches |
| Poor ranking | Internal holdout vs the primary-ΚΑΔ baseline (Recall@500/2000, 5 splits) |
| Over-reliance on scores | Scores are uncalibrated similarity, stated in the UI and description: "not a purchase probability" |
| Help assistant gives wrong guidance | Answers only from a fixed help text and read-only analysis facts; credit/solvency questions refused without the model; rule-based hand-off to human support (request for a person, problem report, repeated question) plus a permanent support button |
| Geographic bias | Profiles that are mainly geographic are documented as such; business traits shared across regions are shown separately |

**AI literacy (Art. 4).** The UI section "Πώς λειτουργεί — και πού χρειάζεται προσοχή", the [architecture](ARCHITECTURE.md) and the [model & data card](MODEL_DATA_CARD.md) explain the inputs, weights, statistics, model limits and failure modes, so both builders and users know what the system does and where it fails.

**Local boundary.**
- Serving is loopback-only, and models run locally (Ollama on 127.0.0.1, embeddings offline). No AI service is called at runtime.
- Setup downloads models once.
- The only runtime network call is the opt-in ΓΕΜΗ registry-status check, which fetches data, never AI. Its result never changes scores or ranking.
