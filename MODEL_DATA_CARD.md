# 📄 Model & Data Card — ForceCRM B2B Profile Analysis & Lookalike System

**Project:** Archimedes2026 (NKUA Hackathon · Nov 2026)  
**Deliverable:** Section 10 & Section 5.3 — Model & Data Card

---

## 1. System Purpose & Architecture Summary

**Archimedes2026** is a 100% locally executed B2B Customer Segmentation (Ideal Customer Profile discovery) and Lookalike Recommendation system for Greek food & beverage / hospitality businesses (Main KAD `56`, founded `2020–2024`, $N = 24,424$ companies, $N_c = 300$ clients).

To guarantee 100% deterministic reproducibility and zero hallucination risk, the system separates mathematical discovery from narrative generation:
1. **Deterministic Hybrid Statistical Core (`src/ingest.py`, `src/features.py`, `src/icp_engine.py`):** Performs multi-branch VAT deduplication, Hybrid KAD representation (Neural CPU embeddings + exact ARPACK SVD over Greek character/word TF-IDF and KAD-by-KAD PPMI co-occurrence), static Parquet pre-caching (`data/cached_*.parquet`), 3D spherical spatial density indexing, deterministic turnover-weighted K-Means (`init="k-means++", n_init=50, random_state=42`), Bootstrap ARI stability validation, Hypergeometric FDR + Empirical Bayes Shrinkage Lift calculation, and unified lookalike/assignment scoring.
2. **Grounded Local AI Explainer (`src/llm_explainer.py`):** Uses a local open-weights LLM via Ollama (`temperature=0.0, seed=2026`) solely to translate the verified statistical JSON of each ICP into an executive Greek narrative, backed by a zero-latency deterministic Greek fallback when Ollama is offline.

---

## 2. Model Selection & Four-Axis Justification (Section 5.3)

### 2.1 Hybrid Greek KAD Representation (`src/features.py`)

We encode the 3,736 unique Greek KAD descriptions in `vwHackActivities.csv` using a tunable hybrid representation $\mathbf{e}_k^{\text{hybrid}} = [\sqrt{\alpha}\,\mathbf{e}_k^{\text{neural}}, \sqrt{1-\alpha}\,\mathbf{e}_k^{\text{svd}}]$ and evaluate candidate models across the **four mandatory axes of Section 5.3** using 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)` in `evaluate_holdout.py`):

| Axis | Signal B (`char-ngram-svd` + PPMI ARPACK SVD) | Signal A Option 1: `ibm-granite/granite-embedding-107m-multilingual` | Signal A Option 2: `intfloat/multilingual-e5-base` |
| :--- | :--- | :--- | :--- |
| **1. Greek Quality on Challenge Data** (`5-Fold CV Recall@500 / @2000` vs `1.69% / 6.78%` Baseline) | **`Recall@500`: `14.00% ± 3.43%` (+8.2x)**<br>**`Recall@2000`: `30.67% ± 5.01%` (+4.5x)**<br>Captures Greek morphological roots (`char_wb` 3–5 n-grams), exact 2/4/6/8-digit NACE hierarchy, and empirical KAD-by-KAD PPMI co-occurrence. | **High cross-synonym semantic alignment** across Greek hospitality descriptions (e.g., `Υπηρεσίες παροχής γευμάτων` ↔ `Εστιατόρια`) with `torch.use_deterministic_algorithms(True)`. | Strong multilingual dense retrieval embedding; uses `"passage: "` prefix formatting on CPU. |
| **2. License & Commercial Use** | **BSD / MIT** (`scikit-learn` / `scipy`). **100% Commercial Use Permitted.** | **Apache-2.0 License** (IBM Research). **100% Commercial Use Permitted.** | **MIT License** (Microsoft Research). **100% Commercial Use Permitted.** |
| **3. Origin & Digital Sovereignty** | **100% On-Device Algebraic Engine** (Zero external weights; fitted directly on the official ΓΕΜΗ / ELSTAT Greek KAD taxonomy). | **IBM Research** (Open-source Apache-2.0 release with auditable enterprise training governance). | **Microsoft Research** (Open-weights MIT release). |
| **4. Execution Cost (Ryzen 2600X, 32GB DDR4, CPU)** | **Cold Build:** `~4.5s` / **Warm Parquet Load:** `<0.15s`<br>**Peak RAM:** `~135 MB`<br>**Cache Size:** `~4.2 MB` Parquet | **Cold Build:** `~11.5s` (3,736 KADs) / **Warm Load:** `<0.15s`<br>**Peak RAM:** `~480 MB` (`107M` params) | **Cold Build:** `~28.5s` (3,736 KADs) / **Warm Load:** `<0.15s`<br>**Peak RAM:** `~1.1 GB` (`278M` params) |

*Static Pre-Caching Design:* Because `vwHackActivities.csv` contains only **3,736 unique KADs** (linked 161,964 times in `vwHackCompanyActivity.csv`), we compute the KAD vectors once, store them in `data/cached_kad_vectors.parquet`, and persist the 24,424 Universe feature matrix in `data/cached_universe_features.parquet`.

---

### 2.2 Local Generative LLM for Greek Profile Narratives (`src/llm_explainer.py`)

- **Supported Sovereign & Open Models (via Ollama):**
  - **Greek / European Sovereign Models:** `ilsp/llama-krikri-8b-instruct` (Greece — Athena RC / ILSP), `utter-project/EuroLLM-9B-Instruct` (EU-funded, Apache-2.0), `BSC-LT/salamandra-7b-instruct` (Spain — Barcelona Supercomputing Center, Apache-2.0), `mistralai/Ministral-3-3B-Instruct` (France, Apache-2.0).
  - **Local Multilingual Default & Offline Fallback:** `qwen2.5:7b-instruct` (Apache-2.0) + a built-in **Deterministic Grounded Greek Narrative Engine** (`<0.001s`, zero RAM overhead) when Ollama is offline.
- **Groundedness Enforcement (Requirement A6):**
  - Locked to `temperature = 0.0`, `seed = 2026`, `top_p = 1.0`.
  - Prompts receive only verified statistical metrics (`client_count`, `turnover_share`, `stability_score`, `confidence_score`, and FDR-significant traits) and cannot invent numbers or assess creditworthiness.

---

## 3. Data Card: Sources, Licenses & Preprocessing

| Dataset | Records | Origin & License | Preprocessing & Quality Safeguards (`src/ingest.py`) |
| :--- | :---: | :--- | :--- |
| `vwHackCompany.csv` | 24,424 companies (23,945 unique VATs) | **ΓΕΜΗ Open Data** & **© OpenStreetMap contributors (ODbL)** for GPS coordinates (96.01% coverage). | Strip UTF-8 BOM; zero-pad 9-digit `vat`; parse comma-decimal GPS floats; compute `company_age_years` and `branch_count` per VAT. |
| `vwHackCompanyActivity.csv` | 161,964 links | **ΓΕΜΗ Open Data** (NACE 2.1 / KAD 2025 8-digit codes). | Normalize 1 `Λοιπή` row to `Βοηθητική`; resolve the 993 companies with multiple `Κύρια` KADs (`2026-03-01` NACE 2.1 transition) by selecting the earliest `56*` hospitality KAD as `primary_kad` while keeping all KADs in `all_kads`. |
| `vwHackActivities.csv` | 3,736 unique KADs | **ΓΕΜΗ / ELSTAT KAD 2025 Taxonomy**. | Clean whitespace, zero-pad to 8 digits, and extract 2/4/6/8-digit NACE hierarchy tokens. |
| `ForceCRM_Hackathon_Pelatologio.xlsx` | 300 B2B clients | **ForceCRM Challenge Dataset** (`turnover` €1,360–€87,270). | Zero-pad `ΑΦΜ` to 9 digits; flag all 361 rows matching any client VAT (`is_client_vat = True`) so zero client branches ever leak into `lookalikes.csv`. |

---

## 4. Known Limitations, Failure Modes & Mitigations

1. **Noisy / Declarative Secondary KADs (up to 239 KADs on one company):**
   - *Mitigation:* Smooth IDF weighting ($\gamma_{\text{idf}}=1.25$), hierarchy weights (`3.50` Primary vs `0.75` Secondary / `0.25` Auxiliary), sublinear secondary damping ($n_{\text{sec}}^{-0.35}$), and L2 unit normalization ensure multi-KAD businesses are neither distorted by noise nor penalized in cosine similarity.
2. **Missing GPS Coordinates (3.99% / 975 records):**
   - *Mitigation:* Median spatial density imputation and neutral (`0.50`) spatial kernel score, preserving full Prefecture/Municipality Empirical Bayes Lift scoring.
3. **Small-Sample Lift Inflation on Rare Municipalities / KADs:**
   - *Mitigation:* Exact **Hypergeometric tests** with **Benjamini-Hochberg FDR correction** ($p_{\text{FDR}} \le 0.05$) and **Empirical Bayes Shrinkage** ($m_{\text{prior}} = 12.0$) toward the Universe baseline rate.
4. **Atypical / Outlier Clients & EU AI Act Safeguards:**
   - *Mitigation:* Clients falling below the dynamic similarity floor (`0.35`) are explicitly labeled **`"NO_FIT"`** in `assignments.csv`. All scores strictly measure B2B Commercial Fit and never evaluate personal creditworthiness or financial solvency (`φερεγγυότητα`).
