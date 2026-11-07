# 📄 Model & Data Card — ForceCRM B2B Profile Analysis & Lookalike System

**Project:** Archimedes2026 (NKUA Hackathon · Nov 2026)  
**Deliverable:** Section 10 & Section 5.3 — Model & Data Card (1–2 Pages)

---

## 1. System Purpose & Architecture Summary

Archimedes2026 is a 100% locally executed B2B Customer Segmentation (Ideal Customer Profile discovery) and Lookalike Recommendation system tailored to Greek food & beverage / hospitality businesses (Main KAD 56, founded 2020–2024, $N = 24,424$ businesses).

To ensure statistical rigor, full reproducibility, and zero hallucination risk, the architecture enforces a strict separation of concerns:
1. **Deterministic Statistical & Geometric Core (`src/ingest.py`, `src/features.py`, `src/icp_engine.py`):** Performs all data cleaning, multi-branch VAT resolution, KAD embedding aggregation, 3D spherical `cKDTree` spatial density computation, turnover-weighted K-Means clustering ($K \in [1, 6]$), Bootstrap ARI stability validation, Hypergeometric + Benjamini-Hochberg FDR + Empirical Bayes Shrinkage Lift calculation, and unified lookalike/assignment scoring.
2. **Local Generative AI Explainer (`src/llm_explainer.py`):** Invokes a local open-source LLM via Ollama (`temperature = 0.0`, `seed = 2026`) **solely** to translate the already-proven statistical profile JSON into a concise Greek summary for sales teams, with an automatic deterministic grounded fallback when Ollama is offline.

---

## 2. Model Selection & Four-Axis Justification (Section 5.3)

### 2.1 Greek KAD Activity Embedding Models (`src/features.py`)

We support and benchmark local embedding models from `EMBEDDING_MODELS_REGISTRY` (`src/config.py`) across the **four mandatory axes of Section 5.3** on the 3,736 unique Greek KAD descriptions in `vwHackActivities.csv` and the 10-split 80/20 Hold-Out benchmark (`evaluate_holdout.py`):

| Axis | Option A: `char-ngram-svd` (Default Fast Deterministic Baseline) | Option B: `ibm-granite/granite-embedding-107m-multilingual` | Option C: `intfloat/multilingual-e5-small` / `e5-base` |
| :--- | :--- | :--- | :--- |
| **1. Greek Quality on Challenge Data** (`Recall@500` / `Recall@2000` vs 1.83% / 8.17% Baseline) | **`Recall@500`: 13.83% ± 3.50%** (**7.5x** vs baseline)<br>**`Recall@2000`: 26.50% ± 4.56%** (**3.2x** vs baseline)<br>Captures Greek morphological roots (`Εστιατόρια`, `Καφετέριες`, `Τροφοδοσία`) + exact 4-digit NACE hierarchy. | **High semantic generalization** across synonymous Greek hospitality descriptions (e.g. `Υπηρεσίες παροχής γευμάτων` $\leftrightarrow$ `Εστιατόρια`). | Strong multilingual retrieval baseline; requires `"passage: "` prefix formatting. |
| **2. License & Commercial Use** | **MIT / Built-in** (Scikit-Learn / SciPy). **100% Commercial Use Permitted.** | **Apache-2.0 License** (verified in model repository). **100% Commercial Use Permitted.** | **MIT License** (verified in repository). **100% Commercial Use Permitted.** |
| **3. Origin & Digital Sovereignty** | **100% Local / On-Device** (Zero external weights required; trained deterministically on the ΓΕΜΗ Greek KAD taxonomy). | **IBM Research** (Open-source Apache-2.0 release with enterprise governance & documented multilingual training data). | **Microsoft Research** (Open-weights MIT release). |
| **4. Execution Cost (Hardware: Ryzen 2600X, 32GB DDR4, CPU)** | **Time:** `4.66s` cold / `<0.05s` cached (`0.19 ms` per company)<br>**Peak RAM:** `131.4 MB`<br>**Disk Footprint:** `1.8 MB` cached `.npz` | **Time:** `~11.5s` cold for 3,736 KADs / `<0.05s` cached<br>**Peak RAM:** `~480 MB`<br>**Params:** `107M` | **Time:** `~14.2s` (`e5-small`) / `~28.5s` (`e5-base`) cold for 3,736 KADs<br>**Peak RAM:** `~520 MB`–`1.1 GB` |

*Architectural Insight:* Because `vwHackActivities.csv` contains only **3,736 unique KAD descriptions** (while `vwHackCompanyActivity.csv` has 161,964 links), we embed the 3,736 unique descriptions **once**, cache them to `output/cache/`, and compute all 24,424 company vectors via sparse matrix multiplication (`W_sparse @ kad_embeddings`) in **< 0.12 seconds**.

---

### 2.2 Local Generative LLM for Greek Profile Descriptions (`src/llm_explainer.py`)

- **Supported Local Models (via Ollama 4-bit quantization):**
  - **European / Greek Sovereign Models:** `ilsp/llama-krikri-8b-instruct` (Greece — Athena RC / ILSP, Llama 3.1 License), `utter-project/EuroLLM-9B-Instruct` (EU-funded, Apache-2.0), `BSC-LT/salamandra-7b-instruct` (Spain — Barcelona Supercomputing Center, Apache-2.0), `mistralai/Ministral-3-3B-Instruct` (France, Apache-2.0).
  - **General Multilingual Local Fallback:** `qwen2.5:7b-instruct` (Apache-2.0) and a built-in **Deterministic Grounded Greek Narrative Engine** (`0.0001s` per profile, zero RAM overhead) when Ollama is not running.
- **Reproducibility & Groundedness Enforcement (Requirement A6):**
  - Generation parameters are locked to `temperature = 0.0`, `seed = 2026`, `top_p = 1.0`.
  - The prompt receives **only** the verified statistical JSON of the profile (`client_count`, `turnover_share`, `stability_score`, `confidence_score`, and the Hypergeometric/Empirical Bayes proven traits) and is strictly forbidden from inventing numbers or making creditworthiness claims.

---

## 3. Data Card: Sources, Licenses & Preprocessing

| Dataset / Source | Records | Origin & License | Preprocessing & Quality Safeguards (`src/ingest.py`) |
| :--- | :---: | :--- | :--- |
| `vwHackCompany.csv` | 24,424 companies (23,945 unique VATs) | **ΓΕΜΗ Open Data** (General Commercial Registry) & **© OpenStreetMap contributors (ODbL)** for GPS coordinates (96.01% coverage). Exclusive Hackathon use. | Strip UTF-8 BOM; preserve 9-digit `vat` leading zeros; convert `,` decimal GPS strings to `Float64`; compute `company_age_years` and `branch_count` per VAT. |
| `vwHackCompanyActivity.csv` | 161,964 links | **ΓΕΜΗ Open Data** (NACE 2.1 / KAD 2025 8-digit codes). | Normalize 1 `Λοιπή` row to `Βοηθητική`; resolve the 993 companies with $>1$ `Κύρια` KAD (due to the `2026-03-01` NACE 2.1 transition) by selecting the earliest `56*` hospitality KAD as `primary_kad` while keeping all KADs in `all_kads`. |
| `vwHackActivities.csv` | 3,736 unique KADs | **ΓΕΜΗ / ELSTAT KAD 2025 Taxonomy**. | Clean whitespace, zero-pad to 8 digits, and prepend 4-digit category code before embedding. |
| `ForceCRM_Hackathon_Pelatologio.xlsx` | 300 B2B clients | **ForceCRM Challenge Dataset** (indicative `turnover` €1,360–€87,270; 20% held out by organizers for evaluation). | Zero-pad `ΑΦΜ` to 9 digits; flag **all** `company_id` rows belonging to any client VAT (`is_client_vat = True`) so zero client branches ever leak into `lookalikes.csv`. |

---

## 4. Known Limitations, Failure Modes & Mitigations

1. **Declarative / Noisy Secondary KADs:** Many Greek food businesses declare dozens of secondary KADs (up to 239 on a single company) that they do not actively practice.
   - *Mitigation:* We apply smooth IDF weighting ($\text{idf\_power}=1.25$), lower base weights for secondary/auxiliary KADs ($0.75$ / $0.25$ vs $3.50$ for Primary), sublinear secondary-count damping ($n_{\text{sec}}^{-0.35}$), and **unit L2 normalization** so multi-KAD companies are neither distorted by noise nor penalized in cosine similarity (`Mean Sim = 0.4615` for 1-KAD vs `0.4747` for $\ge 20$-KAD companies).
2. **Missing GPS Coordinates (3.99% of Universe / 975 records):**
   - *Mitigation:* Companies missing `(latitude, longitude)` receive median spatial density imputation and neutral (`0.50`) spatial kernel affinity while retaining full Prefecture/Municipality Empirical Bayes Lift scoring.
3. **Small-Sample Statistical Artifacts in ICP Traits:**
   - *Mitigation:* Raw lift can be artificially high when a rare municipality or KAD appears 1–2 times in a cluster. We enforce **Hypergeometric exact tests** with **Benjamini-Hochberg FDR correction** ($p_{\text{FDR}} \le 0.05$) and **Empirical Bayes Shrinkage** ($m_{\text{prior}} = 10.0$) toward the Universe baseline rate.
4. **Outlier / Atypical Clients:**
   - *Mitigation:* Clients whose maximum similarity score across all discovered ICPs falls below the dynamic threshold (`no_fit_min_score = 0.35`) are explicitly flagged as **`NO_FIT`** in `assignments.csv` (14 out of 300 clients) rather than being forced into a poor-fit cluster.
