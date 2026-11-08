# 📄 Model & Data Card — ForceCRM B2B Profile Analysis & Lookalike System

**Project:** Archimedes2026 (NKUA Hackathon · Nov 2026)  
**Deliverable:** Section 10 & Section 5.3 — Model & Data Card

---

## 1. System Purpose & Architecture Summary

**Archimedes2026** is a 100% locally executed B2B Customer Segmentation (Ideal Customer Profile discovery) and Lookalike Recommendation system for Greek food & beverage / hospitality businesses (Main KAD `56`, founded `2020–2024`, $N = 24,424$ companies, $N_c = 300$ clients).

To guarantee 100% deterministic reproducibility and zero hallucination risk, the system separates mathematical discovery from narrative generation:
1. **Deterministic Hybrid Statistical Core (`src/ingest.py`, `src/features.py`, `src/icp_engine.py`, `src/verifier.py`):** Performs multi-branch VAT deduplication, Hybrid KAD representation (Neural CPU embeddings + exact ARPACK SVD over Greek character/word TF-IDF and KAD-by-KAD PPMI co-occurrence), static Parquet pre-caching (`data/cached_*.parquet`), 3D spherical spatial density indexing, deterministic turnover-weighted K-Means (`init="k-means++", n_init=200, random_state=42`), 200-resample Bootstrap ARI stability validation (`94.9%`), Hypergeometric FDR + Empirical Bayes Shrinkage Lift calculation, premise address turnover supersession detection, and unified lookalike/assignment scoring with sub-5ms SHA-256 client portfolio caching.
2. **Grounded Local AI Explainer (`src/llm_explainer.py`):** Uses a local open-weights LLM via Ollama (`temperature=0.0, seed=2026`) solely to translate the verified statistical JSON of each ICP into an executive Greek narrative, backed by a zero-latency deterministic Greek fallback when Ollama is offline.

---

## 2. Model Selection & Four-Axis Justification (Section 5.3)

### 2.1 Hybrid Greek KAD Representation (`src/features.py`)

We encode the 3,736 unique Greek KAD descriptions in `vwHackActivities.csv` using a tunable hybrid representation $\mathbf{e}_k^{\text{hybrid}} = [\sqrt{\alpha}\,\mathbf{e}_k^{\text{neural}}, \sqrt{1-\alpha}\,\mathbf{e}_k^{\text{svd}}]$ and evaluate candidate models across the **four mandatory axes of Section 5.3** using Repeated 5-Fold Cross-Validation (`RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)` -> 100 hold-out evaluations in `evaluate_holdout.py`):

| Axis | Signal B (`char-ngram-svd` + PPMI ARPACK SVD) | Signal A Option 1: `ibm-granite/granite-embedding-107m-multilingual` | Signal A Option 2: `intfloat/multilingual-e5-base` |
| :--- | :--- | :--- | :--- |
| **1. Greek Quality on Challenge Data** (`100-Fold CV Recall@500 / @2000` vs `1.87% / 7.48%` Baseline) | **`Recall@500`: `14.82% ± 4.15%` (+7.9x)**<br>**`Recall@2000`: `31.87% ± 6.16%` (+4.3x)**<br>Captures Greek morphological roots (`char_wb` 3–5 n-grams), exact 2/4/6/8-digit NACE hierarchy, and empirical KAD-by-KAD PPMI co-occurrence. | **High cross-synonym semantic alignment** across Greek hospitality descriptions (e.g., `Υπηρεσίες παροχής γευμάτων` ↔ `Εστιατόρια`) with `torch.use_deterministic_algorithms(True)`. | Strong multilingual dense retrieval embedding; uses `"passage: "` prefix formatting on CPU. |
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

---

## 5. Extra Points Implementation & Predictive Feature Engineering

### 5.1 Extra Point X1: Geographical ICP & HoReCa Micro-Cluster Density (`src/features.py` & `src/icp_engine.py`)
Using `scipy.spatial.cKDTree` on 3D spherical coordinates, we extract key spatial signals:
- **Local Competitor Density ($250\text{m}$ & $1\text{km}$):** Log1p count of total Universe KAD 56 venues within 250m and 1km, identifying dense culinary corridors vs. standalone destinations.
- **Micro-Neighborhood Bar/Cafe Share ($500\text{m}$):** Local fraction of Bar/Cafe venues (`5630`) vs Restaurants (`5610`) within 500m walking radius, capturing nightlife strips vs family dining zones.
- **Bar/Cafe to Restaurant Log-Ratio ($1\text{km}$):** Computes $\ln((N_{5630} + 1)/(N_{5610} + 1))$, distinguishing beverage-dominant nightlife zones (e.g. Ladadika, Gazi) from dining clusters.
- **Geographical Commercial Regime (`geo_regime`):** Partitions all Greek municipalities into `METRO_ATHENS_THESS` (41.1%), `MAINLAND_REGIONAL` (38.9%), `ISLAND_TOURISM` (16.1%), and `COASTAL_RESORT` (3.8%, showing >4x client representation lift!). Integrated as a testable categorical trait in the Hypergeometric + Empirical Bayes Shrinkage engine.
- **Per-ICP B2B Delivery Route Gravity:** Proximity specifically to existing active clients assigned to ICP $c$ (blended 65% per-ICP + 35% global) and exact `nearest_client_km` distance included in sales justifications for venues within $\le 2\text{km}$.
- **Interactive OpenStreetMap Route Planner (`app.py`):** Interactive zero-API-key Plotly `scatter_mapbox` (`mapbox_style="open-street-map"`) mapping clients sized by turnover alongside top Lookalikes with Municipality (`Δήμος`) and ICP filters for neighborhood sales reps.

### 5.2 Extra Point X2: OpenStreetMap Store Matching with 3-Tier Ambiguity Resolution (`data/external_enrichment.parquet`)
Matches stores against cached OSM hospitality POIs (`amenity=restaurant|cafe|bar|fast_food|pub`) within $75\text{m}$ using `cKDTree` and RapidFuzz token-set transliteration:
- **Tier 1 (Confirmed Match — 36.2%, 8,835 venues):** Distance $<50\text{m}$ AND Name Similarity $\ge 75\%$. Full OSM metadata transferred.
- **Tier 2 (Spatial-Only Ambiguous — 9.2%, 2,259 venues):** Distance $<25\text{m}$, single OSM venue, but GEMI name is generic `«Επιχείρηση Εστίασης»`. Physical premise verified; brand name unconfirmed.
- **Tier 3 (Quarantined Ambiguous — 17.0%, 4,147 venues):** Multiple OSM venues within $30\text{m}$ with Name Similarity $<75\%$. Excluded from tag transfer to prevent data contamination.
- **Unmatched (37.6%, 9,183 venues):** Non-overlapping locations or missing GPS.

### 5.3 Extra Point X3: Municipality Demographics & Transit Hub Foot-Traffic Catchment (`data/external_enrichment.parquet`)
Enriches stores with a built-in Greek municipal reference dataset:
- **Municipality Typology:** Classifies all 330 municipalities into *Urban Metropolitan* (34.1%), *Provincial/Rural* (32.0%), *Island Tourist* (16.1%), *Regional Commercial City* (14.0%), and *Seasonal Coastal Resort* (3.8%).
- **Socioeconomic Indicators:** Attaches `purchasing_power_index`, `student_population_density`, `tourist_intensity_index`, and `coastal_distance_km`.
- **Transit Hub Foot-Traffic Catchment:** Measures distance via `cKDTree` to 37 major Greek transit generators (Athens & Thessaloniki Metro stations, passenger ports, airports, university campuses). Identifies venues within $<500\text{m}$ walking distance (`is_transit_hub_catchment`, 8.8%, 2,143 venues).

### 5.4 Behavioral Expansion Velocity, Seasonality & Concept Tier
- **Post-Launch Expansion Velocity:** Measures days from `incorporation_date` to secondary KAD `date_from`. Companies expanding activities $\ge 180$ days post-launch (77.1%) represent genuine commercial pivots vs. Day-1 accountant noise.
- **Seasonality & Tourism Regime:** Cyclical month encoding ($\sin/\cos$) and pre-summer launch indicator (Months 3–6, 37.8%) capturing seasonal hospitality investments.
- **Brand Sophistication & Concept Tier:** Flags modern hospitality concepts (*Coffee, Roasters, Bistro, Burger, Street, Brunch, Bar, Cocktail, Project, Gastro*) and captures multi-company franchise networks via `shared_brand_network_size`.
