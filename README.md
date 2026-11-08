# 🧭 Archimedes2026 — ForceCRM B2B ICP Discovery & Lookalike Engine

**Archimedes2026** is a 100% locally executed B2B Customer Segmentation (Ideal Customer Profile discovery) and Lookalike Recommendation system built for the **ForceCRM Challenge** (NKUA Hackathon · Nov 2026).

Given a universe of **24,424 Greek Food & Beverage / Hospitality companies** (main KAD `56`, founded `2020–2024`) and an active client portfolio of **300 B2B customers** (`Pelatologio.xlsx`), the system automatically:
1. **Discovers $K \in [1, 6]$ statistically validated Ideal Customer Profiles (ICPs)** weighted by commercial value (`turnover`), reporting Bootstrap cluster stability and composite confidence scores.
2. **Assigns all 300 clients** to their best-matching ICP (`output/assignments.csv`), explicitly isolating atypical outliers as `"NO_FIT"`.
3. **Ranks the 24,063 non-client Universe companies** (`output/lookalikes.csv`) using the exact same similarity metric to uncover high-conversion B2B sales leads.
4. **Generates grounded Greek executive summaries** (`output/profiles.json`) via a local LLM (Ollama at `temperature=0.0`) or a zero-latency deterministic fallback, citing only statistically proven traits ($p_{\text{FDR}} \le 0.05$).

---

## 1. Quick Start & Usage

### Prerequisites
- **Python 3.12+** managed via [`uv`](https://docs.astral.sh/uv/)
- *(Optional)* [Ollama](https://ollama.com/) running locally for live generative AI summaries (works 100% offline without Ollama via our deterministic Greek narrative fallback)

### Installation
```bash
git clone https://github.com/gribone/Archimedes2026.git
cd Archimedes2026
uv sync
```

### A. Launch the Executive Web UI (`app.py`)
```bash
# Standard launch (offline, instant sub-40ms calculation)
python app.py --port 8501

# (Optional) Launch with live Google Places API verification enabled
export GOOGLE_PLACES_API_KEY="your-google-api-key"
python app.py --port 8501

# (Optional) Launch Streamlit legacy interface
python app.py --streamlit
```
Opens at `http://localhost:8501` and provides:
- **Executive Precision Dark UI:** High-density responsive corporate layout (Tailwind CSS, JetBrains Mono, Plus Jakarta Sans) with **zero horizontal scrolling**.
- **Sales Intelligence Cockpit:** 4-tab interactive sales intelligence cockpit replacing raw statistical charts:
  - **🔥 Lead Propensity & Heat Tiers:** Real-time breakdown of Hot (≥90%), Warm (80-89%), and Nurture (<80%) prospects with 1-click table filtering (`Προβολή {N} Hot Leads`).
  - **🗺️ Sales Territory Hotspots:** Top 5 Municipalities horizontal bar chart with concrete lead counts and regional sales blitz strategy tips.
  - **💼 Commercial Sales Playbooks:** Deal sizing, decision maker personas, operational pain points (myDATA, order volumes, seasonal cash-flow), verbatim Greek elevator pitch phone openers, and objection rebuttals.
  - **📊 Polished Benchmark Lift:** Categorized badges (`[ΚΑΔ]`, `[Μορφή]`, `[Δήμος]`, `[Νομός]`) and clean comparison bars vs. the 24,424 market baseline.
- **Neural Embeddings Toggle:** Instant switch in both the sidebar drawer and the header status pill to alternate between Neural Transformer + SVD embeddings and Pure Algebraic SVD baseline with zero page reload.
- **Interactive Greece Map:** Real-time Leaflet map with regional presets (Attica, Halkidiki, Crete, Islands), color-coded ICP clusters, and 1-click center-and-zoom cross-navigation from table rows.
- **Top Lookalikes & Lead Verification Suite:** Filterable leads with action buttons (`📍 Map Focus`, `🗺️ Google Maps`, `🏢 GEMI Portal`, `🛡️ Status Check`, `🚫 Exclude`).
- **Automated Defunct Company Filtering:** Live status verification badges (`🟢 Operational` / `🔴 Permanently Closed` / `🟠 Superseded`) powered by the Premise Supersession Engine and Google Places API.
- **Excluded Companies Management:** Header modal tracking all excluded venues with 1-click restore functionality.
- **Hidden Analyst Drawer:** Submenu for instant parameter re-scoring ($<40\text{ ms}$) and hold-out evaluation.

### B. Generate Official Deliverables via CLI
```bash
# 1. Stage 1 Data Ingestion & Baseline Verification
uv run python -m src.ingest

# 2. Build / Refresh Static Parquet Feature Caches (<0.5s warm load)
uv run python -m src.features

# 3. Run Full Pipeline & Export Official Deliverables to output/
uv run python -m src.llm_explainer
```
Exports:
- `output/profiles.json` — Full statistical & Greek narrative specifications for each ICP
- `output/assignments.csv` — `company_id, profile_id, score` for all client rows (including `"NO_FIT"`)
- `output/lookalikes.csv` — `profile_id, company_id, rank, score, αιτιολόγηση` for top $\ge 2,000$ non-client candidates

### C. Run 100-Fold (`5x20`) Cross-Validation Hold-Out Evaluation (`evaluate_holdout.py`)
```bash
# Default RepeatedKFold (5 splits x 20 repeats = 100 hold-out evaluations)
uv run python evaluate_holdout.py

# Run 100-Fold CV Grid Search over Hybrid alpha and similarity weights (<60s)
uv run python evaluate_holdout.py --grid-search

# Run with Pure Algebraic SVD baseline (no PyTorch/neural embeddings required)
uv run python evaluate_holdout.py --no-embeddings

# Compare multiple Greek KAD embedding models side-by-side (Section 5.3)
uv run python evaluate_holdout.py --compare-models char-ngram-svd granite-107m e5-base
```

### D. Run Automated Test Suite (`pytest`)
The project features a comprehensive 121-test suite verifying data ingestion contracts, feature stores, deterministic clustering, API contracts, deliverable schema integrity, and adversarial corruption tests:
```bash
# Run full 121-test suite
uv run pytest

# Run deliverable schema & SHA-256 baseline verification tests
uv run pytest tests/test_deliverables.py

# Run Starlette ASGI REST API and web service integration tests
uv run pytest tests/test_api.py
```

---

## 2. System Architecture & Pipeline Overview

All tunable parameters live in a single central dataclass (`PipelineConfig` in [`src/config.py`](file:///home/marios/Projects/Archimedes2026/src/config.py)) and automatically load the winning 100-fold CV weights from [`data/optimal_weights.json`](file:///home/marios/Projects/Archimedes2026/data/optimal_weights.json).

| Stage | Module | What It Does |
| :--- | :--- | :--- |
| **1. Data Ingestion** | [`src/ingest.py`](file:///home/marios/Projects/Archimedes2026/src/ingest.py) | Cleans UTF-8 BOMs and comma-decimal GPS strings, zero-pads 9-digit VATs (`ΑΦΜ`), resolves the 993 companies with multiple Primary KADs (from the `2026-03-01` NACE 2.1 transition), and flags all 361 rows belonging to the 300 client VATs (`is_client_vat`) so zero client branches leak into lookalikes. |
| **2. Hybrid Feature Store** | [`src/features.py`](file:///home/marios/Projects/Archimedes2026/src/features.py) | Encodes all 3,736 unique KADs via **Signal A** (Neural Transformer CPU embeddings) + **Signal B** (Greek NFD character/word TF-IDF + NACE hierarchy + KAD-by-KAD PPMI co-occurrence with exact ARPACK SVD). Combines them with 3D spherical spatial density, municipality typology, and transit hub proximity, pre-cached to `data/cached_universe_features.parquet` and `data/cached_universe_features.npz`. |
| **3. ICP & Scoring Engine** | [`src/icp_engine.py`](file:///home/marios/Projects/Archimedes2026/src/icp_engine.py) | Runs deterministic turnover-weighted K-Means (`init="k-means++", n_init=200, random_state=42`) across $K \in \{1..6\}$, validates clusters via 200-resample Bootstrap ARI stability (`95.8%`), extracts traits using Hypergeometric FDR + Empirical Bayes Shrinkage Lift, and computes unified similarity scores with SHA-256 client portfolio caching. |
| **4. Grounded AI Explainer** | [`src/llm_explainer.py`](file:///home/marios/Projects/Archimedes2026/src/llm_explainer.py) | Translates the verified statistical JSON of each ICP into an executive Greek narrative (`temperature=0.0, seed=2026`), strictly forbidding fabricated metrics or creditworthiness claims. |
| **5. Status Verification & Premise Supersession** | [`src/verifier.py`](file:///home/marios/Projects/Archimedes2026/src/verifier.py) | Discovers physically defunct and closed venues by analyzing chronological premise turnover across all 24,424 records (1,096 superseded entities detected offline). Integrates with Google Places API and CRM blacklist to guarantee only genuinely active businesses rank in top lookalikes. |
| **6. Interactive Web Service** | [`src/service.py`](file:///home/marios/Projects/Archimedes2026/src/service.py) | Stateful, thread-safe application service wrapping the pipeline; computes real-time Sales Intelligence (lead propensity funnels, top territories, commercial battlecards), manages sub-40ms parameter re-scoring, and enforces client SHA-256 caching. |
| **7. ASGI Server & REST API** | [`src/server.py`](file:///home/marios/Projects/Archimedes2026/src/server.py) | High-performance Starlette ASGI backend serving static Single-Page Application assets and JSON API endpoints (`/api/bootstrap`, `/api/run`, `/api/company/{id}`, `/api/verify`, `/api/export`). |

### 100-Fold (`5x20`) Repeated K-Fold Benchmark Results (`RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)`)

| Evaluation Metric | Official Baseline (Primary KAD Frequency) | Archimedes2026 Hybrid ICP Engine | Improvement |
| :--- | :---: | :---: | :---: |
| **`Recall@500`** (Top 2.08% of Universe) | `1.87%` analytical (`2.30%` empirical) | **`14.83% ± 4.19%`** | **+7.9x Lift** |
| **`Recall@2000`** (Top 8.31% of Universe) | `7.48%` analytical (`7.73%` empirical) | **`31.87% ± 6.18%`** | **+4.3x Lift** |

---

## 3. Configuration & Default Weights (`src/config.py`)

Locked-in optimal weights discovered via 5-Fold Cross-Validation Grid Search:

| Parameter | Default | Role |
| :--- | :---: | :--- |
| `hybrid_alpha` ($\alpha$) | `0.50` (`0.0` for pure SVD) | Mixing ratio between Neural Signal A ($\sqrt{\alpha}$) and Algebraic PPMI-SVD Signal B ($\sqrt{1-\alpha}$) |
| `block_weight_semantic` ($w_{\text{sem}}$) | `1.30` | Weight of the 64d KAD semantic activity block in the company feature vector |
| `block_weight_spatial` ($w_{\text{spat}}$) | `1.00` | Weight of the 6d 3D-Haversine spatial & client-density block |
| `block_weight_legal` ($w_{\text{leg}}$) | `0.85` | Weight of legal form (`ΑΕ`, `ΙΚΕ`, `ΕΕ`, `ΟΕ`, `ΑΤΟΜΙΚΗ`), company age, and branch count |
| `sim_weight_feature_cosine` ($w_{\text{cos}}$) | `0.60` | Lookalike/assignment weight on dense feature cosine similarity to the ICP centroid |
| `sim_weight_eb_lift_affinity` ($w_{\text{eb}}$) | `0.30` | Weight on Empirical Bayes log-lift categorical affinity (Prefecture, Municipality, KAD, Legal Type) |
| `sim_weight_spatial_kernel` ($w_{\text{sp}}$) | `0.10` | Weight on Haversine Gaussian spatial proximity ($\sigma_{\text{km}} = 25\text{ km}$) |
| `eb_prior_strength` ($m_{\text{prior}}$) | `12.0` | Beta-Binomial shrinkage strength toward the Universe base rate |

---

## 4. Mathematical Appendix (Under the Hood)

This section details the exact equations used across `src/features.py` and `src/icp_engine.py`.

### 4.1 Deterministic Hybrid KAD Representation (`src/features.py`)

Each of the $M = 3,736$ unique KAD activities $k$ in `vwHackActivities.csv` is represented by two orthogonal signals:

1. **Signal A — Neural Linguistic Embedding ($\mathbf{e}_k^{\text{neural}}$):**
   Computed on CPU (`device="cpu"`) with `torch.use_deterministic_algorithms(True)` over the Greek KAD description (supporting `ibm-granite/granite-embedding-107m-multilingual`, `intfloat/multilingual-e5-base`, and `BAAI/bge-m3`), L2-normalized to unit length.
2. **Signal B — Exact Algebraic N-Gram + PPMI Behavioral SVD ($\mathbf{e}_k^{\text{svd}} \in \mathbb{R}^{64}$):**
   - **Text & NACE Hierarchy Matrix:** Greek descriptions are NFD-normalized (accent-stripped, lowercased) and vectorized via character `char_wb` 3–5 n-grams, word 1–2 n-grams, and explicit 2/4/6/8-digit NACE hierarchical prefix tokens.
   - **Behavioral Co-Occurrence Matrix (PPMI):** From the 161,964 company-KAD links in `vwHackCompanyActivity.csv`, let $C_{a,b}$ be the number of companies declaring both KAD $a$ and KAD $b$. The Positive Pointwise Mutual Information matrix is:

$$
\text{PPMI}(a, b) = \max\left(0,\; \ln\frac{C_{a,b} \cdot \sum_{u,v} C_{u,v}}{\left(\sum_v C_{a,v}\right)\left(\sum_u C_{u,b}\right)}\right)
$$

   - **Exact ARPACK SVD:** The concatenated `[TF-IDF, 0.55 * PPMI]` matrix is decomposed via deterministic `TruncatedSVD(n_components=64, algorithm="arpack")` with sign canonicalization and L2-normalized to $\mathbf{e}_k^{\text{svd}}$.

3. **Hybrid Fusion ($\alpha \in [0, 1]$):**

$$
\mathbf{e}_k^{\text{hybrid}} = \left[\sqrt{\alpha}\,\mathbf{e}_k^{\text{neural}},\; \sqrt{1 - \alpha}\,\mathbf{e}_k^{\text{svd}}\right], \qquad \left\|\mathbf{e}_k^{\text{hybrid}}\right\|_2 = 1
$$

### 4.2 Multi-KAD Company Aggregation & Noise Damping

Companies declare between 1 and 239 KADs. To prevent noisy secondary KADs from distorting a company's profile while ensuring multi-KAD businesses are never penalized in cosine similarity:

1. **Smooth Inverse Document Frequency:** $\text{IDF}(k) = \left(\ln\frac{1 + N}{1 + \text{df}(k)} + 1\right)^{1.25}$, where $N = 24,424$.
2. **Hierarchy & Sublinear Secondary Damping:** Primary (`Κύρια`) KADs receive base weight $w_{\text{primary}} = 3.50$. For a company $i$ with $n_{\text{sec}}(i)$ non-primary KADs, secondary (`Δευτερεύουσα`, $w_{\text{sec}} = 0.75$) and auxiliary (`Βοηθητική`, $w_{\text{aux}} = 0.25$) weights are damped by $n_{\text{sec}}(i)^{-0.35}$.
3. **L2 Unit Normalization:**

$$
\mathbf{v}_i^{\text{sem}} = \text{Normalize}_2\left(\sum_{k \in \mathcal{A}(i)} W_{i,k}\,\mathbf{e}_k^{\text{hybrid}}\right)
$$

### 4.3 3D Spherical Spatial Coordinates & Unified Feature Vector

GPS coordinates $(\phi_i, \lambda_i)$ in radians are projected onto the unit 3D sphere $\mathbf{x}_i = [\cos\phi_i\cos\lambda_i,\; \cos\phi_i\sin\lambda_i,\; \sin\phi_i]$ so Euclidean chord distance monotonically tracks great-circle Haversine distance. Combined with log-distances to the nearest 5 Universe neighbors, nearest 3 active clients, and 25 km client count via `scipy.spatial.cKDTree`, each company $i$ is mapped to the unified L2-normalized vector:

$$
\mathbf{z}_i = \text{Normalize}_2\Big(\big[w_{\text{sem}}\,\mathbf{v}_i^{\text{sem}},\; w_{\text{spat}}\,\mathbf{v}_i^{\text{spat}},\; w_{\text{leg}}\,\mathbf{v}_i^{\text{leg}}\big]\Big)
$$

### 4.4 Turnover-Weighted Deterministic Clustering & Stability (`src/icp_engine.py`)

Each active client $j \in \{1,\dots,300\}$ with annual revenue $T_j$ receives sample weight:

$$
w_j = 1 + 0.65 \cdot \max\left(0,\; \ln\frac{T_j}{\text{median}(T)}\right)
$$

We run deterministic K-Means (`init="k-means++", n_init=200, random_state=42`) across $K \in \{2,\dots,6\}$ and select the optimal $K^*$ maximizing a composite of weighted Silhouette, $B = 200$ Bootstrap Adjusted Rand Index ($\text{ARI}$) stability (`95.8%`), balance penalty, and client-vs-Universe separation.

### 4.5 Hypergeometric FDR & Empirical Bayes Shrinkage Lift

For each categorical trait $f$ appearing $K_{\text{univ}}(f)$ times in the Universe ($N = 24,424$) and $x_{k,f}$ times in ICP $k$ (of size $n_k$):

- **Exact Hypergeometric Tail & Benjamini-Hochberg FDR:**

$$
p(k, f) = \sum_{j = x_{k,f}}^{\min(n_k, K_{\text{univ}}(f))} \frac{\binom{K_{\text{univ}}(f)}{j}\binom{N - K_{\text{univ}}(f)}{n_k - j}}{\binom{N}{n_k}}, \qquad p_{\text{FDR}}(k, f_{(r)}) = \min_{j \ge r}\left(1,\; \frac{M_{\text{tests}}}{j}\,p(k, f_{(j)})\right)
$$

- **Empirical Bayes Shrinkage Lift ($m_{\text{prior}} = 12.0$, $p_{\text{univ}}(f) = K_{\text{univ}}(f)/N$):**

$$
\hat{p}_{\text{EB}}(k, f) = \frac{x_{k,f} + m_{\text{prior}} \cdot p_{\text{univ}}(f)}{n_k + m_{\text{prior}}}, \qquad \text{Lift}_{\text{EB}}(k, f) = \frac{\hat{p}_{\text{EB}}(k, f)}{p_{\text{univ}}(f)}
$$

### 4.6 HoReCa Micro-Cluster Density & B2B Delivery Route Gravity (Extra Point X1 - Geographical ICP)

Using `scipy.spatial.cKDTree` on 3D spherical unit coordinates $(\cos\phi\cos\lambda, \cos\phi\sin\lambda, \sin\phi)$, we engineer micro-spatial features that capture actual hospitality clustering:
1. **Competitor Density ($250\text{m}$ & $1\text{km}$):** Counts neighboring HoReCa venues within chord distance corresponding to walking distance ($250\text{m}$) and commercial catchment ($1\text{km}$).
2. **Local Bar/Cafe to Restaurant Ratio ($1\text{km}$):** Computes $\ln\frac{N_{5630} + 1}{N_{5610} + 1}$ within $1\text{km}$, separating beverage/nightlife clusters (e.g. Ladadika, Gazi) from dining corridors.
3. **B2B Delivery Route Gravity:** Evaluates logistical delivery efficiency to existing clients:
   $$G_i = \sum_{c \in \text{Clients}} w_c \cdot \exp\left(-\frac{1}{2}\left(\frac{\text{dist}_{\text{km}}(i, c)}{15\text{ km}}\right)^2\right)$$
   where client weights $w_c = \ln(1 + \text{Turnover}_c)$ prioritize high-volume customer distribution routes.

### 4.7 OpenStreetMap Store Matching & 3-Tier Ambiguity Resolution (Extra Point X2)

We match the 24,424 universe stores against OpenStreetMap hospitality POIs (`amenity=restaurant|cafe|bar|fast_food|pub`) within $75\text{m}$ using `cKDTree` and RapidFuzz token-set transliteration:
- **Tier 1 (Confirmed Match — 36.2%, 8,835 venues):** Distance $<50\text{m}$ AND Name Similarity $\ge 75\%$. Full OSM tag transfer (`osm_amenity`, `osm_cuisine`, `osm_venue_name`) is permitted.
- **Tier 2 (Spatial-Only Ambiguous — 9.2%, 2,259 venues):** Distance $<25\text{m}$, single OSM venue, but GEMI legal name is generic `«Επιχείρηση Εστίασης»`. Physical premise verified; brand name unconfirmed.
- **Tier 3 (Quarantined Ambiguous — 17.0%, 4,147 venues):** Multiple OSM venues within $30\text{m}$ with Name Similarity $<75\%$. Excluded from tag transfer to prevent data contamination.
- **Unmatched (37.6%, 9,183 venues):** Stores with no OSM hospitality POI within $75\text{m}$ or isolated provincial locations.

Pre-cached to `data/external_enrichment.parquet` for instant offline execution in $<0.02\text{s}$.

### 4.8 Municipality Demographics & Transit Hub Foot-Traffic Catchment (Extra Point X3)

Integrates a built-in reference taxonomy across all 330 Greek municipalities:
1. **Municipality Typology:** Classifies Greece into *Urban Metropolitan* (34.1%), *Provincial/Rural* (32.0%), *Island Tourist* (16.1%), *Regional Commercial City* (14.0%), and *Seasonal Coastal Resort* (3.8%).
2. **Socioeconomic Indicators:** Attaches `purchasing_power_index` (e.g. Kifisia/Glyfada/Mykonos = 1.40–1.55, Athens center = 1.05–1.15, provincial = 0.80), `student_population_density`, `tourist_intensity_index`, and `coastal_distance_km`.
3. **High-Flow Transit Hub Catchment:** Measures distance via `cKDTree` to 37 major Greek transit generators (Athens & Thessaloniki Metro stations, passenger ports, airports, and university campuses). Flags stores within $<500\text{m}$ walking distance (`is_transit_hub_catchment`, 8.8%, 2,143 stores).

### 4.9 Post-Launch Expansion Velocity, Seasonality & Brand Sophistication

1. **Post-Launch Expansion Velocity (`has_post_launch_expansion`, `post_launch_expansion_days`):** Measures days elapsed between `incorporation_date` and secondary KAD `date_from`. Businesses that added secondary activities $\ge 180$ days post-launch (77.1%) represent genuine commercial pivots and revenue expansion vs. Day-1 accountant noise.
2. **Seasonality & Tourism Regime (`is_pre_summer_launch`, `inc_month_sin`, `inc_month_cos`):** Identifies pre-summer incorporation windows (Months 3–6, 37.8%) which strongly correlate with seasonal tourist destinations.
3. **Brand Sophistication & Concept Tier:**
   - `has_distinctive_brand`: Distinguishes the 15,200 unique commercial trade names from the 9,224 generic default `«Επιχείρηση Εστίασης»` registrations.
   - `is_modern_concept`: Flags modern hospitality keywords (*Coffee, Roasters, Bistro, Burger, Street, Brunch, Bar, Cocktail, Project, Gastro*).
   - `shared_brand_network_size`: Counts distinct VATs sharing the same brand name to capture franchise networks operating under separate legal entities.

### 4.10 Unified Lookalike & Client Assignment Score

The exact same similarity function $S(i, k) \in [0, 1]$ scores both active clients (`assignments.csv`) and non-client Universe candidates (`lookalikes.csv`):

$$
S(i, k) = \frac{w_{\text{cos}}\,S_{\text{cos}}(i, k) + w_{\text{eb}}\,S_{\text{EB}}(i, k) + w_{\text{sp}}\,S_{\text{spat}}(i)}{w_{\text{cos}} + w_{\text{eb}} + w_{\text{sp}}}
$$

where:
- $S_{\text{cos}}(i, k) = \frac{1}{2}(\mathbf{z}_i^\top \boldsymbol{\mu}_k + 1)$ is the scaled cosine similarity to ICP $k$'s turnover-weighted centroid $\boldsymbol{\mu}_k$,
- $S_{\text{EB}}(i, k) = \left(1 + \exp\big(-0.55 \sum_c \gamma_c \ln\text{Lift}_{\text{EB}}(k, v_c(i))\big)\right)^{-1}$ is the sigmoid-squashed sum of Empirical Bayes log-lifts across Prefecture, Municipality, Legal Type, 8-digit Primary KAD, 4-digit NACE class, Age bracket, and Branch status, and
- $S_{\text{spat}}(i)$ is the 95th-percentile-normalized Haversine Gaussian kernel density ($\sigma_{\text{km}} = 25\text{ km}$) around active clients.

Clients whose best score $\max_k S(i, k)$ falls below the dynamic threshold $\tau_{\text{no-fit}} = \max(0.35,\; Q_{0.05}(S^*_{\text{clients}}) - 0.02)$ are assigned **`"NO_FIT"`**.

### 4.11 Premise Supersession & Closed Business Detection Engine (`src/verifier.py`)

A critical challenge with the raw hackathon snapshot (`vwHackCompany.csv`) is that it contains company registrations from 2020–2024 but **zero cessation dates, liquidation states, or dissolution fields**. In Greek F&B, 35%–40% of venues founded between 2020 and 2023 have closed or relocated by 2025/2026. Without filtering, historical models inevitably recommend defunct businesses as top leads (e.g. *So So So Ramen Ψυχικό*).

To solve this deterministically without requiring external API calls for all 24,424 companies, we engineered the **Premise Supersession Engine**:

#### 1. Physical Domain Intuition (Commercial Space Recycling)
Food service premises require specialized, high-capital infrastructure: industrial smoke exhaust ducts (*φουγάρο*), grease traps (*λιποσυλλέκτες*), 3-phase high-voltage power, and municipal health permits costing €50,000–€200,000. When a food venue shuts down, the physical unit is leased by a new food entrepreneur who registers a new legal entity at the identical address. Because two independent full-service restaurants cannot physically operate in the exact same kitchen at the same time, any prior entity registered at that address has been physically and operationally superseded.

#### 2. Vectorized Algorithmic Formulation
1. **Cadastral Premise Keying:** Each company $i$ with non-empty street data is mapped to:
   $$\text{Premise Key}(i) = \big(\text{Municipality}_i, \; \text{Normalize}(\text{Street}_i), \; \text{Street Number}_i\big)$$
2. **Temporal Trajectory Sequencing:** For each premise key with $|\text{Entities}| \ge 2$, entities are sorted chronologically by incorporation date:
   $$E_1 (t_1), \; E_2 (t_2), \; \dots, \; E_n (t_n) \quad \text{where } t_1 < t_2 < \dots < t_n$$
3. **Supersession Inference:**
   - Entity $E_n$ at $t_n$ is the **Active Occupant**.
   - All previous entities $E_1, \dots, E_{n-1}$ incorporated at $t_i < t_n$ are classified as **`CLOSED_SUPERSEDED`**:
     $$\text{Status}(E_i) = \begin{cases} \text{OPERATIONAL}, & i = n \\ \text{CLOSED\_SUPERSEDED}, & i < n \end{cases}$$
   - An explainable audit trail is generated: *"Αντικατάσταση στην ίδια διεύθυνση ({address}): διαδέχθηκε από την επιχείρηση «{name}_n» ({date}_n)"*.

#### 3. Empirical Findings on the Universe Dataset
- **914 unique physical addresses** had multiple sequential food business registrations between 2020 and 2024.
- **2,010 total companies** were involved in premise turnover.
- **1,096 older companies** across Greece were identified as physically superseded and defunct.
- **224 companies out of the top 2,000 lookalikes (11.2%)** were superseded entities and purged from recommendations.

#### 4. Multi-Tier Verification Hierarchy
[`CompanyStatusVerifier`](file:///home/marios/Projects/Archimedes2026/src/verifier.py) integrates four defense tiers:
1. **CRM Blacklist (`data/excluded_companies.json`):** Manually verified closures (e.g. *So So So Ramen Ψυχικό*, *Maison Colette Ψυχικό*, *Πρυτανείον Κολωνάκι*).
2. **Premise Supersession Intelligence:** Instant zero-cost offline detection of 1,096 superseded entities in $<15\text{ ms}$.
3. **Google Places API (`findplacefromtext`):** Enabled via `GOOGLE_PLACES_API_KEY` (permitted under Section 6 of the brief) for inspecting real-time `business_status` (`OPERATIONAL`, `CLOSED_PERMANENTLY`, `CLOSED_TEMPORARILY`).
4. **Persistent Cache (`data/verified_companies_cache.json`):** Persists all API results to preserve quota and avoid latency.

---

## 5. Data Sources, Licenses & EU AI Act Compliance

- **Registry & Taxonomy Data:** ΓΕΜΗ (General Commercial Registry) & ELSTAT KAD 2025 Open Data, provided by ForceCRM for the NKUA Hackathon.
- **GPS Coordinates:** © OpenStreetMap contributors, licensed under the **Open Database License (ODbL)**.
- **External Data APIs & Provenance (Brief Section 6):**
  - Section 6 of the technical brief explicitly permits external data sources: *"Εξωτερικές πηγές δεδομένων επιτρέπονται... Για το Google Places χρησιμοποιείτε δικό σας API key, εντός των δωρεάν μηνιαίων credits."*
  - **Google Places API:** Used for live business operating status verification (`findplacefromtext`), fully compliant with Google Maps Platform Terms of Service and hackathon limits.
- **EU AI Act Compliance (Regulation (EU) 2024/1689):**
  - **Not High-Risk (Annex III):** B2B commercial segmentation and lead prioritization is outside Annex III high-risk categories.
  - **Sole Proprietorship Safeguard (`ΑΤΟΜΙΚΗ`, 12,090 records):** Our system **never evaluates creditworthiness or financial solvency (`φερεγγυότητα`)**. All scores strictly measure B2B Commercial Fit / ICP Similarity.
  - **Article 50 Transparency:** All AI-generated Greek narratives carry an explicit disclosure badge and are restricted to verified statistical traits.

---

## 6. Repository Layout & Official Deliverables

```
Archimedes2026/
├── app.py                      # Executive application entrypoint (Starlette ASGI SPA default / Streamlit)
├── evaluate_holdout.py         # Vectorized <60s 100-Fold (5x20) RepeatedKFold Hold-Out Optimizer
├── pyproject.toml              # Project dependencies, packaging & pytest configuration
├── README.md                   # System documentation & mathematical specification
├── MODEL_DATA_CARD.md          # Comprehensive Model & Data Card (Section 5.3)
├── src/
│   ├── config.py               # Central PipelineConfig & optimal weights loading
│   ├── ingest.py               # Data ingestion, VAT zero-padding & client representative deduplication
│   ├── features.py             # Deterministic Hybrid KAD embedding fusion & feature pre-caching
│   ├── features/               # Modular spatial, transit, and enrichment feature subpackages
│   ├── icp_engine.py           # Turnover-weighted K-Means, Bootstrap ARI stability & scoring engine
│   ├── llm_explainer.py        # Grounded Greek AI executive summaries (Ollama / deterministic)
│   ├── verifier.py             # Premise supersession engine & Google Places API status verification
│   ├── service.py              # Stateful engine service, sales intelligence cockpit & caching
│   ├── server.py               # Starlette ASGI backend & REST API endpoints
│   └── static/                 # Executive Single-Page Application (HTML5, Tailwind CSS, Leaflet, Chart.js)
├── data/                       # Cached features, optimal weights, and hackathon datasets
│   ├── optimal_weights.json    # Winning 100-fold cross-validation hyperparameters
│   ├── cached_universe_features.parquet # Pre-cached 24,424 universe feature matrix
│   ├── cached_universe_features.npz     # Pre-cached compressed NumPy feature blocks
│   └── excluded_companies.json # CRM closed venues & blacklist registry
├── output/                     # Official deliverables
│   ├── profiles.json           # ICP statistical specifications & grounded Greek narratives
│   ├── assignments.csv         # Best ICP assignment & score for all 300 clients (with NO_FIT)
│   └── lookalikes.csv          # Ranked top 2,000+ prospective B2B sales leads
└── tests/                      # 121 automated unit, integration, and adversarial tests
```

