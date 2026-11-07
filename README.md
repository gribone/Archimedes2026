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
git clone <your-repo-url>
cd Archimedes2026
uv sync
```

### A. Launch the Executive Web UI (`app.py`)
```bash
uv run streamlit run app.py
```
Opens at `http://localhost:8501` and provides:
- **Executive KPI Banner:** Discovered ICPs, Assigned Clients, Outlier (`NO_FIT`) count, and Hold-Out `Recall@500` lift over the official baseline.
- **Tab 1 — Ideal Customer Profiles (ICPs):** Clean executive cards with Greek summaries, revenue share donut chart, paired *"Our ICP vs. Market Average"* comparison charts, and a hidden submenu for statistical deep-dives (`p-value`, `FDR`, `Empirical Bayes Lift`).
- **Tab 2 — Target Leads & Interactive Map:** Interactive Greece map (`plotly` MapLibre) plotting active clients vs. top lookalike leads, filterable lead table, and one-click CSV export.
- **Tab 3 — Client Portfolio Assignments:** Full assignment table with confidence scores and `"NO_FIT"` filtering.
- **Tab 4 — Engine Tuning & Diagnostics:** Interactive parameter tuning (`K`, Hybrid $\alpha$, feature block weights, similarity weights) and single-split hold-out tester.

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
- `output/assignments.csv` — `company_id, profile_id, confidence_score` for all 300 clients
- `output/lookalikes.csv` — `company_id, best_profile_id, similarity_score, rank` for all 24,063 non-client candidates

### C. Run 5-Fold Cross-Validation Hold-Out Evaluation (`evaluate_holdout.py`)
```bash
# Standard 5-Fold CV benchmark (240 train / 60 hidden test clients per fold)
uv run python evaluate_holdout.py --n-splits 5

# Run 5-Fold CV Grid Search over Hybrid alpha and similarity weights
uv run python evaluate_holdout.py --grid-search

# Compare multiple Greek KAD embedding models side-by-side (Section 5.3)
uv run python evaluate_holdout.py --compare-models char-ngram-svd granite-107m e5-base
```

---

## 2. System Architecture & Pipeline Overview

All tunable parameters live in a single central dataclass (`ICPConfig` in [`src/config.py`](file:///home/marios/Projects/Archimedes2026/src/config.py)).

| Stage | Module | What It Does |
| :--- | :--- | :--- |
| **1. Data Ingestion** | [`src/ingest.py`](file:///home/marios/Projects/Archimedes2026/src/ingest.py) | Cleans UTF-8 BOMs and comma-decimal GPS strings, zero-pads 9-digit VATs (`ΑΦΜ`), resolves the 993 companies with multiple Primary KADs (from the `2026-03-01` NACE 2.1 transition), and flags all 361 rows belonging to the 300 client VATs (`is_client_vat`) so zero client branches leak into lookalikes. |
| **2. Hybrid Feature Store** | [`src/features.py`](file:///home/marios/Projects/Archimedes2026/src/features.py) | Encodes all 3,736 unique KADs via **Signal A** (Neural Transformer CPU embeddings) + **Signal B** (Greek NFD character/word TF-IDF + NACE hierarchy + KAD-by-KAD PPMI co-occurrence with exact ARPACK SVD). Combines them with 3D spherical spatial density and legal/maturity features, pre-cached to `data/cached_*.parquet`. |
| **3. ICP & Scoring Engine** | [`src/icp_engine.py`](file:///home/marios/Projects/Archimedes2026/src/icp_engine.py) | Runs deterministic turnover-weighted K-Means (`init="k-means++", n_init=50, random_state=42`) across $K \in \{2..6\}$, validates clusters via Bootstrap ARI stability, extracts traits using Hypergeometric FDR + Empirical Bayes Shrinkage Lift, and computes unified similarity scores. |
| **4. Grounded AI Explainer** | [`src/llm_explainer.py`](file:///home/marios/Projects/Archimedes2026/src/llm_explainer.py) | Translates the verified statistical JSON of each ICP into an executive Greek narrative (`temperature=0.0, seed=2026`), strictly forbidding fabricated metrics or creditworthiness claims. |

### 5-Fold Cross-Validation Benchmark Results (`KFold(n_splits=5, shuffle=True, random_state=42)`)

| Evaluation Metric | Official Baseline (Primary KAD Frequency) | Archimedes2026 Hybrid ICP Engine | Improvement |
| :--- | :---: | :---: | :---: |
| **`Recall@500`** (Top 2.08% of Universe) | `1.69%` analytical (`1.00% ± 1.33%` empirical) | **`14.00% ± 3.43%`** | **+8.2x Lift** |
| **`Recall@2000`** (Top 8.31% of Universe) | `6.78%` analytical (`6.00% ± 3.27%` empirical) | **`30.67% ± 5.01%`** | **+4.5x Lift** |

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

We run deterministic K-Means (`init="k-means++", n_init=50, random_state=42`) across $K \in \{2,\dots,6\}$ and select the optimal $K^*$ maximizing a composite of weighted Silhouette, $B = 40$ Bootstrap Adjusted Rand Index ($\text{ARI}$) stability, balance penalty, and client-vs-Universe separation.

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

### 4.6 Unified Lookalike & Client Assignment Score

The exact same similarity function $S(i, k) \in [0, 1]$ scores both active clients (`assignments.csv`) and non-client Universe candidates (`lookalikes.csv`):

$$
S(i, k) = \frac{w_{\text{cos}}\,S_{\text{cos}}(i, k) + w_{\text{eb}}\,S_{\text{EB}}(i, k) + w_{\text{sp}}\,S_{\text{spat}}(i)}{w_{\text{cos}} + w_{\text{eb}} + w_{\text{sp}}}
$$

where:
- $S_{\text{cos}}(i, k) = \frac{1}{2}(\mathbf{z}_i^\top \boldsymbol{\mu}_k + 1)$ is the scaled cosine similarity to ICP $k$'s turnover-weighted centroid $\boldsymbol{\mu}_k$,
- $S_{\text{EB}}(i, k) = \left(1 + \exp\big(-0.55 \sum_c \gamma_c \ln\text{Lift}_{\text{EB}}(k, v_c(i))\big)\right)^{-1}$ is the sigmoid-squashed sum of Empirical Bayes log-lifts across Prefecture, Municipality, Legal Type, 8-digit Primary KAD, 4-digit NACE class, Age bracket, and Branch status, and
- $S_{\text{spat}}(i)$ is the 95th-percentile-normalized Haversine Gaussian kernel density ($\sigma_{\text{km}} = 25\text{ km}$) around active clients.

Clients whose best score $\max_k S(i, k)$ falls below the dynamic threshold $\tau_{\text{no-fit}} = \max(0.35,\; Q_{0.05}(S^*_{\text{clients}}) - 0.02)$ are assigned **`"NO_FIT"`**.

---

## 5. Data Sources, Licenses & EU AI Act Compliance

- **Registry & Taxonomy Data:** ΓΕΜΗ (General Commercial Registry) & ELSTAT KAD 2025 Open Data, provided by ForceCRM for the NKUA Hackathon.
- **GPS Coordinates:** © OpenStreetMap contributors, licensed under the **Open Database License (ODbL)**.
- **EU AI Act Compliance (Regulation (EU) 2024/1689):**
  - **Not High-Risk (Annex III):** B2B commercial segmentation and lead prioritization is outside Annex III high-risk categories.
  - **Sole Proprietorship Safeguard (`ΑΤΟΜΙΚΗ`, 12,090 records):** Our system **never evaluates creditworthiness or financial solvency (`φερεγγυότητα`)**. All scores strictly measure B2B Commercial Fit / ICP Similarity.
  - **Article 50 Transparency:** All AI-generated Greek narratives carry an explicit disclosure badge and are restricted to verified statistical traits.
