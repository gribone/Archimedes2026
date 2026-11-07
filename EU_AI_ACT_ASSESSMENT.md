# ⚖️ EU AI Act Self-Assessment & AI Tool Disclosure (Section 9 & 10)

**Project:** Archimedes2026 — ForceCRM B2B Profile Analysis & Lookalike Engine  
**Regulation Reference:** Regulation (EU) 2024/1689 (EU AI Act) & Regulation (EU) 2026/1744 (Digital Omnibus on AI)

---

## 1. Risk Classification & The Sole Proprietorship Safeguard

- **Risk Category:** **Minimal / Limited Risk (Transparency Obligations under Article 50).**
- **Why it is NOT High-Risk (Annex III):** Commercial B2B segmentation (grouping existing business clients into Ideal Customer Profiles and recommending similar prospective businesses for sales outreach) does not fall under any high-risk category in Annex III of the EU AI Act.
- **Neutralizing the Annex III Point 5(b) Trap (Sole Proprietorships / `ΑΤΟΜΙΚΗ`):**
  - Under Annex III Point 5(b) of the EU AI Act, AI systems used to evaluate the **creditworthiness (`πιστοληπτική ικανότητα`) or credit score (`φερεγγυότητα`) of natural persons** are classified as High-Risk (with compliance obligations taking effect on 2 December 2027 under Regulation (EU) 2026/1744).
  - In our dataset (`vwHackCompany.csv`), **12,090 out of 24,424 hospitality businesses (49.5%)**—and **118 out of 300 clients (39.3%)**—have `legal_type = ΑΤΟΜΙΚΗ` (Sole Proprietorships), which legally constitute **natural persons**.
  - **Architectural Guarantee:** Archimedes2026 **never computes, infers, or outputs creditworthiness, financial solvency (`φερεγγυότητα`), default probability, or credit risk scores**. Client `turnover` from the uploaded Excel is used strictly at the aggregate cluster level to weight the formation of B2B Ideal Customer Profiles. Every company-level score in `assignments.csv` and `lookalikes.csv` represents **B2B Commercial Fit / Morphological & Geographic Similarity** ($S \in [0, 1]$) to an ICP.

---

## 2. Four Mandatory Self-Assessment Questions (Section 9)

### Q1. Τι αποφασίζει ή προτείνει το σύστημα; (What does the system decide or recommend?)
The system makes **zero autonomous legal, financial, or contractual decisions**. It assists B2B sales teams by:
1. Grouping an uploaded list of existing B2B hospitality clients into **1 to 6 Ideal Customer Profiles (ICPs)** based on business activity (KAD descriptions & hierarchy), geography (GPS coordinates, local hospitality density, Prefecture/Municipality), legal form, company age, and branch structure.
2. Ranking non-client hospitality businesses from the 24,424 ΓΕΜΗ Universe (`lookalikes.csv`) by their B2B Commercial Fit / similarity to those ICPs so sales representatives can prioritize commercial outreach.

### Q2. Ποιος επηρεάζεται από αυτό; (Who is affected by it?)
- **Direct Users:** B2B sales representatives and marketing analysts using the tool to understand their customer portfolio and organize outbound sales campaigns.
- **Indirect Subjects:** Greek hospitality businesses in the ΓΕΜΗ registry (including 12,090 `ΑΤΟΜΙΚΗ` sole proprietorships). Because the output is used solely to prioritize B2B sales introductions—and never denies access to credit, essential services, insurance, or public benefits—no natural or legal person suffers adverse legal or fundamental-rights impacts.

### Q3. Τι βλέπει ο χρήστης για το πώς προέκυψε το αποτέλεσμα; (What does the user see regarding how the result was produced?)
Every output is fully transparent, auditable, and explainable (Article 50 & Article 4 AI Literacy):
1. **Explicit AI Content Labeling (Article 50, effective 2 August 2026):** Every Greek ICP description generated or formatted by the local AI module (`src/llm_explainer.py` & `profiles.json`) carries a prominent badge (`🤖 [Περιεχόμενο παραγόμενο από Τοπικό AI — Άρθρο 50 EU AI Act]`) and a mandatory disclaimer stating that scores measure B2B Commercial Fit and not solvency (`φερεγγυότητα`).
2. **Statistical Proof per ICP (`profiles.json` & UI Tab 1):** Users see the exact client count, turnover share, **Bootstrap Stability (Adjusted Rand Index / cluster stability %)**, **Confidence Score %**, and a table of defining characteristics backed by **Empirical Bayes Shrinkage Lift** and **Benjamini-Hochberg FDR-corrected Hypergeometric $p$-values**.
3. **Per-Lookalike Justification (`lookalikes.csv` & UI Tab 3):** Every recommended business includes a human-readable Greek explanation (`αιτιολόγηση`) citing the exact matching ICP, similarity percentage, geographic lift, legal form lift, and Primary KAD.

### Q4. Πού μπορεί να κάνει λάθος και πώς το εντοπίζουμε; (Where can it make mistakes and how do we detect them?)
| Potential Failure Mode | How We Detect & Mitigate It |
| :--- | :--- |
| **1. Declarative / Noisy Secondary KADs** (Companies declaring 50–239 KADs they do not practice) | Detected via `total_kads` diagnostics in `src/ingest.py`. Mitigated in `src/features.py` via Smooth IDF weighting ($\gamma_{\text{idf}}=1.25$), sublinear secondary-count damping ($n_{\text{sec}}^{-0.35}$), and L2 vector normalization. Verified via our Multi-KAD Fairness Check (`Mean Sim = 0.4615` for 1-KAD vs `0.4747` for $\ge 20$-KAD companies). |
| **2. Spurious Rare Features in Small Clusters** (A rare municipality or KAD appearing 1–2 times by luck) | Detected by comparing `raw_lift` vs `eb_lift`. Mitigated in `src/icp_engine.py` via Exact Hypergeometric tests with Benjamini-Hochberg FDR multiple-testing correction ($p_{\text{FDR}} \le 0.05$) and Empirical Bayes Shrinkage ($m_{\text{prior}}=10.0$). |
| **3. Unstable / Over-Segmented Clusters** | Detected via 30x Bootstrap resampling (`compute_bootstrap_stability`) reporting both Global Adjusted Rand Index (`ARI = 0.9849`) and per-profile stability in `profiles.json`, plus automatic fallback to $K=1$ («ένα ICP») if Silhouette $< 0.08$. |
| **4. Atypical Clients Forced into Wrong ICP** | Detected and prevented via the dynamic `NO_FIT` threshold in `assignments.csv` (flagging 14 atypical clients whose similarity score falls below the cutoff). |
| **5. LLM Hallucinations in Greek Descriptions** | Prevented by computing all numbers deterministically first, running the local LLM at `temperature = 0.0, seed = 2026` strictly on the statistical payload, and providing a 100% deterministic grounded Greek generator fallback. |

---

## 3. Declaration of AI Tools Used During Development (Section 6 & 10)

In accordance with Section 6 (*«Τα AI coding assistants επιτρέπονται στην ανάπτυξη. Δηλώνετε ποια εργαλεία χρησιμοποιήσατε και για ποιο σκοπό»*) and Section 10 of the ForceCRM Technical Brief, we declare the following development-time AI assistance:

- **Tool Used:** Google DeepMind **Antigravity** (Agentic AI Coding Assistant) in pair-programming mode with the developer.
- **Purpose & Scope of Use During Development:**
  1. **Data Exploration & Edge-Case Auditing:** Inspecting CSV schemas, discovering the `2026-03-01` NACE 2.1 multi-Primary-KAD transition (`993` companies) and the `239`-KAD noise outlier.
  2. **Boilerplate & Pipeline Scaffolding:** Assisting in writing modular Python code across `src/ingest.py`, `src/config.py`, `src/features.py`, `src/icp_engine.py`, `src/llm_explainer.py`, `evaluate_holdout.py`, and the Streamlit dashboard (`app.py`).
  3. **Documentation & Mathematical Typesetting:** Formatting the mathematical documentation in `README.md`, `MODEL_DATA_CARD.md`, and `EU_AI_ACT_ASSESSMENT.md`.
- **Runtime Guarantee:** During execution, the application makes **zero calls to external AI APIs or cloud services**. All embeddings, statistical tests, clustering, and LLM explanations execute 100% locally on the host machine.

