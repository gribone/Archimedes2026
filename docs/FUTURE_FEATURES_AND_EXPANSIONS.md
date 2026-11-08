# Archimedes 2026 — Strategic Roadmap & Future Feature Expansions

**Document Version:** 1.0.0  
**Date:** November 2026  
**Author:** AI Systems Architecture Team  
**Status:** Living Strategy Document  

---

## 1. Executive Summary

Archimedes 2026 currently serves as a B2B Ideal Customer Profile (ICP) Discovery and Lookalike Recommendation Engine tailored for the Greek Food & Beverage (F&B) and Hospitality ecosystem (NACE sector 56, 24,424 active businesses).

With the completion of **Milestone M1 (Test Safety Harness & Baseline Snapshot)** and the ongoing modularization into enterprise-grade data and engine pipelines, the platform is architected for expansion. This document outlines **strategic feature expansions, data integrations, and functional capabilities** that will transform Archimedes from a hackathon prototype into a continuous B2B Revenue Intelligence & Territory Execution Platform.

---

## 2. Core Feature Expansion Roadmap

### 2.1 Multi-Sector Domain Generalization
*Current State: Hardcoded to NACE code `56` (Food and Beverage Service Activities).*

- **Dynamic Sector Onboarding:** Expand ingestion beyond sector 56 to support:
  - **Wholesale Trade (NACE 46):** B2B distributors, beverage importers, food service supply chains.
  - **Retail Trade (NACE 47):** Delis, boutique grocers, organic markets, specialty wine shops.
  - **Hospitality & Accommodation (NACE 55):** Hotels, boutique resorts, serviced apartments.
  - **Professional Services & Technology (NACE 62 & 69):** Software vendors, accounting firms.
- **Hierarchical NACE 2.1 Auto-Clustering:** Instead of fixing NACE 56 as root, allow users to select multiple NACE 2-digit branches or upload mixed portfolios. The engine dynamically computes sector-specific baseline frequency distributions.

---

### 2.2 Active Feedback & Continuous Closed-Loop Learning
*Current State: Static lookalike scoring based strictly on initial client portfolio.*

- **Sales Rep Outcome Tagging:** In the web UI and via CRM API, allow reps to tag generated leads:
  - `🟢 Won / Closed Deal`
  - `🟡 Contacted / Opportunity Open`
  - `🔴 Disqualified / Bad Fit (with reason)`
- **Online Empirical Bayes & Centroid Drift:**
  - Positive outcomes (`Won`) automatically receive positive sample weights, gently nudging the corresponding ICP centroid toward high-converting lead profiles.
  - Disqualified reasons ("Too small", "Wrong concept") dynamically update feature penalty masks without requiring full retraining from scratch.
- **Active Learning Sampling:** Introduce an "Exploration vs Exploitation" toggle:
  - *Exploit Mode (Default):* Ranks candidates strictly by top similarity score.
  - *Explore Mode:* Surfaces high-uncertainty candidates close to cluster boundaries to map out untargeted market niches.

---

### 2.3 Financial Health, Balance Sheet & Credit Risk Enrichment
*Current State: Firmographics rely on age, legal form, branch count, and location; revenue is only available for the 300 client records.*

- **GEMI Financial Statement Ingestion (Ισολογισμοί Γ.Ε.ΜΗ.):**
  - Parse published financial statements for corporate entities (`Α.Ε.`, `Ε.Π.Ε.`, `Ι.Κ.Ε.`):
    - Net Turnover (Κύκλος Εργασιών)
    - EBITDA and Net Profit Margin
    - Total Assets and Debt-to-Equity Ratio
    - Days Sales Outstanding (DSO)
- **Solvency & Creditworthiness Pre-Filter:**
  - Compute a deterministic Commercial Solvency Index (0–100) separating high-value accounts from distressed businesses with high credit default risk.
  - *EU AI Act Article 50 Compliance:* Clearly demarcate commercial fit from financial credit scoring with transparent risk disclaimers.

---

### 2.4 Real-time GEMI Webhook & "Day-Zero" Lead Alerts
*Current State: Batch ingestion of historical company data (founded 2020–2024).*

- **Daily Incorporation Scraper / API Integration:**
  - Connect to official GEMI publicity announcements and the Official Government Gazette (ΦΕΚ Τεύχος Β').
  - Detect new hospitality and corporate business formations within **24 to 48 hours of incorporation**.
- **Day-Zero Sales Trigger:**
  - When a new business registers with KAD 56/47/55 matching an active ICP profile, generate an immediate high-priority lead alert.
  - In F&B and retail, purchasing decisions for POS systems, commercial software, and wholesale suppliers occur 30–60 days *before* the venue opens its doors.

---

### 2.5 Multi-Channel Digital Footprint & Tech Stack Detection
*Current State: Structured firmographic features + OSM distance matching.*

- **Digital Maturity Index:**
  - **Website & Domain Resolution:** Resolve company domain from GEMI records or automated web search.
  - **Technology Fingerprinting (BuiltWith / Wappalyzer approach):**
    - Detect current ecommerce/ordering platforms: WooCommerce, Shopify, Magento, custom portals.
    - Detect POS integrations: Cardlink, Viva Wallet, Worldline, Efood / Wolt delivery integrations.
  - **Social Media & Review Signals:**
    - Google Business Profile rating, review volume, and weekly review velocity.
    - Instagram presence and follower tiers (distinguishing viral/modern concepts from traditional outlets).

---

### 2.6 Geo-Spatial Territory Route Planning (Sales Blitz Optimizer)
*Current State: Leaflet map display + 1-click coordinate navigation.*

- **Field Sales Route Generation (Traveling Salesperson / Route Heuristics):**
  - Allow sales managers to select 10–25 lookalike venues in a specific municipality (e.g. Athens Center or Chania).
  - Automatically calculate an optimal walking or driving route with estimated visit duration (30 min per meeting).
- **Competitor Density Heatmaps:**
  - Layer kernel density estimates of competing clients vs unpenetrated lookalikes to identify underserved micro-markets.

---

### 2.7 Native 2-Way CRM Integration & Workflow Automation
*Current State: CSV download (`output/lookalikes.csv`, `output/assignments.csv`).*

- **Direct Push Connectors:**
  - **ForceCRM / Salesforce / HubSpot / Zoho:** One-click "Push to CRM as Leads" button in the UI.
  - Automatically populate custom CRM fields: `Archimedes_ICP_Profile`, `Archimedes_Score`, `Archimedes_Reason`, and `Sales_Playbook_Persona`.
- **Automated SDR Outbound Cadence Generation:**
  - Export tailored outbound email sequences and call scripts directly into sales engagement tools (Outreach, Salesloft, Apollo).

---

## 3. Architecture Evolution Matrix

| Horizon | Feature Area | Key Deliverable | Business Impact |
|:---:|:---|:---|:---|
| **Near-Term (Q1)** | Closed-Loop Feedback | Sales outcome tagging (Won/Lost) + Online EB Lift tuning | Dynamic adaptation to rep conversion reality |
| **Near-Term (Q1)** | CRM Connector | Direct HubSpot / ForceCRM webhook integration | Removes manual CSV handling; zero friction for reps |
| **Mid-Term (Q2)** | Multi-Sector | Dynamic NACE multi-code ingestion (Retail 47, Wholesale 46) | 5x expansion of Addressable Market (TAM) |
| **Mid-Term (Q2)** | Day-Zero Alerts | Real-time GEMI new incorporation monitor | First-mover advantage on brand new venues |
| **Long-Term (Q3)** | Financial Solvency | Published balance sheet parsing & DSO risk filters | Drastic reduction in client default / churn risk |
| **Long-Term (Q3)** | Route Planning | Traveling Salesperson daily itinerary generator | 30% increase in field rep meetings per day |

---

## 4. Preservation of Core Principles
All future expansions must adhere to Archimedes' core architectural pillars:
1. **Mathematical Determinism:** Every score is explainable with explicit feature lifts ($p_{\text{FDR}} \le 0.05$); zero ungrounded black-box predictions.
2. **Sub-Second Performance:** Maintain vectorized NumPy/Polars memory structures for sub-40ms parameter re-scoring.
3. **100% Data Sovereignty:** Offline-first architecture compatible with on-premise deployments and EU AI Act Article 50 transparency requirements.

