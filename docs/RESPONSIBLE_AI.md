# Responsible AI self-assessment

This is an engineering scope assessment, not a certification or legal opinion.

## What does the system recommend?

It groups an uploaded company's customers and ranks other businesses by similarity to those groups. A person decides whether to research or contact a prospect. The system neither sends messages nor sets prices, determines eligibility, predicts credit risk, estimates solvency, or recommends lending. Revenue describes the user's past sales, not a natural person's financial reliability.

## Who is impacted?

Sales representatives, the uploading company, and businesses listed as prospective customers. Sole proprietorships may identify natural persons; public registry availability does not erase privacy obligations. Keep access restricted, minimize uploads to VAT/revenue, use the supplied competition purpose, and do not publish raw datasets. Generated run folders contain recommendations and portfolio aggregates. The user controls retention by deleting selected local run folders and narrative caches after use. The app does not implement organization-wide access control or production retention automation.

## What can the end user inspect?

Each profile exposes counts, revenue share, measured traits, Fisher p-values, globally adjusted q-values, lift, membership stability, fit confidence, and acceptance distance. Every displayed narrative sentence maps to a fact identifier. AI selection/ordering is explicitly labeled; demo text is labeled as a deterministic template. Candidate justifications identify matching supported traits when available and otherwise state the model's feature basis. Exploratory candidates outside the acceptance radius are identified, not disguised as confident matches.

## Where can it fail, and how is this detected?

| Failure | Detection / response |
|---|---|
| Inactive secondary registrations | Primary-weighted representation, semantic filtering and capped secondary influence; human verification remains necessary |
| Branch duplication or leakage | VAT normalization, revenue divided by venue count, VAT bootstrap and VAT holdout, existing-customer exclusion across all venues |
| Chance features | Minimum five independent VATs, lift threshold, global BH correction; explicitly exploratory post-clustering inference |
| Unstable cohorts | Per-profile bootstrap Jaccard and global ARI; limited-quality label |
| Hallucinated text | Restricted evidence IDs and sentence variants; invalid LLM output stops real analysis |
| Poor retrieval | Internal holdout against primary-NACE baseline; real hidden-set performance cannot be inferred from the demo |
| Geographic/legal-form bias | Visible feature weights and distributions; similarity is never presented as credit or conversion probability |
| Broken/absent local models | Explicit setup errors; no silent cloud call or demo substitution |

## Regulatory scope and literacy

The intended B2B prospect-research use is not designed for an Annex III high-risk decision. Repurposing it for creditworthiness or natural-person eligibility would require a fresh assessment and is outside this application. AI-generated descriptions are visibly labeled regardless of whether a particular legal transparency obligation applies. Developers and operators should understand model inputs, branch handling, confidence limitations, human oversight and failure modes, addressing the AI literacy objective.

The challenge brief contains specific claims about the Digital Omnibus and delayed deadlines. **Do not copy those claims into a compliance declaration without checking enacted legislation.** This implementation does not depend on them. Consult the [European Commission's current AI Act overview](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) and [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) for actual scope, dates and obligations. A project's own labeling is not evidence of full EU AI Act or GDPR compliance.

## Local boundary

Application serving is loopback-only. The browser has no CDN or map requests; model inference addresses literal loopback and ignores proxy environment variables. Embedding loading enforces local files and offline flags. Explicit setup tools contact model registries to download software/weights. Those setup downloads are separate from analysis. Colab is an optional cloud development environment and is not represented as a local deployment.
