# Competition verification

The attached [five-split results](ranking-audit.json) compare both versions on identical training VATs and candidate pools. Each split withholds 60 of 300 customers. Test sets overlap across splits. Only training customer labels enter the concentration estimates. The registry supplies unlabeled universe frequencies.

| Mean VAT recall | New ranking | Previous cosine ranking | Primary-NACE baseline |
|---|---:|---:|---:|
| Top 500 | 18.0% | 10.0% | 1.3% |
| Top 2,000 | 50.3% | 28.0% | 8.0% |

Both cutoffs improved on all five splits. This measures internal retrieval performance. The organizer holds the labels needed for an official result. Reproduce with `python -m scripts.audit_ranking` in the provisioned environment.

| Requirement | Verification and limits |
|---|---|
| R1 | Local E5 embeds Greek activity descriptions. Capped relevant secondary activities, GPS, prefecture, legal form and age contribute to the venue vector. |
| R2 | The discovery routine considers 1–6 profiles. The homogeneous-input regression test exercises the single-profile case. Revenue weights are capped and divided across venues. |
| R3 | Formation and assignment use cosine distance. The export includes a score and permits NONE. The additional market concentration signal changes prospect ordering only. |
| R4 | Fisher tests use VAT-level support. A global BH correction, minimum support and lift threshold filter the reported traits. Post-clustering significance remains exploratory. |
| R5 | Each profile has a VAT-bootstrap membership Jaccard score. The report also records global ARI. Stability is conditional on the chosen profile count. |
| R6 | The local LLM orders verified Greek sentence variants. Exports identify AI text and record evidence, seed, digest, confidence and quality. Independent Greek fluency ratings remain outstanding. |
| R7 | The validator verifies unique candidate VATs, existing-customer exclusion and global rank order. Each rationale now includes cosine similarity and relative customer concentration. |
| R8 | The localhost app accepts the supplied Excel headers and exposes profiles, evidence, recommendations and exports. The real CLI run produced three profiles, 24,424 assignments and 23,645 prospects. |
| Model selection | Three embedding models and three LLMs were compared locally. The card records timing, provenance and licenses. The E5 weight repository declares MIT but lacks a separate weight-license file. |
| Responsible AI | The self-assessment covers recommendations, impacted businesses, transparency and failures. The application supports sales research and excludes credit scoring. |
| Delivery and optional points | Source, localhost launcher, Colab instructions, fixed-format exports and two-page card are provided. GPS positioning implements EP1. EP2 venue linkage and EP3 additional open datasets remain unimplemented. Hosted Colab execution and an independent clean-laptop reproduction remain unverified. |

The ranking multiplies cosine similarity by L/(1+L), where L is a weighted geometric mean of observed-to-expected customer concentrations. Counts are VAT-balanced and regularized with five expected customers. Inputs are prefecture, municipality, primary activity, prefecture plus activity, legal form and incorporation year. The score prioritizes calls and is uncalibrated as a purchase probability.
