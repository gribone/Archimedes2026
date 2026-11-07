# ForceCRM Model & Data Card

7 November 2026 · [Two-page printable card](ForceCRM-Model-Data-Card.pdf) · [Responsible AI assessment](RESPONSIBLE_AI.md)

## Intended use and data

ForceCRM recommends Greek foodservice businesses for a sales representative to review. Its inputs are customer VATs, indicative revenue, registry activities, legal form, geography and incorporation date. The supplied workbook contains 300 distinct customers. All matched the registry. The three CSVs contain 24,424 venues, 23,945 VATs and 3,736 activity descriptions. GPS coverage is 96.01%. All activity IDs have descriptions. One row has the unknown activity type `Λοιπή` and is ignored.

Business data comes from GEMI Open Data. Geocoding is powered by OpenStreetMap contributors under ODbL. ForceCRM supplied these extracts exclusively for competition use. Coordinates support local geographic features. Offline venue matching uses an ODbL OpenStreetMap extract. Market context combines ELSTAT municipal hotel nights in 2024 with 2021 census residents. The prospects map uses a public-domain Natural Earth outline. An optional, opt-in live check queries the ΓΕΜΗ Open Data API (ODC-BY 1.0) for the top prospects' registry status. It sends only their ΑΦΜ and needs the user's own key. [Sources, coverage, terms and limitations](ENRICHMENT.md).

## Model selection

**Ministral 3 3B Instruct** runs through Ollama using Q4_K_M weights. Mistral AI developed the family in France. European development, an Apache-licensed distribution and successful structured Greek drafting drove the choice. The approximately 3 GB package runs on CPU. Its precise training corpus is incompletely disclosed. The family report describes the training approach. [Model report](https://arxiv.org/abs/2601.08584) · [Exact quantized package](https://ollama.com/library/ministral-3:3b)

The distributed Ollama model license contains Apache 2.0, which permits commercial use subject to its terms. The upstream model tree labels the weights Apache 2.0 and lacks a separate LICENSE file. [Inspected distribution license](https://ollama.com/library/ministral-3:3b/blobs/43070e2d4e53) · [Upstream tree](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/tree/main)

Each full portfolio analysis generates a new Greek paragraph per profile with the local model. Every profile fact is first written as a Greek template sentence. The model rewrites one sentence at a time. Numbers, percentages and names are checked against the source fact, and misleading wording is rejected. A sentence that fails twice keeps its verified template. Every description ends with a computed sentence stating profile quality, bootstrap stability and fit confidence. The run records the model digest, prompt hash and rejected drafts. Generation uses temperature 0, seed 42, top-k 1, a 2,048-token context and a 220-token cap per sentence.

The comparison uses Qwen3 1.7B from Alibaba in China and Gemma 3 1B from Google in the United States. These smaller models fit the test laptop. All models receive the same three profiles, prompt and checks.

| LLM | LLM-written sentences (passed all checks) | Time per profile | Ollama memory |
|---|---:|---:|---:|
| Ministral 3 3B | 11/18 (61%) | 50.0 s | 2.33 GiB |
| Qwen3 1.7B | 18/18 (100%) | 32.2 s | 1.13 GiB |
| Gemma 3 1B | 17/18 (94%) | 16.8 s | 0.82 GiB |

The smaller alternatives follow the few-shot patterns more closely, so more of their sentences pass the checks, and they run faster. All three produced correct Greek on the shared profiles. Ministral remains the default for its European provenance and Apache-licensed distribution; the brief prefers European models. A native Greek speaker has yet to rate fluency or narrative coherence. [Qwen package license](https://ollama.com/library/qwen3:1.7b/blobs/d18a5cc71b84) · [Gemma package terms](https://ollama.com/library/gemma3:1b/blobs/dd084c7d92a3)

**Multilingual E5-small** produces 384-dimensional vectors from Greek NACE descriptions. Microsoft Research's E5 team used multilingual text pairs from sources including mC4, CC News, NLLB and Wikipedia, followed by supervised retrieval training. The model has approximately 118M parameters. The tested revision is `614241f622f53c4eeff9890bdc4f31cfecc418b3`. [Publisher model card](https://huggingface.co/intfloat/multilingual-e5-small)

The weight repository declares MIT in its metadata. Inspection of its file list found no standalone license file. The E5 source repository contains an MIT license that permits commercial use under its terms. The weight-specific record consists of the publisher's declaration. [Exact weight tree](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) · [Official source license](https://github.com/microsoft/unilm/blob/master/LICENSE)

## Measured performance

The embedding comparison encoded 75 Greek foodservice descriptions. Eligible queries had at least one sibling in the same four-digit activity family. A hit means at least one sibling appeared among the five nearest descriptions. This metric measures recovery of the code hierarchy. Human relevance judgments would provide a separate semantic quality assessment.

| Encoder | Family hit@5 | Milliseconds per description | Python RSS after encoding |
|---|---:|---:|---:|
| E5-small | 93.2% | 17.6 | 300 MiB |
| E5-base | 93.2% | 87.2 | 330 MiB |
| Multilingual MiniLM | 89.0% | 15.9 | 381 MiB |

E5-small tied E5-base on this proxy while encoding descriptions about 5 times faster. The hit@5 values reproduce the original author's Intel laptop measurements exactly; timings come from the Apple A18 Pro, 8 GiB machine. RSS includes interpreter and allocator memory. See the benchmark runner and the local results in `outputs/embedding-comparison.json` and `outputs/llm-comparison.json`.

On an Apple A18 Pro with 8 GiB RAM, the portfolio run took 153 seconds with cached embeddings and fresh Greek drafts. Each of three descriptions took 42–54 seconds. Ollama held 2.33 GiB for Ministral. Python RSS at completion was 173 MiB. (The earlier constrained-variant design took 584.5 seconds on an Intel Core i3-1215U with 7.7 GiB RAM.)

| Internal evaluation on 60 held-out VATs | ForceCRM | Primary-code baseline |
|---|---:|---:|
| Recall@500 | 15/60 (25.0%) | 1/60 (1.7%) |
| Recall@2,000 | 33/60 (55.0%) | 5/60 (8.3%) |

The seeded split trained on the other 240 customers. The organizer retains a separate hidden set. The three profiles had mean VAT-bootstrap Jaccard scores of 93.0%, 88.4% and 91.4%. All 24,424 venues received a score, 4,564 fell outside their nearest profile radius, and ranking yielded 23,645 distinct prospective VATs.

## Failure modes and disclosure

Self-declared activities may be inactive. GPS can be missing or inaccurate. Multiple venues may represent different business types under a single VAT. Statistical tests performed after clustering provide exploratory evidence. Fit confidence measures similarity to a cluster and has no calibration against conversion or credit outcomes. On the supplied portfolio the profiles are mainly geographic, because all ΚΑΔ 56 activity embeddings are very similar and the customers' activity mix is close to the market's. Business traits shared across regions are reported separately (ΙΚΕ 2.4×, school canteens 2.4×). Human reviewers can inspect sample sizes, lift, corrected p-values, stability and the evidence for every generated sentence. Profiles concern B2B outreach. Creditworthiness, solvency and eligibility decisions fall outside the intended use.

OpenAI Codex assisted architecture, implementation, documentation, testing and debugging. Claude Code (Anthropic) assisted code review, holdout verification, setup fixes, the per-sentence grounded Greek narration and documentation. Runtime inference uses only the local models described here.
