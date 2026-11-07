# ForceCRM Model & Data Card

7 November 2026 · [Two-page printable card](ForceCRM-Model-Data-Card.pdf) · [Responsible AI assessment](RESPONSIBLE_AI.md)

## Intended use and data

ForceCRM recommends Greek foodservice businesses for a sales representative to review. Its inputs are customer VATs, indicative revenue, registry activities, legal form, geography and incorporation date. The supplied workbook contains 300 distinct customers. All matched the registry. The three CSVs contain 24,424 venues, 23,945 VATs and 3,736 activity descriptions. GPS coverage is 96.01%. All activity IDs have descriptions. One row has the unknown activity type `Λοιπή` and is ignored.

Business data comes from GEMI Open Data. Geocoding is powered by OpenStreetMap contributors under ODbL. ForceCRM supplied these extracts exclusively for competition use. Coordinates support local geographic features. Offline venue matching uses an ODbL OpenStreetMap extract. Market context combines ELSTAT municipal hotel nights in 2024 with 2021 census residents. [Sources, coverage, terms and limitations](ENRICHMENT.md).

## Model selection

**Ministral 3 3B Instruct** runs through Ollama using Q4_K_M weights. Mistral AI developed the family in France. European development, an Apache-licensed distribution and successful structured Greek drafting drove the choice. The approximately 3 GB package runs on CPU. Its precise training corpus is incompletely disclosed. The family report describes the training approach. [Model report](https://arxiv.org/abs/2601.08584) · [Exact quantized package](https://ollama.com/library/ministral-3:3b)

The distributed Ollama model license contains Apache 2.0, which permits commercial use subject to its terms. The upstream model tree labels the weights Apache 2.0 and lacks a separate LICENSE file. [Inspected distribution license](https://ollama.com/library/ministral-3:3b/blobs/43070e2d4e53) · [Upstream tree](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512/tree/main)

The model selects and orders verified Greek sentence variants. Each sentence references its statistic. This design limits stylistic freedom and makes grounding auditable. Inference uses temperature 0, seed 42, top-k 1, a 2,048-token context, a 240-token output cap and four CPU threads. Each profile records the model digest and the validated response. A digest-and-prompt cache replays prior drafts.

The comparison uses Qwen3 1.7B from Alibaba in China and Gemma 3 1B from Google in the United States. These smaller models fit the test laptop. All models receive the same Greek evidence and schema. The printable card reports actual validity and latency. Both smaller alternatives generated valid output on the first shared profile in less time than Ministral. The selection favors European provenance and the distribution license. A native Greek speaker has yet to rate fluency or narrative coherence. [Qwen package license](https://ollama.com/library/qwen3:1.7b/blobs/d18a5cc71b84) · [Gemma package terms](https://ollama.com/library/gemma3:1b/blobs/dd084c7d92a3)

**Multilingual E5-small** produces 384-dimensional vectors from Greek NACE descriptions. Microsoft Research's E5 team used multilingual text pairs from sources including mC4, CC News, NLLB and Wikipedia, followed by supervised retrieval training. The model has approximately 118M parameters. The tested revision is `614241f622f53c4eeff9890bdc4f31cfecc418b3`. [Publisher model card](https://huggingface.co/intfloat/multilingual-e5-small)

The weight repository declares MIT in its metadata. Inspection of its file list found no standalone license file. The E5 source repository contains an MIT license that permits commercial use under its terms. The weight-specific record consists of the publisher's declaration. [Exact weight tree](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) · [Official source license](https://github.com/microsoft/unilm/blob/master/LICENSE)

## Measured performance

The embedding comparison encoded 75 Greek foodservice descriptions. Eligible queries had at least one sibling in the same four-digit activity family. A hit means at least one sibling appeared among the five nearest descriptions. This metric measures recovery of the code hierarchy. Human relevance judgments would provide a separate semantic quality assessment.

| Encoder | Family hit@5 | Milliseconds per description | Python RSS after encoding |
|---|---:|---:|---:|
| E5-small | 93.2% | 59.5 | 804 MiB |
| E5-base | 93.2% | 195.5 | 1,066 MiB |
| Multilingual MiniLM | 89.0% | 50.3 | 827 MiB |

E5-small tied E5-base on this proxy while encoding descriptions about 3.3 times faster. RSS includes interpreter and allocator memory. See the benchmark runner and the local results in `outputs/embedding-comparison.json` and `outputs/llm-comparison.json`.

On an Intel Core i3-1215U with 7.7 GiB RAM, the original portfolio run took 584.5 seconds with cached embeddings and fresh Greek drafts. Each of three descriptions took 170–188 seconds. Ollama reported 2.34 GiB allocated during CPU inference. Python RSS at completion was 173 MiB. A subsequent web run with both caches populated took 31.4 seconds and produced byte-identical assignment and recommendation CSVs.

| Internal evaluation on 60 held-out VATs | ForceCRM | Primary-code baseline |
|---|---:|---:|
| Recall@500 | 13/60 (21.7%) | 1/60 (1.7%) |
| Recall@2,000 | 33/60 (55.0%) | 5/60 (8.3%) |

The seeded split trained on the other 240 customers. The organizer retains a separate hidden set. The three profiles had mean VAT-bootstrap Jaccard scores of 93.0%, 88.4% and 91.4%. All 24,424 venues received a score, 4,564 fell outside their nearest profile radius, and ranking yielded 23,645 distinct prospective VATs.

## Failure modes and disclosure

Self-declared activities may be inactive. GPS can be missing or inaccurate. Multiple venues may represent different business types under a single VAT. Statistical tests performed after clustering provide exploratory evidence. Fit confidence measures similarity to a cluster and has no calibration against conversion or credit outcomes. Human reviewers can inspect sample sizes, lift, corrected p-values, stability and the evidence for every generated sentence. Profiles concern B2B outreach. Creditworthiness, solvency and eligibility decisions fall outside the intended use.

OpenAI Codex assisted architecture, implementation, documentation, testing and debugging. Runtime inference uses the local models described here.
