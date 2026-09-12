# Model Card — Bayan Semantic Search

## Intended use
Retrieve and rerank bilingual historical Bayan cases.

## Artefact / data versions
- Model/checkpoint: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 + multilingual cross-encoder
- Preprocessing version: bayan_search_v1
- Data version/snapshot: 20k Bayan historical case corpus

## Metrics
| Metric | Result |
|---|---:|
| bi-encoder recall@10 | 0.0256 |
| bi-encoder MRR@10 | 0.0175 |
| reranked recall@10 | 0.0026 |
| reranked MRR@10 | 0.0026 |
| no-answer correctness | 20/20 |

## Slice metrics
| Slice | recall@10 |
|---|---:|
| same-language | 0.0038 |
| cross-language | 0.0000 |

## Behavioural tests
Retrieval-labelled evaluation is documented in BENCHMARKS.md.

## Known limitations
Retrieval quality is substantially below the course targets: reranked Recall@10 is 0.0026 and MRR@10 is 0.0026, and cross-language Recall@10 is 0.0000. The supplied 20,000-case corpus contains 14,599 duplicate-text rows, while each answerable query provides only three judged relevant case IDs; exact-text duplicate cases may therefore be retrieved without being counted as relevant, and large tied groups can affect top-k case IDs. No-answer behaviour reached 20/20 on the supplied no-answer slice, but this result should not be assumed to generalise beyond that evaluation set.

## Contact / owner
Bayan course project
