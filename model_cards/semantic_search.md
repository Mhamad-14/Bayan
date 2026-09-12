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
| bi-encoder recall@10 | 0.0692 |
| bi-encoder MRR@10 | 0.0175 |
| reranked recall@10 | 0.0077 |
| reranked MRR@10 | 0.0026 |
| no-answer correctness | 20/20 |

## Slice metrics
| Slice | recall@10 |
|---|---:|
| same-language | 0.0077 |
| cross-language | 0.0000 |

## Behavioural tests
Retrieval-labelled evaluation is documented in BENCHMARKS.md.

## Known limitations
Lab 6:
Retrieval quality remains the main limitation. Bi-encoder recall@10 is 0.0692 and MRR@10 is 0.0175, while reranking decreases recall@10 to 0.0077 and cross-language recall@10 is 0.0000. No-answer behaviour is reliable at 20/20, but the current retrieval configuration should not be treated as production-ready.

## Contact / owner
Bayan course project
