# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit

| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.143 | 1.509 | 27.0 | 25.0 | 0.455% |
| XLM-R | 1.667 | 1.433 | 21.0 | 23.0 | 0.000% |
| CAMeLBERT | 1.400 | 2.704 | 20.0 | 38.0 | 0.807% |
| DistilBERT | 4.517 | 1.297 | 47.0 | 21.0 | 0.216% |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%


## Lab 2 — Transformer anatomy

| Metric | Result |
|---|---:|
| Attention max difference vs PyTorch | 0.0000002384 |
| Attention equivalence within 1e-6 | Passed |
| Maximum future attention with causal mask | 0.0000000000 |
| mBERT total parameters | 177,853,440 |
| mBERT embedding share | 51.84% |
| CAMeLBERT total parameters | 109,081,344 |
| CAMeLBERT embedding share | 21.48% |
| PAD attention without mask | 0.046848 |
| PAD attention with mask | 0.000000 |
| Most adjacency-looking head | 7 |
| Adjacency score | 0.242445 |
| Strongest [SEP]-sink head | 2 |
| Mean [SEP] attention | 0.166924 |


## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
