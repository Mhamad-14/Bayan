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
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | 0.86 s |
| Topic classifier (CAMeLBERT) | macro-F1 | 1.0000 | 1.0000 | 367.32 s |
| NER | entity-F1 | 1.0000 | 1.0000 | 73.89 s |
| QA | span/null smoke | 12/12 spans | 0/0 nulls | N/A |

- Topic classifier validation accuracy: 1.0000
- Topic classifier frozen-test accuracy: 1.0000
- Topic classifier frozen-test delta over TF-IDF baseline: +0.0000
- Grouped split citizen overlap: 0 across train/validation/test.
- The TF-IDF baseline saturated at macro-F1 = 1.0000 on this run, so the course target of +0.08 improvement is mathematically unattainable without exceeding the maximum F1 of 1.0000.
- Exact cleaned-text duplicates were present across different citizen groups, although citizen-group leakage remained zero.

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent (CAMeLBERT-mix) | 0.9992 | 1.0000 | 1.0000 | 1.9049 |
| Arabic dialect-aware (CAMeLBERT-DA) | 0.9992 | 1.0000 | 1.0000 | 1.9049 |
- NER LOCATION recall before segmentation: 1.0000
- NER LOCATION recall with segmentation: 1.0000
- LOCATION recall delta: +0.0000
- The Lab 3 baseline LOCATION recall was already 1.0000, so a +0.04 absolute improvement is mathematically impossible without exceeding the recall ceiling.

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| supplied BM25 baseline | 0.6933 | 0.7567 | — |
| bi-encoder only | 0.0692 | 0.0175 | 35.55 ms |
| + cross-encoder rerank | 0.0077 | 0.0026 | 1315.60 ms |

- same-language recall@10: 0.0077
- cross-language recall@10: 0.0000
- cross-lingual gap (same - cross): 0.0077
- no-answer empty-correct: 20 / 20
- tuned min_score threshold: 0.6651
- answerable retention at threshold: 1.0000
- benchmark target recall@10 >= 0.80: not met
- benchmark target MRR@10 >= 0.70: not met

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [95% CI] | Accuracy [95% CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | 0.8333 [0.8290, 0.8373] | 0.8750 [0.8617, 0.8888] | 0.5000 | 0.9600 |

- paired comparison: normal vs confidence-gated correctness delta +0.1000, 95% CI [+0.0883, +0.1117]
- error taxonomy top categories: Other / taxonomy extension needed, Arabic orthographic variation, Preprocessing or serving skew
- top-3 prioritised fixes: Review the new category and define a targeted remediation., Expand Arabic normalization and orthographic augmentation., Enforce one shared preprocessing contract across train/eval/serve.

## Lab 7 — Optimisation ladder
CPU evidence: `OMP_NUM_THREADS=4`, 160 requests sampled deterministically from `data/serving/bench_mix.npy`.

| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | 150.02 ms | 159.14 ms | —; speed-up 1.00× | — |
| fp32 torch @128 dynamic | 30.21 ms | 32.09 ms | —; speed-up 4.96× | — |
| ONNX fp32 @128 | 9.92 ms | 17.20 ms | —; speed-up 9.25× | 1060.9 MB |
| ONNX INT8 @128 | 3.70 ms | 6.08 ms | macro-F1 1.0000; tax +0.0000; 95% CI [+0.0000, +0.0000]; speed-up 26.16× | 266.0 MB |

- classifier bare p99 target <= 25 ms: met
- speed-up target >= 6x: met
- classifier quality-tax target <= 0.01 macro-F1: met
- HTTP p99, 16 concurrent: 74.00 ms; request errors: 0; target <= 40 ms: not met / inspect errors
- classifier quantisation decision: use INT8
- NER quantisation decision: see `artifacts/serving/ner_decision.json` if a local Lab 3 NER artefact is available.
