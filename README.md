# Bayan — NLP with Transformers Labs

**Author:** Moudi Alhomoud  

**Training Program:** [SDAIA Academy](https://github.com/SDAIAAcademy)  

**Trainer:** Aljawhara Albahlal

**Instructor Repository:** [SDA-AIE-211-Bayan-Course](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course)  

**My Repository:** [Bayan](https://github.com/Mhamad-14/Bayan)



## Overview

This repository contains my implementation and experimental work for the **SDA-AIE-211 NLP with Transformers** Bayan training labs.

The work completed so far covers:

- Arabic/English text preprocessing and tokenizer auditing
- Transformer attention and parameter analysis
- Leakage-safe grouped dataset splitting
- TF-IDF + LinearSVC topic-classification baseline
- Transformer topic classification with CAMeLBERT
- Named Entity Recognition (NER) with subword BIO-label alignment
- Extractive Question Answering (QA) span selection and smoke testing

Labs 1–7 are implemented and evaluated below. Targets that were not achieved are reported explicitly.

---

## Progress

| Lab | Topic | Status |
|---|---|---|
| Lab 1 | Preprocessing + Tokenizer Audit | Completed |
| Lab 2 | Transformer Anatomy + Attention | Completed |
| Lab 3A | Topic Classification | Implemented and evaluated |
| Lab 3B | NER + Extractive QA | Completed |
| Lab 4 | Arabic Pipeline + Dialect-Aware Fine-tuning | Completed |
| Lab 5 | Search / Retrieval | Implemented; retrieval targets not met |
| Lab 6 | Evaluation | Completed |
| Lab 7 | Optimisation / Serving | Completed; HTTP p99 target not met |

---

## Lab 1 — Preprocessing and Tokenizer Audit

Lab 1 focused on building and validating the shared preprocessing pipeline and comparing multilingual and Arabic-focused tokenizers.

### Completed work

- Implemented the course preprocessing pipeline.
- Validated golden preprocessing examples.
- Validated PII masking.
- Audited tokenizer fertility, sequence lengths, and unknown-token rates.
- Compared mBERT, XLM-R, CAMeLBERT, and DistilBERT.

### Results

| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.143 | 1.509 | 27.0 | 25.0 | 0.455% |
| XLM-R | 1.667 | 1.433 | 21.0 | 23.0 | 0.000% |
| CAMeLBERT | 1.400 | 2.704 | 20.0 | 38.0 | 0.807% |
| DistilBERT | 4.517 | 1.297 | 47.0 | 21.0 | 0.216% |

Additional validation:

- Golden preprocessing: **25 / 25 passed**
- PII masking recall: **60 / 60 = 100%**

The tokenizer audit showed XLM-R as a strong balanced multilingual option, while CAMeLBERT achieved the lowest Arabic fertility among the audited models.

---

## Lab 2 — Transformer Anatomy and Attention

Lab 2 focused on understanding and validating transformer attention behavior rather than only using a pretrained model as a black box.

### Completed work

- Implemented scaled dot-product attention.
- Compared the implementation against PyTorch attention.
- Verified causal masking.
- Audited model parameter allocation.
- Inspected padding attention behavior.
- Examined attention-head behavior, including adjacency-like and `[SEP]`-sink patterns.

### Selected results

| Metric | Result |
|---|---:|
| Attention max difference vs PyTorch | 0.0000002384 |
| Attention equivalence within `1e-6` | Passed |
| Maximum future attention with causal mask | 0.0000000000 |
| mBERT total parameters | 177,853,440 |
| mBERT embedding share | 51.84% |
| CAMeLBERT total parameters | 109,081,344 |
| CAMeLBERT embedding share | 21.48% |
| PAD attention without mask | 0.046848 |
| PAD attention with mask | 0.000000 |

---

## Lab 3A — Topic Classification

Lab 3A builds a topic-classification pipeline with both a traditional machine-learning baseline and a transformer model.

### Leakage-safe grouped split

The dataset is split by `citizen_group_id` using grouped splitting so that the same citizen does not appear in more than one split.

Final split sizes:

| Split | Rows |
|---|---:|
| Train | 8,389 |
| Validation | 2,403 |
| Frozen test | 1,208 |

Citizen-group overlap across train, validation, and test was verified to be **0**.

### TF-IDF + LinearSVC baseline

The baseline uses the same grouped splits as the transformer classifier.

| Metric | Validation | Frozen test | Train time |
|---|---:|---:|---:|
| Macro-F1 | 1.0000 | 1.0000 | 0.86 s |

### CAMeLBERT topic classifier

Checkpoint:

`CAMeL-Lab/bert-base-arabic-camelbert-mix`

Training configuration includes:

- Hugging Face `Trainer`
- Matching tokenizer
- `max_length=256`
- Validation macro-F1 and accuracy
- Best-checkpoint selection
- One frozen-test evaluation after model selection
- Model/tokenizer artifact saving

Results:

| Metric | Validation | Frozen test |
|---|---:|---:|
| Macro-F1 | 1.0000 | 1.0000 |
| Accuracy | 1.0000 | 1.0000 |

Training time: **367.32 seconds** on a Tesla T4 GPU.

### Baseline-saturation note

The measured TF-IDF baseline already reached the maximum possible macro-F1 of **1.0000**, so a further `+0.08` macro-F1 improvement is mathematically impossible on this run.

The required citizen-group split remains leakage-safe with zero citizen overlap. A diagnostic also found repeated cleaned feedback text across different citizen groups, which helps explain why the classification task is unusually easy for TF-IDF.

---

## Lab 3B — Named Entity Recognition

The NER work implements correct BIO label alignment between word-level annotations and tokenizer subwords.

### Completed work

- Implemented `align_labels()`.
- Labels are assigned to the first subword of each word.
- Special tokens and non-first subword pieces are masked with `-100`.
- Fine-tuned `AutoModelForTokenClassification`.
- Evaluated with `seqeval` entity-level metrics.
- Saved the trained model and tokenizer artifact.

Checkpoint used:

`xlm-roberta-base`

Results:

| Metric | Validation | Frozen test | Train time |
|---|---:|---:|---:|
| Entity-level F1 | 1.0000 | 1.0000 | 73.89 s |

The NER alignment contract tests passed successfully.

---

## Lab 3B — Extractive Question Answering

The QA component implements constrained answer-span selection with support for an honest null/no-answer path.

### Completed work

- Implemented `best_span()`.
- Rejects invalid and inverted spans.
- Enforces a maximum answer length.
- Compares the best candidate span against a null score.
- Ran the supplied 12-question smoke set.

Current smoke-test checkpoint:

`deepset/xlm-roberta-base-squad2`

Results:

- Answerable smoke questions: **12 / 12 correct**
- No-answer evaluation: **20 / 20 correct**
- Course no-answer target: **>= 17 / 20 — met**
- QA smoke and null-handling checks: **Passed**

---

## Lab 3 Contract Verification

The final Lab 3 contract suite completed successfully:

```text
11 passed
```

A final unfinished-code scan also reported:

```text
No unfinished Lab 3 TODOs found
```

---

## Lab 3 — Sentiment Evaluation

CAMeLBERT sentiment classifier results:

- Validation macro-F1: **0.5556**
- Validation accuracy: **0.8000**
- Frozen-test macro-F1: **0.5556**
- Frozen-test accuracy: **0.8000**
- Frozen-test class F1: negative **1.0000**, neutral **0.0000**, positive **0.6667**

The neutral class was not predicted on the frozen test, so this is reported as a model limitation rather than hidden.

---

## Lab 4 — Arabic Model Bake-off

| Model | Macro-F1 all | Gulf | MSA |
|---|---:|---:|---:|
| CAMeLBERT-mix | 0.9992 | 1.0000 | 1.0000 |
| CAMeLBERT-DA | 0.9992 | 1.0000 | 1.0000 |

NER LOCATION recall remained **1.0000** before and after segmentation. The incumbent CAMeLBERT-mix was retained because the Gulf-slice result tied.

---

## Lab 5 — Semantic Search

| Configuration | Recall@10 | MRR@10 | p50 |
|---|---:|---:|---:|
| Supplied BM25 evidence | 0.6933 | 0.7567 | — |
| Bi-encoder | 0.0256 | 0.0175 | 9.56 ms |
| Cross-encoder reranked | 0.0026 | 0.0026 | 78.65 ms |

- Same-language Recall@10: **0.0038**
- Cross-language Recall@10: **0.0000**
- No-answer correctness: **20 / 20**
- Recall@10 target >= 0.80: **not met**
- MRR@10 target >= 0.70: **not met**

The supplied corpus contains substantial duplicate text while each answerable query has only three judged relevant case IDs. The relevance labels were not modified to inflate results.

---

## Lab 6 — Evaluation

| Macro-F1 [95% CI] | Accuracy [95% CI] | Invariance | MFT |
|---|---|---:|---:|
| 0.8333 [0.8290, 0.8373] | 0.8750 [0.8617, 0.8888] | 0.8000 | 0.9600 |

- MFT target: **met**
- Invariance target ~0.95: **not met**
- Manual error review: **120 examples**
- Model cards are included in `model_cards/`.

---

## Lab 7 — Optimisation and Serving

| Rung | p50 | p99 | Speed-up |
|---|---:|---:|---:|
| PyTorch FP32 @512 | 150.02 ms | 159.14 ms | 1.00x |
| PyTorch FP32 @128 | 30.21 ms | 32.09 ms | 4.96x |
| ONNX FP32 @128 | 9.92 ms | 17.20 ms | 9.25x |
| ONNX INT8 @128 | 3.70 ms | 6.08 ms | 26.16x |

INT8 macro-F1 remained **1.0000** with **0.0000 quality tax**.

HTTP load test with 16 concurrent clients:

- Responses: **36,530 HTTP 200**
- Errors: **0**
- Throughput: **622.08 req/s**
- HTTP p99: **55 ms**
- Classifier concurrency is capped at **8 in-flight predictions** to reduce CPU contention and tail latency.
- Target <= 40 ms: **not met**


---

## Environment

The GPU-based Lab 3 work was run in Google Colab using:

- Python **3.12**
- PyTorch
- Hugging Face Transformers
- Accelerate
- scikit-learn
- seqeval
- Tesla T4 GPU

Large trained artifacts are stored outside the Git repository, with Google Drive used for model outputs when appropriate.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Mhamad-14/Bayan.git
cd Bayan
```

Create/activate a Python 3.12 environment and install the project dependencies according to the course repository.

Run the main Lab 3 checks:

```bash
python -m pytest tests/test_model_data.py tests/test_ner_alignment.py tests/test_qa.py -q
python scripts/tfidf_baseline.py
python scripts/qa_smoke.py
```

GPU training scripts:

```bash
python scripts/train_classifier.py
python scripts/train_ner.py
```

A custom artifact output directory can be supplied with `--output-dir`.

---

## Repository Notes

- Benchmark values in `BENCHMARKS.md` are from my own runs.
- Frozen-test results are recorded only after model development/model selection.
- Large model weights are intentionally kept outside Git.
- Labs 1–7 are represented with measured evidence; unmet targets are reported explicitly.

---

## Acknowledgements

This work was completed as part of the **SDAIA Academy** training program.

- Training program: https://github.com/SDAIAAcademy
- Instructor course repository: https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course

**Moudi Alhomoud**



## Integrated Bayan API

The final service exposes the following FastAPI endpoints:

- `GET /health` — startup health, preprocessing version, canary status, and active classifier artefact.
- `POST /v1/classify` — single-text topic classification.
- `POST /v1/classify:batch` — batch topic classification capstone extension.
- `POST /v1/entities` — NER extraction using the trained XLM-R token-classification artefact.
- `POST /v1/search` — bilingual semantic case search using the versioned FAISS index and cross-encoder reranking.
- `POST /v1/analyse` — end-to-end classification, entity extraction, and similar-case retrieval.

Large generated model and serving artefacts are intentionally excluded from Git. Reproduce the NER artefact with:

`python scripts/train_ner.py`

The service is started with:

`make serve`

The serving command constrains CPU threading for stable FAISS/Transformer execution on the lab environment.
