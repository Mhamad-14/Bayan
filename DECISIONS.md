# Decision Records

## tokenizer
- Chosen checkpoint(s): xlm-roberta-base (XLM-R)
- Arabic fertility evidence: XLM-R achieved an Arabic fertility of 1.667, compared with 2.143 for mBERT, 1.400 for CAMeLBERT, and 4.517 for DistilBERT.
- English fertility evidence: XLM-R achieved an English fertility of 1.433, providing strong English tokenization while maintaining good Arabic performance.
- p95 length evidence: XLM-R produced p95 sequence lengths of 21 tokens for Arabic and 23 tokens for English.
- Operational trade-off / rationale: XLM-R provides the best balanced bilingual performance for Bayan. CAMeLBERT performed slightly better on Arabic but was substantially less efficient on English, while DistilBERT performed poorly on Arabic. XLM-R also achieved a 0.000% Arabic UNK rate in the audit.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:

## Arabic model

**Decision:** CAMeLBERT-mix

**Evidence:** CAMeLBERT-mix Gulf macro-F1 = 1.0000; CAMeLBERT-DA Gulf macro-F1 = 1.0000; DA-minus-mix delta = +0.0000.

**Rationale:** CAMeLBERT-mix is selected because the Gulf-slice macro-F1 tied, so the incumbent is retained to avoid unnecessary model churn. The decision is based on the Gulf slice rather than aggregate F1 alone.

**Method:** frozen mean-pooled encoder embeddings with the same logistic-regression linear probe on the leakage-safe Lab 3 grouped split.
