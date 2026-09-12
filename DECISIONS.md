# Decision Records

## tokenizer
- Chosen checkpoint(s): xlm-roberta-base (XLM-R)
- Arabic fertility evidence: XLM-R achieved an Arabic fertility of 1.667, compared with 2.143 for mBERT, 1.400 for CAMeLBERT, and 4.517 for DistilBERT.
- English fertility evidence: XLM-R achieved an English fertility of 1.433, providing strong English tokenization while maintaining good Arabic performance.
- p95 length evidence: XLM-R produced p95 sequence lengths of 21 tokens for Arabic and 23 tokens for English.
- Operational trade-off / rationale: XLM-R provides the best balanced bilingual performance for Bayan. CAMeLBERT performed slightly better on Arabic but was substantially less efficient on English, while DistilBERT performed poorly on Arabic. XLM-R also achieved a 0.000% Arabic UNK rate in the audit.

## arabic-model
- Incumbent: CAMeLBERT-mix
- Candidate: CAMeLBERT-DA
- All/Gulf/MSA evidence: both models achieved 0.9992 aggregate macro-F1, 1.0000 Gulf macro-F1, and 1.0000 MSA macro-F1.
- Verdict: retain CAMeLBERT-mix because the Gulf-slice performance tied; there is no measured benefit from switching models.
- Segmentation contract: LOCATION recall was 1.0000 before segmentation and 1.0000 after segmentation, for a +0.0000 delta. The baseline was already at the recall ceiling.

## search-min-score
- Threshold: 0.6651
- No-answer evidence: 20/20 no-answer queries correctly returned empty results.
- False-positive / false-negative trade-off: the selected threshold achieved 1.0000 answerable retention while preserving 20/20 no-answer correctness.

## quantisation-split
- Topic artefact: ONNX INT8 selected.
- NER artefact: no local Lab 3 NER optimisation artefact was available for the final optimisation comparison.
- Latency evidence: ONNX INT8 p50 = 3.70 ms and p99 = 6.08 ms, with 26.16x speed-up over the FP32 @512 baseline.
- Paired quality-tax evidence: macro-F1 = 1.0000 with +0.0000 quality tax and 1.0000 prediction agreement with FP32.
- Decision: use INT8 for the topic classifier; do not claim an NER quantisation decision without measured evidence.

## architecture
- Encoder/decoder rationale by task: encoder-based transformer models are used because Bayan focuses on classification, token classification, extractive QA, and embedding-based retrieval rather than free-form text generation.
- Multilingual vs Arabic-centric rationale: XLM-R was preferred where balanced Arabic/English handling was important, while CAMeLBERT was used for Arabic-focused classification after the tokenizer and model audits.
- Evidence used: tokenizer fertility and sequence-length audit, frozen-test classification metrics, NER/QA evaluation, retrieval metrics, behavioural evaluation, and measured CPU serving latency.
