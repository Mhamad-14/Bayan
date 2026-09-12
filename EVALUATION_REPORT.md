# Bayan Evaluation Report

## Manager headline

Aggregate validation macro-F1 is 0.833 (95% bootstrap CI 0.829–0.837). The weakest adequately sized reported slice is lang=ar at macro-F1 0.600; small slices remain explicitly flagged.

## Aggregate evaluation

- Validation rows: 2400
- Macro-F1: 0.8333
- 95% bootstrap CI for macro-F1: [0.8290, 0.8373]
- Accuracy: 0.8750
- 95% bootstrap CI for accuracy: [0.8617, 0.8888]

### Paired bootstrap comparison

Comparison: normal correctness versus confidence-gated correctness
(`confidence >= 0.60`) on the same validation examples.

- paired delta: +0.1000
- 95% paired bootstrap CI: [+0.0883, +0.1117]
- verdict: signal

## Sliced evaluation

| slice_name | slice_value | n | macro_f1 | accuracy | accuracy_ci | small_slice |
|---|---|---|---|---|---|---|
| all | all | 2400 | 0.833 | 0.875 | 0.862–0.889 | False |
| lang | ar | 1200 | 0.600 | 0.750 | 0.726–0.776 | False |
| lang | en | 1200 | 1.000 | 1.000 | 1.000–1.000 | False |
| dialect_region | MSA | 1200 | 0.600 | 0.750 | 0.726–0.776 | False |
| dialect_region | N/A | 1200 | 1.000 | 1.000 | 1.000–1.000 | False |
| length_bucket | medium | 1646 | 0.846 | 0.889 | 0.874–0.905 | False |
| length_bucket | short | 754 | 0.714 | 0.844 | 0.818–0.869 | False |

Small slices are flagged rather than treated as precise estimates.

## Behavioural evaluation

| Test | Pass rate | Evidence |
|---|---:|---|
| invariance | 0.500 | 200 scorable tests |
| directional | N/A | supplied relation is sentiment-directional; topic classifier cannot legitimately score sentiment direction |
| MFT | 0.960 | 100 labelled examples |

Course benchmark references:
- invariance target: approximately 0.95
- MFT target: approximately 0.90

## Manual error taxonomy

Human-reviewed validation errors: **120**

| category | count |
|---|---|
| Other / taxonomy extension needed | 97 |
| Arabic orthographic variation | 13 |
| Preprocessing or serving skew | 7 |
| Label ambiguity | 2 |
| Dialect or code-switching | 1 |

## Top 3 prioritised fixes

| category | count | prioritised_fix | predicted_macro_f1_delta |
|---|---|---|---|
| Other / taxonomy extension needed | 97 | Review the new category and define a targeted remediation. | +0.0711 |
| Arabic orthographic variation | 13 | Expand Arabic normalization and orthographic augmentation. | +0.0116 |
| Preprocessing or serving skew | 7 | Enforce one shared preprocessing contract across train/eval/serve. | +0.0064 |

The metric deltas above are scenario estimates obtained by correcting the
manually reviewed examples belonging to each category while leaving all
other validation predictions unchanged.

## Limitations

- Behavioural directional templates specify sentiment behaviour, while the
  available Lab 3 artefact is a topic classifier. A sentiment-direction
  pass rate is therefore not fabricated.
- Slice estimates marked as small have high uncertainty.
- The error taxonomy represents a manually sampled set of 120 errors.
- Retrieval limitations are documented separately in BENCHMARKS.md.

## Model cards

Three model-card evidence files are generated under `model_cards/`.
Their **Known limitations** sections must be completed manually before submission.
