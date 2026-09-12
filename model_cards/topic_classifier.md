# Model Card — Bayan Topic Classifier

## Intended use
Classify bilingual citizen feedback into Bayan service topics.

## Artefact / data versions
- Model/checkpoint: artifacts/topic_classifier
- Preprocessing version: Bayan shared preprocessing
- Data version/snapshot: Lab 6 validation snapshot

## Metrics
| Metric | Result |
|---|---:|
| validation macro-F1 | 0.8333 |
| validation accuracy | 0.8750 |

## Slice metrics
| slice_name | slice_value | n | macro_f1 | accuracy | accuracy_ci | small_slice |
|---|---|---|---|---|---|---|
| all | all | 2400 | 0.833 | 0.875 | 0.862–0.889 | False |
| lang | ar | 1200 | 0.600 | 0.750 | 0.726–0.776 | False |
| lang | en | 1200 | 1.000 | 1.000 | 1.000–1.000 | False |
| dialect_region | MSA | 1200 | 0.600 | 0.750 | 0.726–0.776 | False |
| dialect_region | N/A | 1200 | 1.000 | 1.000 | 1.000–1.000 | False |
| length_bucket | medium | 1646 | 0.846 | 0.889 | 0.874–0.905 | False |
| length_bucket | short | 754 | 0.714 | 0.844 | 0.818–0.869 | False |

## Behavioural tests
| Test | Pass rate | Evidence |
|---|---:|---|
| invariance | 0.500 | 200 scorable tests |
| directional | N/A | supplied relation is sentiment-directional; topic classifier cannot legitimately score sentiment direction |
| MFT | 0.960 | 100 labelled examples |

## Known limitations
Lab 6:
The classifier shows weaker behavioural robustness than its aggregate score suggests: invariance pass rate is 0.50, below the course benchmark of approximately 0.95. Manual error review also identified recurring topic-boundary confusion, especially between parks and roads, so strong aggregate performance should not be interpreted as uniform reliability across examples.

## Contact / owner
Bayan course project
