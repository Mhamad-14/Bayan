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
| class | billing | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | digital_services | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | licensing | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | lighting | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | parks | 300 | 0.000 | 0.000 | 0.000–0.000 | False |
| class | roads | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | waste | 300 | 1.000 | 1.000 | 1.000–1.000 | False |
| class | water | 300 | 1.000 | 1.000 | 1.000–1.000 | False |

## Behavioural tests
| Test | Pass rate | Evidence |
|---|---:|---|
| invariance | 0.800 | 200 scorable rows; 10 unique instantiated texts; per-row agreement with the modal topic within each language/template group |
| directional | N/A | supplied relation is sentiment-directional; topic classifier cannot legitimately score sentiment direction |
| MFT | 0.960 | 100 labelled examples |

## Known limitations
The classifier does not meet the course invariance benchmark: the measured invariance pass rate is 0.8000 versus the approximate 0.95 target. English location substitutions such as Dammam and Riyadh caused some predictions to change from digital_services to lighting, while the Arabic invariance examples remained stable. The supplied directional behavioural tests concern sentiment, but this artefact predicts topics, so a sentiment-direction pass rate is not reported. Slice results should also be interpreted cautiously where sample sizes are small.

## Contact / owner
Bayan course project
