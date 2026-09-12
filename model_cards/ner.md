# Model Card — Bayan NER

## Intended use
Extract entities from bilingual Bayan feedback.

## Artefact / data versions
- Model/checkpoint: Lab 3 NER artefact
- Preprocessing version: Bayan Arabic preprocessing
- Data version/snapshot: Lab 3 frozen evaluation snapshot

## Metrics
| Metric | Result |
|---|---:|
| entity-F1 | 1.0000 |
| LOCATION recall | 1.0000 |

## Slice metrics
Arabic segmentation evidence is recorded in BENCHMARKS.md.

## Behavioural tests
No separate NER behavioural suite was supplied for Lab 6.

## Known limitations
The reported NER evidence comes from the supplied Lab 3 frozen evaluation snapshot. Lab 6 did not provide a separate behavioural robustness suite for NER, so invariance and other perturbation-based robustness were not measured for this artefact. Performance outside the supplied evaluation distribution and entity schema has not been established.

## Contact / owner
Bayan course project
