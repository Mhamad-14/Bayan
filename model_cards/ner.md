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
Lab 6:
The available NER evidence is strong on the current labelled evaluation set, but Lab 6 does not include a dedicated NER behavioural suite. Performance may therefore be less certain on unseen dialectal wording, unusual clitic segmentation, and entity-boundary patterns not represented in the frozen evaluation data.

## Contact / owner
Bayan course project
