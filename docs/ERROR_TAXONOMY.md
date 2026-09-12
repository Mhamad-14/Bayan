# Error Taxonomy

Use exactly one primary category for each manually reviewed error.

1. **Label ambiguity** — more than one topic label is reasonably defensible.
2. **Arabic orthographic variation** — spelling, hamza, taa marbuta, elongation, or related Arabic surface variation appears relevant.
3. **Dialect or code-switching** — dialect vocabulary, Arabizi, English/Arabic mixing, or regional wording appears relevant.
4. **Entity boundary or clitic alignment** — token/entity boundary or Arabic clitic segmentation appears relevant.
5. **Long-context truncation** — important evidence may fall outside the model context window.
6. **Retrieval relevance mismatch** — the retrieved/evaluated evidence is semantically related but does not satisfy the labelled relevance judgement.
7. **Preprocessing or serving skew** — evaluation input differs materially from the representation expected by the trained artefact.
8. **Annotation defect** — the supplied gold annotation appears inconsistent or incorrect.
9. **Other / taxonomy extension needed** — use only when none of the categories above fits; add a note explaining why.

## Manual review protocol

Lab 6 uses 120 sampled validation errors. Errors are displayed in pairs and are tagged manually by the reviewer.

The review file stores the original feedback text, gold label, predicted label, confidence, language, dialect, length bucket, primary error category, and an optional human note.
