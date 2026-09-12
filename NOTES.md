# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

## Lab 1 — Defect Safari

### Defect 1

- Class: PII (Personally Identifiable Information)
- Example: 0551234567 and 1023456789
- Why it matters: Phone numbers and national IDs contain sensitive personal information and should not be exposed to the NLP model.
- Decision: Clean — mask them using placeholders such as <PHONE> and <NATIONAL_ID>.

### Defect 2

- Class: HTML remnants
- Example: <br>
- Why it matters: HTML tags do not add useful meaning to the citizen feedback and can create unnecessary tokens.
- Decision: Clean — remove HTML remnants.

### Defect 3

- Class: Emoji
- Example: 😡
- Why it matters: Emojis can carry useful sentiment information, such as anger or frustration.
- Decision: Preserve.

### Defect 4

- Class: Repeated characters
- Example: لووووسمحت
- Why it matters: Repeated characters can increase token count and create inconsistent representations of the same word.
- Decision: Task-dependent — normalize excessive repetition while preserving useful emphasis where appropriate.

### Defect 5

- Class: Extra whitespace
- Example: "  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   "
- Why it matters: Leading, trailing, and repeated spaces create inconsistent text and unnecessary formatting differences.
- Decision: Clean — collapse repeated whitespace and strip leading/trailing spaces.

### Defect 6

- Class: Mixed scripts / code-switching
- Example: تم احتساب رسوم غير صحيحة على الفاتورة رقم BYN-2025-000012
- Why it matters: Arabic text may contain Latin-script identifiers or English content, so preprocessing must handle both scripts without destroying useful information.
- Decision: Preserve useful Latin identifiers and bilingual content.


## Lab 1 — Sentence Segmentation

- English sentences were split correctly at normal sentence boundaries.
- Abbreviations such as "Dr." were preserved and did not create an incorrect sentence break.
- Arabic sentences were separated correctly using punctuation.
- Mixed Arabic-English feedback was segmented into sensible sentence units.
- Arabic text may appear visually reordered in the terminal because of right-to-left rendering, but the underlying text remains correct.



## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.84% | Attention: 15.94%, FFN: 31.86% |
| CAMeLBERT | 109,081,344 | 21.48% | Attention: 25.99%, FFN: 51.95% |

Why is the embedding share different?  
mBERT has a much larger multilingual vocabulary, so its embedding table consumes a larger share of the model parameters; this is part of the multilingual vocabulary tax, while CAMeLBERT uses a more focused Arabic vocabulary.

### Attention diagnostics

- Scaled dot-product attention matched the PyTorch reference within the required 1e-6 tolerance.
- Multi-Head Attention preserved the expected input/output shape.
- The causal mask produced lower-triangular decoder-style attention with zero attention paid to future tokens.
- Mean PAD attention without mask: 0.046848.
- Mean PAD attention with mask: 0.000000.
- Most adjacency-looking head: 7 (score = 0.242445).
- Strongest [SEP]-sink head: 2 (mean [SEP] attention = 0.166924).
- Applying the correct attention mask eliminated PAD leakage.

## Lab 4 — Dialect audit

- Arabic slice: 7200 / 12000 rows
- Dialect distribution:
  - Gulf: 4800
  - MSA: 2400
- Implication: evaluating only on MSA can hide performance differences on dialectal/Gulf Arabic, so model choice should use slice-level evidence.
