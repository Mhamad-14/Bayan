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
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:



