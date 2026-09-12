"""Generate the Lab 6 evaluation report and model-card evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_diff
from bayan.evaluation.slices import sliced_report
from bayan.evaluation.behavioural import run_behavioural_suite


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/eval/validation_predictions.csv"
BEHAVIOUR = ROOT / "data/eval/behavioural_templates.csv"
MANUAL = ROOT / "artifacts/lab6_manual_error_review.csv"
MODEL = ROOT / "artifacts/topic_classifier"
METRICS = MODEL / "metrics.json"
BENCH = ROOT / "BENCHMARKS.md"
REPORT = ROOT / "EVALUATION_REPORT.md"
MODEL_CARDS = ROOT / "model_cards"


def macro_f1_bootstrap(df, n_boot=1000, seed=42):
    point = float(
        f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0)
    )
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[idx]
        scores.append(
            f1_score(
                sample["y_true"],
                sample["y_pred"],
                average="macro",
                zero_division=0,
            )
        )

    lo, hi = np.quantile(scores, [0.025, 0.975])
    return point, float(lo), float(hi)


def load_predictor():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    label2id = metrics.get("label2id", {})
    id2label = {int(v): str(k) for k, v in label2id.items()}

    def predict(texts):
        results = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                ids = model(**encoded).logits.argmax(dim=-1).tolist()

            for i in ids:
                results.append(id2label.get(int(i), model.config.id2label.get(int(i), str(i))))
        return results

    return predict


def markdown_table(df, columns):
    if df.empty:
        return "No data."
    view = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def category_fix(category):
    mapping = {
        "Label ambiguity": "Clarify class definitions and adjudicate ambiguous labels.",
        "Arabic orthographic variation": "Expand Arabic normalization and orthographic augmentation.",
        "Dialect or code-switching": "Add dialect/code-switching examples and targeted augmentation.",
        "Entity boundary or clitic alignment": "Improve token/clitic alignment and boundary-aware training.",
        "Long-context truncation": "Increase effective context coverage or chunk long inputs.",
        "Retrieval relevance mismatch": "Improve retrieval supervision and hard-negative selection.",
        "Preprocessing or serving skew": "Enforce one shared preprocessing contract across train/eval/serve.",
        "Annotation defect": "Correct inconsistent labels and add annotation QA.",
        "Other / taxonomy extension needed": "Review the new category and define a targeted remediation.",
    }
    return mapping.get(category, "Investigate this error family and add targeted evaluation.")


def write_model_card(path, **values):
    template = (ROOT / "templates/model_card.md.j2").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{ " + key + " }}", str(value))
    path.write_text(template, encoding="utf-8")


def main():
    pred = pd.read_csv(PRED)

    macro, macro_lo, macro_hi = macro_f1_bootstrap(pred)

    correct = (
        pred["y_true"].astype(str) == pred["y_pred"].astype(str)
    ).astype(float)

    accuracy, acc_lo, acc_hi = bootstrap_ci(correct, n_boot=2000, seed=42)

    gated = (
        correct.astype(bool) & (pred["confidence"] >= 0.60)
    ).astype(float)

    delta, delta_lo, delta_hi = paired_bootstrap_diff(
        correct, gated, n_boot=2000, seed=42
    )

    slices = sliced_report(pred)
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    slices.to_csv(ROOT / "artifacts/lab6_slices.csv", index=False)

    behavioural_templates = pd.read_csv(BEHAVIOUR)
    predictor = load_predictor()

    mft = pred.sample(n=min(100, len(pred)), random_state=42)
    behavioural = run_behavioural_suite(
        behavioural_templates,
        predict_fn=predictor,
        mft_expected=mft["y_true"].tolist(),
        mft_predicted=mft["y_pred"].tolist(),
    )
    behavioural["detail"].to_csv(
        ROOT / "artifacts/lab6_behavioural.csv", index=False
    )

    summary = behavioural["summary"]
    invariance_rate = summary["invariance"]["pass_rate"]
    mft_rate = summary["mft"]["pass_rate"]

    if not MANUAL.exists():
        raise SystemExit(
            "\nManual review is not complete.\n"
            "Run:\n  python scripts/manual_error_review.py\n"
            "Then rerun this report generator."
        )

    review = pd.read_csv(MANUAL)
    if len(review) != 120:
        raise SystemExit(
            f"\nManual review contains {len(review)}/120 rows.\n"
            "Resume:\n  python scripts/manual_error_review.py"
        )

    histogram = (
        review["error_category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    top3 = histogram.head(3).copy()

    deltas = []
    for category in top3["category"]:
        corrected = pred.copy()
        ids = set(
            review.loc[
                review["error_category"] == category, "feedback_id"
            ].astype(str)
        )
        mask = corrected["feedback_id"].astype(str).isin(ids)
        corrected.loc[mask, "y_pred"] = corrected.loc[mask, "y_true"]
        fixed_macro = float(
            f1_score(
                corrected["y_true"],
                corrected["y_pred"],
                average="macro",
                zero_division=0,
            )
        )
        deltas.append(fixed_macro - macro)

    top3["predicted_macro_f1_delta"] = deltas
    top3["prioritised_fix"] = [category_fix(x) for x in top3["category"]]

    metadata_slices = slices[
        (slices["slice_type"] == "metadata") & (~slices["small_slice"])
    ]

    if len(metadata_slices):
        worst = metadata_slices.sort_values("macro_f1").iloc[0]
        headline = (
            f"Aggregate validation macro-F1 is {macro:.3f} "
            f"(95% bootstrap CI {macro_lo:.3f}–{macro_hi:.3f}). "
            f"The weakest adequately sized reported slice is "
            f"{worst['slice_name']}={worst['slice_value']} at macro-F1 "
            f"{worst['macro_f1']:.3f}; small slices remain explicitly flagged."
        )
    else:
        headline = (
            f"Aggregate validation macro-F1 is {macro:.3f} "
            f"(95% bootstrap CI {macro_lo:.3f}–{macro_hi:.3f}). "
            "All metadata slices are small enough to warrant caution."
        )

    slice_display = slices.copy()
    for c in ["macro_f1", "accuracy"]:
        slice_display[c] = slice_display[c].map(lambda x: f"{x:.3f}")
    slice_display["accuracy_ci"] = (
        slice_display["accuracy_ci_low"].map(lambda x: f"{x:.3f}")
        + "–"
        + slice_display["accuracy_ci_high"].map(lambda x: f"{x:.3f}")
    )

    selected_slices = slice_display[slice_display["slice_type"] != "class"][
        [
            "slice_name",
            "slice_value",
            "n",
            "macro_f1",
            "accuracy",
            "accuracy_ci",
            "small_slice",
        ]
    ]

    behavioural_table = (
        "| Test | Pass rate | Evidence |\n"
        "|---|---:|---|\n"
        f"| invariance | {'N/A' if invariance_rate is None else f'{invariance_rate:.3f}'} | "
        f"{summary['invariance']['n_scorable']} scorable tests |\n"
        "| directional | N/A | supplied relation is sentiment-directional; "
        "topic classifier cannot legitimately score sentiment direction |\n"
        f"| MFT | {'N/A' if mft_rate is None else f'{mft_rate:.3f}'} | "
        f"{summary['mft']['n_scorable']} labelled examples |"
    )

    histogram_md = markdown_table(histogram, ["category", "count"])

    fixes_view = top3.copy()
    fixes_view["predicted_macro_f1_delta"] = fixes_view[
        "predicted_macro_f1_delta"
    ].map(lambda x: f"{x:+.4f}")

    fixes_md = markdown_table(
        fixes_view,
        ["category", "count", "prioritised_fix", "predicted_macro_f1_delta"],
    )

    report = f"""# Bayan Evaluation Report

## Manager headline

{headline}

## Aggregate evaluation

- Validation rows: {len(pred)}
- Macro-F1: {macro:.4f}
- 95% bootstrap CI for macro-F1: [{macro_lo:.4f}, {macro_hi:.4f}]
- Accuracy: {accuracy:.4f}
- 95% bootstrap CI for accuracy: [{acc_lo:.4f}, {acc_hi:.4f}]

### Paired bootstrap comparison

Comparison: normal correctness versus confidence-gated correctness
(`confidence >= 0.60`) on the same validation examples.

- paired delta: {delta:+.4f}
- 95% paired bootstrap CI: [{delta_lo:+.4f}, {delta_hi:+.4f}]
- verdict: {"signal" if not (delta_lo <= 0 <= delta_hi) else "difference compatible with sampling noise"}

## Sliced evaluation

{markdown_table(selected_slices, list(selected_slices.columns))}

Small slices are flagged rather than treated as precise estimates.

## Behavioural evaluation

{behavioural_table}

Course benchmark references:
- invariance target: approximately 0.95
- MFT target: approximately 0.90

## Manual error taxonomy

Human-reviewed validation errors: **120**

{histogram_md}

## Top 3 prioritised fixes

{fixes_md}

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
"""

    REPORT.write_text(report, encoding="utf-8")

    bench = BENCH.read_text(encoding="utf-8")
    start = bench.find("## Lab 6 — Evaluation")
    end = bench.find("## Lab 7 — Optimisation ladder")

    lab6 = f"""## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [95% CI] | Accuracy [95% CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | {macro:.4f} [{macro_lo:.4f}, {macro_hi:.4f}] | {accuracy:.4f} [{acc_lo:.4f}, {acc_hi:.4f}] | {'N/A' if invariance_rate is None else f'{invariance_rate:.4f}'} | {'N/A' if mft_rate is None else f'{mft_rate:.4f}'} |

- paired comparison: normal vs confidence-gated correctness delta {delta:+.4f}, 95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}]
- error taxonomy top categories: {", ".join(top3["category"].tolist())}
- top-3 prioritised fixes: {", ".join(top3["prioritised_fix"].tolist())}

"""

    if start >= 0 and end > start:
        bench = bench[:start] + lab6 + bench[end:]
    else:
        bench += "\n\n" + lab6

    BENCH.write_text(bench, encoding="utf-8")

    MODEL_CARDS.mkdir(parents=True, exist_ok=True)

    classifier_metrics = (
        f"| Metric | Result |\n"
        f"|---|---:|\n"
        f"| validation macro-F1 | {macro:.4f} |\n"
        f"| validation accuracy | {accuracy:.4f} |"
    )
    slices_table = markdown_table(selected_slices.head(12), list(selected_slices.columns))

    write_model_card(
        MODEL_CARDS / "topic_classifier.md",
        model_name="Bayan Topic Classifier",
        intended_use="Classify bilingual citizen feedback into Bayan service topics.",
        checkpoint="artifacts/topic_classifier",
        preproc_version="Bayan shared preprocessing",
        data_version="Lab 6 validation snapshot",
        metrics_table=classifier_metrics,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
    )

    write_model_card(
        MODEL_CARDS / "ner.md",
        model_name="Bayan NER",
        intended_use="Extract entities from bilingual Bayan feedback.",
        checkpoint="Lab 3 NER artefact",
        preproc_version="Bayan Arabic preprocessing",
        data_version="Lab 3 frozen evaluation snapshot",
        metrics_table=(
            "| Metric | Result |\n|---|---:|\n"
            "| entity-F1 | 1.0000 |\n| LOCATION recall | 1.0000 |"
        ),
        slices_table="Arabic segmentation evidence is recorded in BENCHMARKS.md.",
        behavioural_table="No separate NER behavioural suite was supplied for Lab 6.",
    )

    write_model_card(
        MODEL_CARDS / "semantic_search.md",
        model_name="Bayan Semantic Search",
        intended_use="Retrieve and rerank bilingual historical Bayan cases.",
        checkpoint=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "
            "+ multilingual cross-encoder"
        ),
        preproc_version="bayan_search_v1",
        data_version="20k Bayan historical case corpus",
        metrics_table=(
            "| Metric | Result |\n|---|---:|\n"
            "| bi-encoder recall@10 | 0.0692 |\n"
            "| bi-encoder MRR@10 | 0.0175 |\n"
            "| reranked recall@10 | 0.0077 |\n"
            "| reranked MRR@10 | 0.0026 |\n"
            "| no-answer correctness | 20/20 |"
        ),
        slices_table=(
            "| Slice | recall@10 |\n|---|---:|\n"
            "| same-language | 0.0077 |\n"
            "| cross-language | 0.0000 |"
        ),
        behavioural_table="Retrieval-labelled evaluation is documented in BENCHMARKS.md.",
    )

    print("Lab 6 report generated ✅")
    print(f"Aggregate macro-F1: {macro:.4f}")
    print("Invariance:", "N/A" if invariance_rate is None else f"{invariance_rate:.4f}")
    print("MFT:", "N/A" if mft_rate is None else f"{mft_rate:.4f}")
    print("Manual errors:", len(review))
    print("Model cards: 3")


if __name__ == "__main__":
    main()
