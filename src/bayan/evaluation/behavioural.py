"""Lab 6: behavioural evaluation."""

from __future__ import annotations

import pandas as pd


def run_behavioural_suite(
    templates,
    *,
    predict_fn,
    mft_expected=None,
    mft_predicted=None,
):
    """Run invariance, directional coverage, and MFT checks."""
    df = pd.DataFrame(templates).copy()

    required = {
        "test_id",
        "test_type",
        "lang",
        "template",
        "term",
        "expected_relation",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing behavioural columns: {sorted(missing)}")

    records = []

    invariance = df[df["test_type"].astype(str).str.lower() == "invariance"].copy()
    if not invariance.empty:
        invariance["text"] = [
            str(t).replace("{term}", str(term))
            for t, term in zip(invariance["template"], invariance["term"])
        ]
        invariance["prediction"] = predict_fn(invariance["text"].tolist())

        for (_, template), group in invariance.groupby(["lang", "template"], dropna=False):
            preds = group["prediction"].astype(str).tolist()
            passed = None if len(preds) < 2 else (len(set(preds)) == 1)

            for _, row in group.iterrows():
                records.append(
                    {
                        "test_id": row["test_id"],
                        "test_type": "invariance",
                        "text": row["text"],
                        "prediction": row["prediction"],
                        "expected_relation": row["expected_relation"],
                        "passed": passed,
                        "scorable": passed is not None,
                    }
                )

    directional = df[df["test_type"].astype(str).str.lower() == "directional"].copy()
    for _, row in directional.iterrows():
        text = str(row["template"]).replace("{term}", str(row["term"]))
        records.append(
            {
                "test_id": row["test_id"],
                "test_type": "directional",
                "text": text,
                "prediction": None,
                "expected_relation": row["expected_relation"],
                "passed": None,
                "scorable": False,
            }
        )

    if mft_expected is not None and mft_predicted is not None:
        expected = list(mft_expected)
        predicted = list(mft_predicted)
        if len(expected) != len(predicted):
            raise ValueError("MFT arrays must have equal length")

        for i, (gold, pred) in enumerate(zip(expected, predicted), start=1):
            records.append(
                {
                    "test_id": f"MFT-{i:03d}",
                    "test_type": "mft",
                    "text": None,
                    "prediction": str(pred),
                    "expected_relation": str(gold),
                    "passed": str(pred) == str(gold),
                    "scorable": True,
                }
            )

    detail = pd.DataFrame(records)
    summary = {}

    for test_type in ["invariance", "directional", "mft"]:
        subset = detail[detail["test_type"] == test_type]
        scorable = subset[subset["scorable"] == True]  # noqa: E712
        summary[test_type] = {
            "n_total": int(len(subset)),
            "n_scorable": int(len(scorable)),
            "pass_rate": float(scorable["passed"].mean()) if len(scorable) else None,
        }

    return {"summary": summary, "detail": detail}
