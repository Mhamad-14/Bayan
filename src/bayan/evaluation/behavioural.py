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

        for (_, template), group in invariance.groupby(
            ["lang", "template"], dropna=False
        ):
            predictions = group["prediction"].astype(str)
            counts = predictions.value_counts()
            n_unique_terms = int(group["term"].astype(str).nunique())

            # The supplied invariance rows contain perturbations but no
            # explicit reference row. Use the modal topic within each
            # language/template group as the reference prediction.
            unique_mode = (
                len(counts) == 1
                or (
                    len(counts) > 1
                    and int(counts.iloc[0]) > int(counts.iloc[1])
                )
            )

            scorable_group = n_unique_terms >= 2 and unique_mode
            reference_prediction = (
                str(counts.index[0]) if scorable_group else None
            )

            for _, row in group.iterrows():
                passed = (
                    None
                    if not scorable_group
                    else str(row["prediction"]) == reference_prediction
                )

                records.append(
                    {
                        "test_id": row["test_id"],
                        "test_type": "invariance",
                        "text": row["text"],
                        "prediction": row["prediction"],
                        "expected_relation": row["expected_relation"],
                        "reference_prediction": reference_prediction,
                        "passed": passed,
                        "scorable": scorable_group,
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
            "n_unique_texts": (
                int(subset["text"].dropna().nunique())
                if "text" in subset.columns
                else 0
            ),
            "pass_rate": float(scorable["passed"].mean()) if len(scorable) else None,
        }

    return {"summary": summary, "detail": detail}
