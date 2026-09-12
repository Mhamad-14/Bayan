"""Lab 6: sliced evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

from .bootstrap import bootstrap_ci


def sliced_report(
    df,
    *,
    y_true="y_true",
    y_pred="y_pred",
    slice_columns=("lang", "dialect_region", "length_bucket"),
    min_size=30,
):
    """Return aggregate, metadata slices, and class slices."""
    frame = pd.DataFrame(df).copy()

    if y_true not in frame or y_pred not in frame:
        raise ValueError("data must contain y_true and y_pred columns")

    rows = []

    def add_row(kind, name, value, subset):
        if len(subset) == 0:
            return

        correct = (
            subset[y_true].astype(str).to_numpy()
            == subset[y_pred].astype(str).to_numpy()
        ).astype(float)

        acc, lo, hi = bootstrap_ci(correct, n_boot=1000, seed=42)

        macro_f1 = float(
            f1_score(
                subset[y_true].astype(str),
                subset[y_pred].astype(str),
                average="macro",
                zero_division=0,
            )
        )

        rows.append(
            {
                "slice_type": kind,
                "slice_name": name,
                "slice_value": str(value),
                "n": int(len(subset)),
                "macro_f1": macro_f1,
                "accuracy": acc,
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                "small_slice": bool(len(subset) < min_size),
            }
        )

    add_row("aggregate", "all", "all", frame)

    for column in slice_columns:
        if column not in frame:
            continue
        values = frame[column].fillna("N/A").astype(str)
        for value in sorted(values.unique()):
            add_row("metadata", column, value, frame.loc[values == value])

    for label in sorted(frame[y_true].astype(str).unique()):
        subset = frame.loc[frame[y_true].astype(str) == label]
        add_row("class", "class", label, subset)

    return pd.DataFrame(rows)
