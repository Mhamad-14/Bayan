"""Interactive Lab 6 manual review of 120 validation errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/eval/validation_predictions.csv"
RAW = ROOT / "data/raw/bayan_feedback.csv"
OUT = ROOT / "artifacts/lab6_manual_error_review.csv"

CATEGORIES = {
    1: "Label ambiguity",
    2: "Arabic orthographic variation",
    3: "Dialect or code-switching",
    4: "Entity boundary or clitic alignment",
    5: "Long-context truncation",
    6: "Retrieval relevance mismatch",
    7: "Preprocessing or serving skew",
    8: "Annotation defect",
    9: "Other / taxonomy extension needed",
}


def main():
    pred = pd.read_csv(PRED)
    raw = pd.read_csv(RAW)

    errors = pred[pred["y_true"] != pred["y_pred"]].copy()
    errors = errors.merge(raw[["feedback_id", "text"]], on="feedback_id", how="left")

    if len(errors) < 120:
        raise RuntimeError(
            f"Only {len(errors)} validation errors exist; cannot sample the required 120."
        )

    sample = errors.sample(n=120, random_state=42).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    reviewed_ids = (
        set(existing["feedback_id"].astype(str)) if not existing.empty else set()
    )
    rows = existing.to_dict("records")

    print("\nLAB 6 — MANUAL ERROR REVIEW")
    print("============================")
    print("Enter TWO category numbers per pair, e.g. 1 3")
    print("Type q to stop safely; rerun later to resume.\n")

    for number, label in CATEGORIES.items():
        print(f"{number}. {label}")

    for start in range(0, len(sample), 2):
        pair = sample.iloc[start:start + 2]
        remaining = [
            row for _, row in pair.iterrows()
            if str(row["feedback_id"]) not in reviewed_ids
        ]
        if not remaining:
            continue

        print("\n" + "=" * 78)
        print(f"PAIR {start // 2 + 1}/60 — reviewed {len(reviewed_ids)}/120")
        print("=" * 78)

        for j, row in enumerate(remaining, start=1):
            print(f"\n[{j}] {row['feedback_id']}")
            print(
                f"lang={row['lang']} | dialect={row['dialect_region']} | "
                f"length={row['length_bucket']}"
            )
            print(
                f"GOLD={row['y_true']} | PRED={row['y_pred']} | "
                f"confidence={row['confidence']}"
            )
            print("TEXT:")
            print(row["text"])

        while True:
            answer = input(
                f"\nCategory number{'s' if len(remaining) == 2 else ''}: "
            ).strip()

            if answer.lower() == "q":
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f"\nSaved progress: {len(reviewed_ids)}/120")
                return

            parts = answer.replace(",", " ").split()
            try:
                cats = [int(x) for x in parts]
            except ValueError:
                print("Use category numbers only, e.g. 1 3")
                continue

            if (
                len(cats) != len(remaining)
                or any(c not in CATEGORIES for c in cats)
            ):
                print(f"Enter exactly {len(remaining)} valid category number(s).")
                continue
            break

        for row, cat in zip(remaining, cats):
            note = ""
            if cat == 9:
                note = input("Short note for category 9: ").strip()

            record = row.to_dict()
            record["error_category_id"] = cat
            record["error_category"] = CATEGORIES[cat]
            record["review_note"] = note
            rows.append(record)
            reviewed_ids.add(str(row["feedback_id"]))

        pd.DataFrame(rows).to_csv(OUT, index=False)

    final = pd.DataFrame(rows)
    print("\nManual review complete ✅")
    print(f"Saved: {OUT}")
    print("\nCategory histogram:")
    print(final["error_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
