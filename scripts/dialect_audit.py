"""Lab 4: audit the Arabic dialect/region mix and persist evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")
OUT_PATH = Path("artifacts/lab4_dialect_audit.json")


def _pick_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def main():
    df = pd.read_csv(DATA_PATH)

    dialect_col = _pick_column(
        df.columns,
        ["dialect_region", "dialect", "arabic_dialect", "variety", "arabic_variety"],
    )
    region_col = _pick_column(
        df.columns,
        ["region", "geo_region", "area", "country", "country_code"],
    )
    language_col = _pick_column(
        df.columns,
        ["language", "lang", "locale"],
    )

    arabic = df.copy()
    if language_col is not None:
        values = arabic[language_col].astype(str).str.lower()
        mask = values.str.contains(r"(?:^ar$|arab|عرب)", regex=True, na=False)
        if mask.any():
            arabic = arabic[mask].copy()

    result = {
        "rows_total": int(len(df)),
        "rows_arabic_slice": int(len(arabic)),
        "language_column": language_col,
        "dialect_column": dialect_col,
        "region_column": region_col,
        "dialect_distribution": {},
        "region_distribution": {},
    }

    if dialect_col is not None:
        result["dialect_distribution"] = {
            str(k): int(v)
            for k, v in arabic[dialect_col].fillna("<missing>").value_counts().items()
        }

    if region_col is not None:
        result["region_distribution"] = {
            str(k): int(v)
            for k, v in arabic[region_col].fillna("<missing>").value_counts().items()
        }

    print("=== Arabic slice ===")
    print(f"Rows: {len(arabic)} / {len(df)}")
    print("Language column:", language_col)
    print("Dialect column:", dialect_col)
    print("Region column:", region_col)

    print("\\n=== Dialect distribution ===")
    if result["dialect_distribution"]:
        for k, v in result["dialect_distribution"].items():
            print(f"{k}: {v}")
    else:
        print("No explicit dialect column found.")

    print("\\n=== Region distribution ===")
    if result["region_distribution"]:
        for k, v in result["region_distribution"].items():
            print(f"{k}: {v}")
    else:
        print("No explicit region column found.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
