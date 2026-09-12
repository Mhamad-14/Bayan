"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

from bayan.models.data import build_topic_dataset


MODELS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

OUT = Path("artifacts/lab4_arabic_bakeoff.json")
BATCH = int(os.environ.get("LAB4_BATCH", "32"))


def _find_col(df, names):
    lower = {
        c.lower(): c
        for c in df.columns
    }

    for name in names:
        if name in lower:
            return lower[name]

    return None


def _slice_masks(df):
    dialect_col = _find_col(
        df,
        [
            "dialect_region",
            "dialect",
            "arabic_dialect",
            "variety",
            "arabic_variety",
        ],
    )

    region_col = _find_col(
        df,
        [
            "region",
            "geo_region",
            "area",
            "country",
            "country_code",
        ],
    )

    gulf_terms = (
        r"gulf|gcc|saudi|ksa|sa\b|uae|ae\b|"
        r"kuwait|kw\b|qatar|qa\b|bahrain|bh\b|"
        r"oman|om\b|خليج|سعود|كويت|قطر|بحرين|"
        r"عمان|امارات|إمارات"
    )

    msa_terms = (
        r"msa|modern standard|fusha|fus7a|فصح"
    )

    if region_col is not None:
        gulf = (
            df[region_col]
            .astype(str)
            .str.lower()
            .str.contains(
                gulf_terms,
                regex=True,
                na=False,
            )
        )

    elif dialect_col is not None:
        gulf = (
            df[dialect_col]
            .astype(str)
            .str.lower()
            .str.contains(
                gulf_terms,
                regex=True,
                na=False,
            )
        )

    else:
        gulf = pd.Series(
            False,
            index=df.index,
        )

    if dialect_col is not None:
        msa = (
            df[dialect_col]
            .astype(str)
            .str.lower()
            .str.contains(
                msa_terms,
                regex=True,
                na=False,
            )
        )

    else:
        msa = pd.Series(
            False,
            index=df.index,
        )

    return (
        gulf.to_numpy(),
        msa.to_numpy(),
        dialect_col,
        region_col,
    )


def _encode(texts, checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModel.from_pretrained(
        checkpoint
    )

    model.to(device)
    model.eval()

    vectors = []

    for start in range(
        0,
        len(texts),
        BATCH,
    ):
        batch = texts[
            start:start + BATCH
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.no_grad():
            hidden = model(
                **encoded
            ).last_hidden_state

            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
            )

            pooled = (
                (hidden * mask).sum(1)
                / mask.sum(1).clamp_min(1)
            )

        vectors.append(
            pooled.cpu().numpy()
        )

    return (
        np.concatenate(
            vectors,
            axis=0,
        ),
        tokenizer,
    )


def _fertility(
    tokenizer,
    texts,
    n=500,
):
    values = []

    for text in list(texts)[:n]:
        words = str(text).split()

        if not words:
            continue

        pieces = tokenizer.tokenize(
            str(text)
        )

        values.append(
            len(pieces) / len(words)
        )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def main():
    splits = build_topic_dataset()

    train = splits["train"].copy()
    test = splits["test"].copy()

    text_col = (
        "clean_text"
        if "clean_text" in train.columns
        else "text"
    )

    label_col = "topic"

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train[label_col].astype(str)
    )

    y_test = label_encoder.transform(
        test[label_col].astype(str)
    )

    (
        gulf_mask,
        msa_mask,
        dialect_col,
        region_col,
    ) = _slice_masks(test)

    print(
        "Dialect column:",
        dialect_col,
    )

    print(
        "Region column:",
        region_col,
    )

    print(
        "Gulf test rows:",
        int(gulf_mask.sum()),
    )

    print(
        "MSA test rows:",
        int(msa_mask.sum()),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    results = {}

    for name, checkpoint in MODELS.items():

        print(
            f"\n=== {name}: "
            f"{checkpoint} ==="
        )

        X_train, tokenizer = _encode(
            train[text_col]
            .astype(str)
            .tolist(),
            checkpoint,
            device,
        )

        X_test, _ = _encode(
            test[text_col]
            .astype(str)
            .tolist(),
            checkpoint,
            device,
        )

        classifier = LogisticRegression(
            max_iter=1000
        )

        classifier.fit(
            X_train,
            y_train,
        )

        predictions = classifier.predict(
            X_test
        )

        row = {
            "checkpoint": checkpoint,

            "macro_f1_all": float(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                )
            ),

            "gulf_rows": int(
                gulf_mask.sum()
            ),

            "msa_rows": int(
                msa_mask.sum()
            ),

            "macro_f1_gulf": None,
            "macro_f1_msa": None,

            "ar_fertility": _fertility(
                tokenizer,
                test[text_col]
                .astype(str)
                .tolist(),
            ),
        }

        if gulf_mask.sum() > 1:
            row["macro_f1_gulf"] = float(
                f1_score(
                    y_test[gulf_mask],
                    predictions[gulf_mask],
                    average="macro",
                )
            )

        if msa_mask.sum() > 1:
            row["macro_f1_msa"] = float(
                f1_score(
                    y_test[msa_mask],
                    predictions[msa_mask],
                    average="macro",
                )
            )

        results[name] = row

        print(
            json.dumps(
                row,
                indent=2,
            )
        )

        del X_train
        del X_test

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mix_gulf = (
        results[
            "CAMeLBERT-mix"
        ]["macro_f1_gulf"]
    )

    da_gulf = (
        results[
            "CAMeLBERT-DA"
        ]["macro_f1_gulf"]
    )

    if (
        mix_gulf is None
        or da_gulf is None
    ):
        gulf_delta = None

    else:
        gulf_delta = (
            da_gulf
            - mix_gulf
        )

    payload = {
        "method": (
            "frozen mean-pooled embeddings "
            "+ logistic-regression linear probe"
        ),

        "results": results,

        "gulf_delta_da_minus_mix":
            gulf_delta,
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: {OUT}"
    )

    print(
        "Gulf delta CAMeLBERT-DA "
        "- CAMeLBERT-mix:",
        gulf_delta,
    )


if __name__ == "__main__":
    main()
