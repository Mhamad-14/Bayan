"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Calculate subword pieces per whitespace-separated word."""

    total_pieces = 0
    total_words = 0

    for text in texts:
        words = text.split()

        if not words:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)

        total_pieces += len(token_ids)
        total_words += len(words)

    return total_pieces / total_words if total_words else 0.0


def main():
    """Audit tokenizer fertility, sequence lengths, and Arabic UNK rate."""

    # Load the Bayan dataset
    df = pd.read_csv(DATA)

    # Separate Arabic and English text
    ar_texts = [
        preprocess(str(text))
        for text in df.loc[df["lang"] == "ar", "text"].dropna()
    ]

    en_texts = [
        preprocess(str(text))
        for text in df.loc[df["lang"] == "en", "text"].dropna()
    ]

    print(f"Arabic examples: {len(ar_texts)}")
    print(f"English examples: {len(en_texts)}")

    results = []

    for checkpoint, name in CANDIDATES.items():
        print(f"\nLoading {name}...")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Fertility
        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)

        # Sequence lengths, including special tokens
        ar_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in ar_texts
        ]

        en_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in en_texts
        ]

        ar_p95 = float(np.percentile(ar_lengths, 95))
        en_p95 = float(np.percentile(en_lengths, 95))

        # Arabic unknown-token rate
        unk_id = tokenizer.unk_token_id
        total_ar_tokens = 0
        total_ar_unk = 0

        for text in ar_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            total_ar_tokens += len(ids)

            if unk_id is not None:
                total_ar_unk += ids.count(unk_id)

        ar_unk_rate = (
            total_ar_unk / total_ar_tokens
            if total_ar_tokens
            else 0.0
        )

        results.append(
            {
                "Tokenizer": name,
                "AR fertility": ar_fertility,
                "EN fertility": en_fertility,
                "AR p95 len": ar_p95,
                "EN p95 len": en_p95,
                "AR UNK rate": ar_unk_rate,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n=== TOKENIZER AUDIT RESULTS ===\n")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "AR fertility": "{:.3f}".format,
                "EN fertility": "{:.3f}".format,
                "AR p95 len": "{:.1f}".format,
                "EN p95 len": "{:.1f}".format,
                "AR UNK rate": "{:.3%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()