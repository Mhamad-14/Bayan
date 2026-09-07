"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    """Count model parameters by architectural component."""

    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        lower_name = name.lower()

        # LayerNorm parameters
        if "layernorm" in lower_name or "layer_norm" in lower_name:
            buckets["norms"] += count

        # Embedding tables
        elif "embeddings" in lower_name:
            buckets["embeddings"] += count

        # Self-attention projections and attention output
        elif "attention" in lower_name:
            buckets["attention"] += count

        # Transformer feed-forward network
        elif (
            "intermediate" in lower_name
            or (
                "encoder.layer." in lower_name
                and "output.dense" in lower_name
            )
        ):
            buckets["ffn"] += count

        # Pooler
        elif "pooler" in lower_name:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    total = sum(buckets.values())

    result = {
        "total": total,
        "buckets": buckets,
        "percentages": {
            key: (value / total) * 100
            for key, value in buckets.items()
        },
    }

    return result


if __name__ == "__main__":
    checkpoints = [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]

    for checkpoint in checkpoints:
        print("\n" + "=" * 70)
        print(checkpoint)
        print("=" * 70)

        result = audit(checkpoint)

        print(f"Total parameters: {result['total']:,}")

        for bucket, count in result["buckets"].items():
            percentage = result["percentages"][bucket]

            print(
                f"{bucket:12s}: "
                f"{count:>12,} "
                f"({percentage:6.2f}%)"
            )