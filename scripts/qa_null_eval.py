"""Lab 3: deterministic no-answer QA evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from qa_smoke import DEFAULT_CHECKPOINT, predict


DATA_PATH = Path("data/models/bayan_qa.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--null-threshold", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--target", type=int, default=17)
    return parser.parse_args()


def load_unanswerable(path: Path, limit: int):
    obj = json.loads(path.read_text(encoding="utf-8"))
    examples = []

    for article in obj["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    examples.append(
                        {
                            "id": qa["id"],
                            "question": qa["question"],
                            "context": context,
                        }
                    )

    return examples[:limit], len(examples)


def main():
    args = parse_args()

    examples, total_available = load_unanswerable(
        DATA_PATH,
        args.limit,
    )

    print("=== Bayan QA No-Answer Evaluation ===")
    print("Dataset:", DATA_PATH)
    print("Total supplied unanswerable:", total_available)
    print("Evaluated:", len(examples))
    print("Checkpoint:", args.checkpoint)
    print("Null threshold:", args.null_threshold)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.checkpoint
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Device:", device)
    print()

    correct = 0

    for example in examples:
        predicted = predict(
            example["question"],
            example["context"],
            tokenizer,
            model,
            device,
            args.null_threshold,
        )

        passed = predicted is None
        correct += int(passed)

        print(
            example["id"],
            "|",
            "PASS" if passed else "FAIL",
            "| predicted=",
            repr(predicted),
        )

    print("\n=== QA No-Answer Summary ===")
    print(f"Null correct: {correct}/{len(examples)}")
    print(
        f"Course target >= {args.target}/{len(examples)}:",
        correct >= args.target,
    )

    if correct < args.target:
        raise SystemExit("QA no-answer target not met.")


if __name__ == "__main__":
    main()
