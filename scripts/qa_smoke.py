"""Lab 3B: run the supplied 12-question extractive-QA smoke set."""

import argparse
import json
import re
import string
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


DEFAULT_CHECKPOINT = "deepset/xlm-roberta-base-squad2"
SMOKE_PATH = Path("data/eval/qa_smoke_set.json")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Fine-tuned extractive-QA checkpoint. "
            "The course README refers to a provided checkpoint but "
            "does not currently name or include one."
        ),
    )

    parser.add_argument(
        "--null-threshold",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def load_smoke_questions(path):
    with path.open(encoding="utf-8") as file:
        obj = json.load(file)

    examples = []

    for article in obj["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                expected = None

                if not qa.get("is_impossible", False):
                    expected = qa["answers"][0]["text"]

                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "expected": expected,
                        "is_impossible": qa.get(
                            "is_impossible",
                            False,
                        ),
                    }
                )

    return examples


def normalize_answer(text):
    """SQuAD-style normalization for exact-match scoring."""

    if text is None:
        return None

    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(text.split())

    return text


def predict(
    question,
    context,
    tokenizer,
    model,
    device,
    null_threshold,
):
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=384,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    # Keep offsets only for CONTEXT tokens.
    sequence_ids = encoded.sequence_ids(0)

    context_offsets = []

    for sequence_id, offset in zip(
        sequence_ids,
        offsets,
    ):
        if sequence_id == 1:
            context_offsets.append(
                (int(offset[0]), int(offset[1]))
            )
        else:
            context_offsets.append(None)

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = (
        outputs.start_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    # For RoBERTa/XLM-R QA models the first token is used
    # as the null/no-answer candidate.
    null_score = float(
        start_logits[0] + end_logits[0]
    )

    result = best_span(
        start_logits,
        end_logits,
        context_offsets,
        null_score=null_score,
        null_threshold=null_threshold,
        max_answer_len=30,
        top_k=20,
    )

    if result["answer"] is None:
        return None

    char_start, char_end = result["answer"]

    return context[char_start:char_end]


def main():
    args = parse_args()

    print("=== Bayan QA Smoke Test ===")
    print(f"Checkpoint: {args.checkpoint}")

    examples = load_smoke_questions(
        SMOKE_PATH
    )

    answerable_total = sum(
        not example["is_impossible"]
        for example in examples
    )

    null_total = sum(
        example["is_impossible"]
        for example in examples
    )

    print(f"Total questions: {len(examples)}")
    print(f"Answerable: {answerable_total}")
    print(f"Unanswerable: {null_total}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(args.checkpoint)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Device: {device}\n")

    answerable_correct = 0
    null_correct = 0

    for example in examples:
        predicted = predict(
            example["question"],
            example["context"],
            tokenizer,
            model,
            device,
            args.null_threshold,
        )

        expected = example["expected"]

        if example["is_impossible"]:
            correct = predicted is None

            if correct:
                null_correct += 1

        else:
            correct = (
                predicted is not None
                and normalize_answer(predicted)
                == normalize_answer(expected)
            )

            if correct:
                answerable_correct += 1

        print(
            f"{example['id']} | "
            f"expected={expected!r} | "
            f"predicted={predicted!r} | "
            f"{'PASS' if correct else 'FAIL'}"
        )

    print("\n=== QA Smoke Summary ===")

    print(
        f"Answerable correct: "
        f"{answerable_correct}/{answerable_total}"
    )

    print(
        f"Null correct: "
        f"{null_correct}/{null_total}"
    )

    if (
        answerable_correct == answerable_total
        and null_correct == null_total
    ):
        print("QA smoke set passed ✅")
    else:
        raise SystemExit(
            "QA smoke set did not fully pass."
        )


if __name__ == "__main__":
    main()