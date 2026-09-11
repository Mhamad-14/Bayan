"""Lab 3B: fine-tune token classification with correct BIO alignment."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from bayan.models.ner import align_labels


CHECKPOINT = "xlm-roberta-base"
DATA_PATH = Path("data/models/bayan_ner.conll")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)

    return parser.parse_args()


def read_conll(path):
    """Read token<TAB>label CoNLL data."""

    examples = []
    tokens = []
    labels = []

    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            if not line.strip():
                if tokens:
                    examples.append((tokens, labels))
                    tokens = []
                    labels = []
                continue

            token, label = line.rsplit("\t", 1)
            tokens.append(token)
            labels.append(label)

    if tokens:
        examples.append((tokens, labels))

    return examples


def split_examples(examples, seed=42):
    """Create deterministic 80/10/10 train/validation/test splits."""

    examples = list(examples)

    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)

    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    return (
        examples[:train_end],
        examples[train_end:val_end],
        examples[val_end:],
    )


class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id, max_length):
        self.items = []

        for words, labels in examples:
            encoded = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )

            word_ids = encoded.word_ids()

            word_labels = [
                label2id[label]
                for label in labels
            ]

            encoded["labels"] = align_labels(
                word_ids,
                word_labels,
            )

            self.items.append(encoded)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def evaluate(model, dataloader, device, id2label):
    model.eval()

    true_sequences = []
    predicted_sequences = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1).cpu()

            for prediction_row, label_row in zip(
                predictions,
                labels,
            ):
                gold = []
                predicted = []

                for pred_id, label_id in zip(
                    prediction_row.tolist(),
                    label_row.tolist(),
                ):
                    if label_id == -100:
                        continue

                    gold.append(id2label[label_id])
                    predicted.append(id2label[pred_id])

                true_sequences.append(gold)
                predicted_sequences.append(predicted)

    score = f1_score(
        true_sequences,
        predicted_sequences,
    )

    return score, true_sequences, predicted_sequences


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)

    print("=== Bayan NER Training ===")
    print(f"Checkpoint: {CHECKPOINT}")

    examples = read_conll(DATA_PATH)

    train_examples, val_examples, test_examples = split_examples(
        examples
    )

    print("\n=== Dataset sizes ===")
    print(f"Total:      {len(examples)}")
    print(f"Train:      {len(train_examples)}")
    print(f"Validation: {len(val_examples)}")
    print(f"Test:       {len(test_examples)}")

    label_names = sorted(
        {
            label
            for _, labels in examples
            for label in labels
        }
    )

    if "O" in label_names:
        label_names.remove("O")
        label_names.insert(0, "O")

    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("\nLabels:")
    print(label2id)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    train_dataset = NERDataset(
        train_examples,
        tokenizer,
        label2id,
        args.max_length,
    )

    val_dataset = NERDataset(
        val_examples,
        tokenizer,
        label2id,
        args.max_length,
    )

    test_dataset = NERDataset(
        test_examples,
        tokenizer,
        label2id,
        args.max_length,
    )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print(f"\nDevice: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_val_f1 = -1.0
    best_state = None

    start_time = time.perf_counter()

    print("\n=== Training ===")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()

            if step % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"| step {step}/{len(train_loader)} "
                    f"| loss {loss.item():.4f}"
                )

        average_loss = total_loss / len(train_loader)

        val_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
            id2label,
        )

        print(f"\nEpoch {epoch + 1} complete")
        print(f"Average loss: {average_loss:.4f}")
        print(f"Validation entity-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    training_time = time.perf_counter() - start_time

    model.load_state_dict(best_state)
    model.to(device)

    print("\n=== Frozen Test Evaluation ===")

    test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        id2label,
    )

    print(f"Best validation entity-F1: {best_val_f1:.4f}")
    print(f"Frozen test entity-F1: {test_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\nEntity-level classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
        )
    )

    print(f"\nSaving NER artefact to: {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "checkpoint": CHECKPOINT,
        "validation_entity_f1": best_val_f1,
        "frozen_test_entity_f1": test_f1,
        "training_time_seconds": training_time,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "label2id": label2id,
    }

    with (
        output_dir / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("NER artefact saved successfully ✅")


if __name__ == "__main__":
    main()