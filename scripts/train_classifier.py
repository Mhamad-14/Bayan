"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    return parser.parse_args()


class TopicDataset(Dataset):
    """Tokenized Bayan topic-classification dataset."""

    def __init__(
        self,
        dataframe,
        tokenizer,
        label2id,
        max_length,
    ):
        texts = dataframe["clean_text"].tolist()

        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.labels = torch.tensor(
            [
                label2id[label]
                for label in dataframe["topic"]
            ],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


def evaluate(model, dataloader, device):
    """Evaluate macro-F1 without updating the model."""

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            labels = batch["labels"]

            outputs = model(**batch)

            preds = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            predictions.extend(
                preds.cpu().tolist()
            )

            references.extend(
                labels.cpu().tolist()
            )

    macro_f1 = f1_score(
        references,
        predictions,
        average="macro",
    )

    return macro_f1, references, predictions


def choose_device():
    """Choose CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(42)

    print("=== Bayan Topic Classifier ===")
    print(f"Checkpoint: {CHECKPOINT}")

    # Load leakage-safe grouped dataset
    dataset = build_topic_dataset()

    train_df = dataset["train"]
    val_df = dataset["validation"]
    test_df = dataset["test"]

    print("\n=== Dataset sizes ===")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    # Label mapping
    labels = sorted(
        train_df["topic"].unique().tolist()
    )

    label2id = {
        label: index
        for index, label in enumerate(labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("\nLabels:")
    print(label2id)

    # Tokenizer + model from Lab 1 decision
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    train_dataset = TopicDataset(
        train_df,
        tokenizer,
        label2id,
        args.max_length,
    )

    val_dataset = TopicDataset(
        val_df,
        tokenizer,
        label2id,
        args.max_length,
    )

    test_dataset = TopicDataset(
        test_df,
        tokenizer,
        label2id,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Device
    device = choose_device()
    model.to(device)

    print(f"\nDevice: {device}")

    if device.type == "cpu":
        print(
            "WARNING: CPU training will be slow. "
            "Consider Google Colab/GPU."
        )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    total_steps = (
        len(train_loader) * args.epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(
            1,
            int(total_steps * 0.1),
        ),
        num_training_steps=total_steps,
    )

    # Training
    print("\n=== Training ===")

    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"| step {step}/{len(train_loader)} "
                    f"| loss {loss.item():.4f}"
                )

        average_loss = (
            running_loss / len(train_loader)
        )

        val_f1, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch {epoch + 1} complete"
        )
        print(
            f"Average training loss: "
            f"{average_loss:.4f}"
        )
        print(
            f"Validation macro-F1: "
            f"{val_f1:.4f}"
        )

    train_time = (
        time.perf_counter() - start_time
    )

    # Frozen test evaluation
    print("\n=== Frozen Test Evaluation ===")

    test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        f"Frozen test macro-F1: "
        f"{test_f1:.4f}"
    )

    print(
        f"Training time: "
        f"{train_time:.2f} seconds"
    )

    target_names = [
        id2label[index]
        for index in range(len(id2label))
    ]

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )
    )

    # Save rerunnable artefact
    print(
        f"\nSaving artefact to: "
        f"{output_dir}"
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "checkpoint": CHECKPOINT,
        "validation_macro_f1": val_f1,
        "frozen_test_macro_f1": test_f1,
        "train_time_seconds": train_time,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "label2id": label2id,
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Artefact saved successfully ✅")


if __name__ == "__main__":
    main()