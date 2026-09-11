"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


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
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
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
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        self.labels = [
            label2id[label]
            for label in dataframe["topic"]
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


def compute_metrics(eval_pred):
    """Compute macro-F1 and accuracy."""

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    print("=== Bayan Topic Classifier ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Max length: {args.max_length}")

    # Leakage-safe grouped splits from Lab 3A Step 2
    dataset = build_topic_dataset()

    train_df = dataset["train"]
    val_df = dataset["validation"]
    test_df = dataset["test"]

    print("\n=== Dataset sizes ===")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

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

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
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

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            CHECKPOINT,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        learning_rate=args.learning_rate,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,
        save_safetensors=False,

        logging_strategy="steps",
        logging_steps=50,

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print(f"\nDevice: {device_name}")

    print("\n=== Training ===")

    start_time = time.perf_counter()

    trainer.train()

    training_time = (
        time.perf_counter() - start_time
    )

    print("\n=== Validation Evaluation ===")

    validation_metrics = trainer.evaluate(
        val_dataset,
        metric_key_prefix="validation",
    )

    validation_f1 = validation_metrics[
        "validation_macro_f1"
    ]

    validation_accuracy = validation_metrics[
        "validation_accuracy"
    ]

    print(
        f"Validation macro-F1: "
        f"{validation_f1:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    # Evaluate frozen test exactly once after
    # training/best-checkpoint selection.
    print("\n=== Frozen Test Evaluation ===")

    test_output = trainer.predict(
        test_dataset,
        metric_key_prefix="test",
    )

    test_f1 = test_output.metrics[
        "test_macro_f1"
    ]

    test_accuracy = test_output.metrics[
        "test_accuracy"
    ]

    predictions = np.argmax(
        test_output.predictions,
        axis=-1,
    )

    references = test_output.label_ids

    print(
        f"Frozen test macro-F1: "
        f"{test_f1:.4f}"
    )

    print(
        f"Frozen test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    target_names = [
        id2label[index]
        for index in range(len(id2label))
    ]

    print("\nFrozen test classification report:")

    print(
        classification_report(
            references,
            predictions,
            target_names=target_names,
            digits=4,
        )
    )

    print(
        f"\nSaving best classifier artefact to: "
        f"{output_dir}"
    )

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    metrics = {
        "checkpoint": CHECKPOINT,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_macro_f1": validation_f1,
        "validation_accuracy": validation_accuracy,
        "frozen_test_macro_f1": test_f1,
        "frozen_test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "label2id": label2id,
    }

    with (
        output_dir / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Classifier artefact saved successfully ✅")


if __name__ == "__main__":
    main()