"""Lab 3A: TF-IDF + LinearSVC baseline."""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset


def main():
    # Use exactly the same leakage-safe grouped split
    # as the Transformer classifier.
    dataset = build_topic_dataset()

    train_df = dataset["train"]
    val_df = dataset["validation"]
    test_df = dataset["test"]

    print("=== Dataset sizes ===")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )

    start_time = time.perf_counter()

    X_train = vectorizer.fit_transform(
        train_df["clean_text"]
    )

    X_val = vectorizer.transform(
        val_df["clean_text"]
    )

    X_test = vectorizer.transform(
        test_df["clean_text"]
    )

    model = LinearSVC()

    model.fit(
        X_train,
        train_df["topic"],
    )

    train_time = time.perf_counter() - start_time

    val_pred = model.predict(X_val)

    val_macro_f1 = f1_score(
        val_df["topic"],
        val_pred,
        average="macro",
    )

    test_pred = model.predict(X_test)

    test_macro_f1 = f1_score(
        test_df["topic"],
        test_pred,
        average="macro",
    )

    print("\n=== TF-IDF + LinearSVC Baseline ===")
    print(
        f"Validation macro-F1: "
        f"{val_macro_f1:.4f}"
    )
    print(
        f"Frozen test macro-F1: "
        f"{test_macro_f1:.4f}"
    )
    print(
        f"Train time: "
        f"{train_time:.2f} seconds"
    )

    print("\n=== Frozen Test Classification Report ===")

    print(
        classification_report(
            test_df["topic"],
            test_pred,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()