"""Lab 3A: dataset construction and leakage-safe grouped splits."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bayan.preprocessing.core import preprocess


DATA_PATH = "data/raw/bayan_feedback.csv"


def build_topic_dataset(
    data_path: str = DATA_PATH,
    random_state: int = 42,
):
    """Build leakage-safe train/validation/test topic splits."""

    df = pd.read_csv(data_path)

    required_columns = {
        "text",
        "topic",
        "citizen_group_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Use the shared Lab 1 preprocessing.
    df = df.copy()
    df["clean_text"] = (
        df["text"]
        .astype(str)
        .apply(preprocess)
    )

    # ---------------------------------------------------------
    # Split 1:
    # 70% train
    # 30% temporary validation/test pool
    #
    # GroupShuffleSplit keeps each citizen_group_id entirely
    # inside one side of the split.
    # ---------------------------------------------------------
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        first_split.split(
            df,
            y=df["topic"],
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # ---------------------------------------------------------
    # Split 2:
    # Divide the remaining 30% into approximately:
    # 20% validation
    # 10% test
    #
    # 1/3 of the temporary 30% becomes test.
    # ---------------------------------------------------------
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=1 / 3,
        random_state=random_state,
    )

    validation_idx, test_idx = next(
        second_split.split(
            temp_df,
            y=temp_df["topic"],
            groups=temp_df["citizen_group_id"],
        )
    )

    validation_df = temp_df.iloc[validation_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Reset indices so each split is clean and independent.
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }