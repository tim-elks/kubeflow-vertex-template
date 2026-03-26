"""Preprocessing pipeline component.

This component reads the raw dataset produced by data_ingestion, applies
feature engineering and train/test splitting, and writes the results to
output paths that downstream components can consume.
"""

import click
import json
import os


def preprocess(input_path: str, output_train_path: str, output_test_path: str, test_size: float = 0.2) -> None:
    """Apply preprocessing to the raw dataset and split into train/test sets.

    Args:
        input_path:        Path to the raw dataset artifact from data_ingestion.
        output_train_path: Destination path for the training split.
        output_test_path:  Destination path for the test split.
        test_size:         Fraction of data to reserve for testing (0–1).
    """
    print(f"[preprocessing] Loading data from: {input_path}")
    with open(input_path) as f:
        raw = json.load(f)

    # -----------------------------------------------------------------
    # Replace the block below with your actual preprocessing logic, e.g.:
    #   df = pd.DataFrame(raw)
    #   df = df.dropna()
    #   df["feature_1"] = (df["feature_1"] - mean) / std
    #   train, test = train_test_split(df, test_size=test_size)
    # -----------------------------------------------------------------
    total_rows = raw.get("rows", 1000)
    test_rows = int(total_rows * test_size)
    train_rows = total_rows - test_rows

    train_data = {**raw, "split": "train", "rows": train_rows}
    test_data = {**raw, "split": "test", "rows": test_rows}

    for path, data in [(output_train_path, train_data), (output_test_path, test_data)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[preprocessing] Written: {path} ({data['rows']} rows)")


@click.command()
@click.option("--input-path", required=True, help="Raw dataset path")
@click.option("--output-train-path", required=True, help="Training split output path")
@click.option("--output-test-path", required=True, help="Test split output path")
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Test fraction")
def main(input_path: str, output_train_path: str, output_test_path: str, test_size: float) -> None:
    preprocess(input_path, output_train_path, output_test_path, test_size)


if __name__ == "__main__":
    main()
