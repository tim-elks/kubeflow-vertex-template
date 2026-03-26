"""Data preprocessing component for the ML pipeline.

Reads raw data from GCS, applies preprocessing steps, and outputs
a cleaned dataset back to GCS for the training component to consume.
"""

import argparse
import logging
import os

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_from_gcs(gcs_uri: str, local_path: str) -> None:
    """Download a file from GCS to a local path."""
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logger.info("Downloaded %s to %s", gcs_uri, local_path)


def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s to %s", local_path, gcs_uri)


def preprocess(
    input_data_uri: str,
    train_output_uri: str,
    test_output_uri: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess raw data and split into train/test sets.

    Args:
        input_data_uri: GCS URI of the raw input CSV data.
        train_output_uri: GCS URI to write the preprocessed training CSV.
        test_output_uri: GCS URI to write the preprocessed test CSV.
        target_column: Name of the target/label column in the dataset.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.
    """
    local_input = "/tmp/raw_data.csv"
    local_train = "/tmp/train_data.csv"
    local_test = "/tmp/test_data.csv"

    # Download raw data
    download_from_gcs(input_data_uri, local_input)

    # Load and inspect
    df = pd.read_csv(local_input)
    logger.info("Loaded dataset with shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())

    # Drop rows with missing values
    df = df.dropna()
    logger.info("Shape after dropping NaN rows: %s", df.shape)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train.values

    test_df = X_test.copy()
    test_df[target_column] = y_test.values

    # Save locally then upload
    train_df.to_csv(local_train, index=False)
    test_df.to_csv(local_test, index=False)

    upload_to_gcs(local_train, train_output_uri)
    upload_to_gcs(local_test, test_output_uri)

    logger.info(
        "Preprocessing complete. Train size: %d, Test size: %d",
        len(train_df),
        len(test_df),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Data preprocessing component")
    parser.add_argument("--input-data-uri", required=True)
    parser.add_argument("--train-output-uri", required=True)
    parser.add_argument("--test-output-uri", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    preprocess(
        input_data_uri=args.input_data_uri,
        train_output_uri=args.train_output_uri,
        test_output_uri=args.test_output_uri,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
