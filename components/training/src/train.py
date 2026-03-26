"""Model training component for the ML pipeline.

Reads preprocessed training data from GCS, trains a scikit-learn model,
and saves the trained model artifact back to GCS.
"""

import argparse
import json
import logging
import os
import pickle

import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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


def train(
    train_data_uri: str,
    model_output_uri: str,
    metrics_output_uri: str,
    target_column: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> None:
    """Train a RandomForest classifier and save the model artifact.

    Args:
        train_data_uri: GCS URI of the preprocessed training CSV.
        model_output_uri: GCS URI to write the serialized model (pickle).
        metrics_output_uri: GCS URI to write training metrics as JSON.
        target_column: Name of the target/label column.
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum depth of each tree.
        random_state: Random seed for reproducibility.
    """
    local_train = "/tmp/train_data.csv"
    local_model = "/tmp/model.pkl"
    local_metrics = "/tmp/train_metrics.json"

    # Download training data
    download_from_gcs(train_data_uri, local_train)

    df = pd.read_csv(local_train)
    logger.info("Loaded training data with shape: %s", df.shape)

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]

    # Train model
    logger.info(
        "Training RandomForestClassifier (n_estimators=%d, max_depth=%d)",
        n_estimators,
        max_depth,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    report = classification_report(y_train, y_pred, output_dict=True)

    metrics = {
        "train_accuracy": train_accuracy,
        "classification_report": report,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
    }
    logger.info("Training accuracy: %.4f", train_accuracy)

    # Save model
    with open(local_model, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Upload artifacts to GCS
    upload_to_gcs(local_model, model_output_uri)
    upload_to_gcs(local_metrics, metrics_output_uri)

    logger.info("Training complete. Model saved to %s", model_output_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model training component")
    parser.add_argument("--train-data-uri", required=True)
    parser.add_argument("--model-output-uri", required=True)
    parser.add_argument("--metrics-output-uri", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(
        train_data_uri=args.train_data_uri,
        model_output_uri=args.model_output_uri,
        metrics_output_uri=args.metrics_output_uri,
        target_column=args.target_column,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
