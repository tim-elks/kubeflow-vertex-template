"""Model evaluation component for the ML pipeline.

Loads the trained model and test dataset from GCS, computes evaluation
metrics, and writes a metrics JSON artifact. The component outputs an
accuracy value that can be used to gate deployment.
"""

import argparse
import json
import logging
import pickle

import pandas as pd
from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def evaluate(
    model_uri: str,
    test_data_uri: str,
    metrics_output_uri: str,
    accuracy_output_path: str,
    target_column: str,
) -> None:
    """Evaluate a trained model on the held-out test set.

    Args:
        model_uri: GCS URI of the serialized model (pickle).
        test_data_uri: GCS URI of the preprocessed test CSV.
        metrics_output_uri: GCS URI to write full evaluation metrics as JSON.
        accuracy_output_path: Local path to write the scalar accuracy value.
            Kubeflow Pipelines reads this to pass the value between components.
        target_column: Name of the target/label column.
    """
    local_model = "/tmp/model.pkl"
    local_test = "/tmp/test_data.csv"
    local_metrics = "/tmp/eval_metrics.json"

    # Download artifacts
    download_from_gcs(model_uri, local_model)
    download_from_gcs(test_data_uri, local_test)

    # Load model and data
    with open(local_model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(local_test)
    logger.info("Loaded test data with shape: %s", df.shape)

    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    logger.info("Test accuracy: %.4f", accuracy)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

    # Write metrics JSON and upload to GCS
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    upload_to_gcs(local_metrics, metrics_output_uri)

    # Write scalar accuracy to local path for Kubeflow Pipelines to consume
    with open(accuracy_output_path, "w") as f:
        f.write(str(accuracy))

    logger.info("Evaluation complete. Metrics saved to %s", metrics_output_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model evaluation component")
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--test-data-uri", required=True)
    parser.add_argument("--metrics-output-uri", required=True)
    parser.add_argument("--accuracy-output-path", required=True)
    parser.add_argument("--target-column", required=True)
    args = parser.parse_args()

    evaluate(
        model_uri=args.model_uri,
        test_data_uri=args.test_data_uri,
        metrics_output_uri=args.metrics_output_uri,
        accuracy_output_path=args.accuracy_output_path,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
