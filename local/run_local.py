"""Run the pipeline components locally for quick iteration.

This script imports and calls each component function directly in Python —
no Docker, no Kubernetes, no GCP credentials required.

It is a fast feedback loop for testing your component logic before you
containerise and submit to Vertex AI.

Usage:
    uv run python local/run_local.py
"""

import os
import sys
import tempfile

# Make the repo root importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from components.data_ingestion.component import ingest_data  # noqa: E402
from components.preprocessing.component import preprocess  # noqa: E402
from components.training.component import train  # noqa: E402


def run_pipeline_locally(
    dataset_name: str = "imdb",
    config_name: str = "default",
    model_name: str = "distilbert-base-uncased",
    text_column: str = "text",
    test_size: float = 0.2,
    max_length: int = 128,
    num_labels: int = 2,
    num_epochs: int = 1,
    per_device_batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> None:
    """Execute the full pipeline in-process using a temporary directory.

    Args:
        dataset_name:          Hugging Face Hub dataset name (e.g. "imdb").
        config_name:           Dataset configuration / subset name.
        model_name:            Hugging Face model identifier to fine-tune.
        text_column:           Name of the text column in the dataset.
        test_size:             Fraction of data held out for testing (0–1).
        max_length:            Max token sequence length for the tokenizer.
        num_labels:            Number of classification labels.
        num_epochs:            Number of training epochs (1 for local smoke-test).
        per_device_batch_size: Batch size per device.
        learning_rate:         Learning rate for AdamW.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("Local Pipeline Run")
        print(f"Working directory: {tmpdir}")
        print("=" * 60)

        # Step 1: Data ingestion
        raw_path = os.path.join(tmpdir, "raw")
        print(f"\n[Step 1] Data ingestion — loading {dataset_name!r}")
        ingest_data(dataset_name=dataset_name, output_path=raw_path, config_name=config_name)

        # Step 2: Preprocessing / tokenization
        train_path = os.path.join(tmpdir, "processed", "train")
        test_path = os.path.join(tmpdir, "processed", "test")
        print(f"\n[Step 2] Preprocessing — tokenizing with {model_name!r}")
        preprocess(
            input_path=raw_path,
            output_train_path=train_path,
            output_test_path=test_path,
            model_name=model_name,
            text_column=text_column,
            test_size=test_size,
            max_length=max_length,
        )

        # Step 3: Training
        model_path = os.path.join(tmpdir, "model")
        print(f"\n[Step 3] Training — fine-tuning for {num_epochs} epoch(s)")
        train(
            train_data_path=train_path,
            model_output_path=model_path,
            model_name=model_name,
            num_labels=num_labels,
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
        )

        print("\n" + "=" * 60)
        print("Local pipeline run complete!")
        print("Next step: containerise each component and submit to Vertex AI.")
        print("  uv run python pipeline/pipeline.py          # compile spec")
        print("  uv run python vertex_ai/submit_pipeline.py  # submit to Vertex AI")
        print("=" * 60)


if __name__ == "__main__":
    run_pipeline_locally()
