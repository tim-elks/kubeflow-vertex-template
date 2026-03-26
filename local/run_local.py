"""Run the pipeline components locally for quick iteration.

This script imports and calls each component function directly in Python —
no Docker, no Kubernetes, no GCP credentials required.

It is a fast feedback loop for testing your component logic before you
containerise and submit to Vertex AI.

Usage:
    python local/run_local.py
"""

import json
import os
import sys
import tempfile

# Make the repo root importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from components.data_ingestion.component import ingest_data
from components.preprocessing.component import preprocess
from components.training.component import train


def run_pipeline_locally(
    data_source: str = "local://sample-data",
    test_size: float = 0.2,
    n_estimators: int = 10,
    max_depth: int = 3,
) -> None:
    """Execute the full pipeline in-process using a temporary directory.

    Args:
        data_source:   Identifier passed to the data ingestion component.
        test_size:     Test split fraction (0–1).
        n_estimators:  Number of estimators for the training component.
        max_depth:     Max depth for the training component.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("Local Pipeline Run")
        print(f"Working directory: {tmpdir}")
        print("=" * 60)

        # Step 1: Data ingestion
        raw_path = os.path.join(tmpdir, "raw", "dataset.json")
        print("\n[Step 1] Data ingestion")
        ingest_data(data_source=data_source, output_path=raw_path)
        _print_artifact(raw_path, "Raw dataset")

        # Step 2: Preprocessing
        train_path = os.path.join(tmpdir, "processed", "train.json")
        test_path = os.path.join(tmpdir, "processed", "test.json")
        print("\n[Step 2] Preprocessing")
        preprocess(
            input_path=raw_path,
            output_train_path=train_path,
            output_test_path=test_path,
            test_size=test_size,
        )
        _print_artifact(train_path, "Training split")
        _print_artifact(test_path, "Test split")

        # Step 3: Training
        model_path = os.path.join(tmpdir, "model", "model.json")
        print("\n[Step 3] Training")
        train(
            train_data_path=train_path,
            model_output_path=model_path,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        _print_artifact(model_path, "Model artifact")

        print("\n" + "=" * 60)
        print("Local pipeline run complete!")
        print("Next step: containerise each component and submit to Vertex AI.")
        print("  uv run python pipeline/pipeline.py          # compile spec")
        print("  uv run python vertex_ai/submit_pipeline.py  # submit to Vertex AI")
        print("=" * 60)


def _print_artifact(path: str, label: str) -> None:
    """Print a summary of a JSON artifact file."""
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"  {label}: {data}")
    except Exception as exc:
        print(f"  {label}: [could not read: {exc}]")


if __name__ == "__main__":
    run_pipeline_locally()
