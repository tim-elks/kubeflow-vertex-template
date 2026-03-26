"""Kubeflow Pipelines pipeline definition.

This module assembles the individual pipeline components into a complete
KFP v2 pipeline that can be compiled to a Vertex AI-compatible YAML spec.

Usage:
    python pipeline/pipeline.py

This will produce `pipeline/pipeline.yaml` which can be submitted to
Vertex AI Pipelines via `vertex_ai/submit_pipeline.py`.
"""

import os
import sys

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# ---------------------------------------------------------------------------
# Configuration — update these to match your environment
# ---------------------------------------------------------------------------
BASE_IMAGE = os.environ.get(
    "BASE_IMAGE",
    "us-central1-docker.pkg.dev/your-project-id/your-repo/training:latest",
)

DATA_INGESTION_IMAGE = os.environ.get(
    "DATA_INGESTION_IMAGE",
    "us-central1-docker.pkg.dev/your-project-id/your-repo/data-ingestion:latest",
)

PREPROCESSING_IMAGE = os.environ.get(
    "PREPROCESSING_IMAGE",
    "us-central1-docker.pkg.dev/your-project-id/your-repo/preprocessing:latest",
)

TRAINING_IMAGE = os.environ.get(
    "TRAINING_IMAGE",
    "us-central1-docker.pkg.dev/your-project-id/your-repo/training:latest",
)

OUTPUT_SPEC = os.path.join(os.path.dirname(__file__), "pipeline.yaml")


# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------


@dsl.component(base_image=DATA_INGESTION_IMAGE)
def ingest_data_op(
    data_source: str,
    raw_dataset: Output[Dataset],
) -> None:
    """KFP component wrapper for components/data_ingestion/component.py."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "/app/component.py",
            "--data-source",
            data_source,
            "--output-path",
            raw_dataset.path,
        ],
        check=True,
    )


@dsl.component(base_image=PREPROCESSING_IMAGE)
def preprocess_op(
    raw_dataset: Input[Dataset],
    test_size: float,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
) -> None:
    """KFP component wrapper for components/preprocessing/component.py."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "/app/component.py",
            "--input-path",
            raw_dataset.path,
            "--output-train-path",
            train_dataset.path,
            "--output-test-path",
            test_dataset.path,
            "--test-size",
            str(test_size),
        ],
        check=True,
    )


@dsl.component(base_image=TRAINING_IMAGE)
def train_op(
    train_dataset: Input[Dataset],
    n_estimators: int,
    max_depth: int,
    model: Output[Model],
) -> None:
    """KFP component wrapper for components/training/component.py."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "/app/component.py",
            "--train-data-path",
            train_dataset.path,
            "--model-output-path",
            model.path,
            "--n-estimators",
            str(n_estimators),
            "--max-depth",
            str(max_depth),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name="kubeflow-vertex-template-pipeline",
    description="Example ML pipeline: ingest → preprocess → train",
)
def ml_pipeline(
    data_source: str = "gs://your-bucket/raw-data/dataset.csv",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> None:
    """End-to-end ML pipeline.

    Args:
        data_source:   GCS URI or local path for the raw dataset.
        test_size:     Fraction of data to hold out for testing (0–1).
        n_estimators:  Number of estimators for the training model.
        max_depth:     Maximum depth for tree-based models.
    """
    ingest_task = ingest_data_op(data_source=data_source)

    preprocess_task = preprocess_op(
        raw_dataset=ingest_task.outputs["raw_dataset"],
        test_size=test_size,
    )

    train_op(
        train_dataset=preprocess_task.outputs["train_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Compile when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Compiling pipeline to: {OUTPUT_SPEC}")
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path=OUTPUT_SPEC,
    )
    print("Done. Submit the spec with: python vertex_ai/submit_pipeline.py")
