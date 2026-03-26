"""Kubeflow Pipelines pipeline definition.

This module assembles the individual pipeline components into a complete
KFP v2 pipeline that can be compiled to a Vertex AI-compatible YAML spec.

Usage:
    uv run python pipeline/pipeline.py

This will produce `pipeline/pipeline.yaml` which can be submitted to
Vertex AI Pipelines via `vertex_ai/submit_pipeline.py`.
"""

import os

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# ---------------------------------------------------------------------------
# Configuration — update these to match your environment
# ---------------------------------------------------------------------------
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
    dataset_name: str,
    config_name: str,
    raw_dataset: Output[Dataset],
) -> None:
    """KFP component wrapper for components/data_ingestion/component.py."""
    import subprocess

    subprocess.run(
        [
            "/app/.venv/bin/python",
            "/app/component.py",
            "--dataset-name",
            dataset_name,
            "--config-name",
            config_name,
            "--output-path",
            raw_dataset.path,
        ],
        check=True,
    )


@dsl.component(base_image=PREPROCESSING_IMAGE)
def preprocess_op(
    raw_dataset: Input[Dataset],
    model_name: str,
    text_column: str,
    test_size: float,
    max_length: int,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
) -> None:
    """KFP component wrapper for components/preprocessing/component.py."""
    import subprocess

    subprocess.run(
        [
            "/app/.venv/bin/python",
            "/app/component.py",
            "--input-path",
            raw_dataset.path,
            "--output-train-path",
            train_dataset.path,
            "--output-test-path",
            test_dataset.path,
            "--model-name",
            model_name,
            "--text-column",
            text_column,
            "--test-size",
            str(test_size),
            "--max-length",
            str(max_length),
        ],
        check=True,
    )


@dsl.component(base_image=TRAINING_IMAGE)
def train_op(
    train_dataset: Input[Dataset],
    model_name: str,
    num_labels: int,
    num_epochs: int,
    per_device_batch_size: int,
    learning_rate: float,
    model: Output[Model],
) -> None:
    """KFP component wrapper for components/training/component.py."""
    import subprocess

    subprocess.run(
        [
            "/app/.venv/bin/python",
            "/app/component.py",
            "--train-data-path",
            train_dataset.path,
            "--model-output-path",
            model.path,
            "--model-name",
            model_name,
            "--num-labels",
            str(num_labels),
            "--num-epochs",
            str(num_epochs),
            "--per-device-batch-size",
            str(per_device_batch_size),
            "--learning-rate",
            str(learning_rate),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name="kubeflow-vertex-template-pipeline",
    description="HuggingFace fine-tuning pipeline: ingest → tokenize → train",
)
def ml_pipeline(
    dataset_name: str = "imdb",
    config_name: str = "default",
    model_name: str = "distilbert-base-uncased",
    text_column: str = "text",
    test_size: float = 0.2,
    max_length: int = 128,
    num_labels: int = 2,
    num_epochs: int = 3,
    per_device_batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    """End-to-end HuggingFace fine-tuning pipeline.

    Args:
        dataset_name:          Hugging Face Hub dataset name (e.g. "imdb").
        config_name:           Dataset configuration / subset name.
        model_name:            Hugging Face model identifier to fine-tune.
        text_column:           Name of the text column in the dataset.
        test_size:             Fraction of data held out for testing (0–1).
        max_length:            Max token sequence length for the tokenizer.
        num_labels:            Number of classification labels.
        num_epochs:            Number of training epochs.
        per_device_batch_size: Batch size per device.
        learning_rate:         Learning rate for AdamW.
    """
    ingest_task = ingest_data_op(dataset_name=dataset_name, config_name=config_name)

    preprocess_task = preprocess_op(
        raw_dataset=ingest_task.outputs["raw_dataset"],
        model_name=model_name,
        text_column=text_column,
        test_size=test_size,
        max_length=max_length,
    )

    train_op(
        train_dataset=preprocess_task.outputs["train_dataset"],
        model_name=model_name,
        num_labels=num_labels,
        num_epochs=num_epochs,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
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
    print("Done. Submit the spec with: uv run python vertex_ai/submit_pipeline.py")
