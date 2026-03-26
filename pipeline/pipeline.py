"""Kubeflow Pipelines + Vertex AI: main pipeline definition.

This module defines the end-to-end ML training pipeline using the
Kubeflow Pipelines v2 SDK.  The pipeline is composed of four steps:

  1. data_preprocessing  – clean and split the raw dataset
  2. training            – train a RandomForest classifier
  3. evaluation          – compute test-set metrics
  4. deployment          – conditionally deploy the model to Vertex AI

Usage (compile only):
    python pipeline.py

Usage (compile + run via scripts/run_pipeline.py):
    python scripts/run_pipeline.py --config config/pipeline_config.yaml
"""

from kfp import dsl
from kfp.dsl import component, pipeline


# ---------------------------------------------------------------------------
# Lightweight components defined with @component
# Each component runs inside the Docker image built for that step.
# ---------------------------------------------------------------------------


def _make_op(image: str, command: str, args: list) -> dsl.ContainerSpec:
    """Helper: build a ContainerSpec for a component."""
    return dsl.ContainerSpec(image=image, command=[command], args=args)


@dsl.component(base_image="python:3.10-slim")
def preprocess_op(
    input_data_uri: str,
    train_output_uri: str,
    test_output_uri: str,
    target_column: str,
    test_size: float,
    random_state: int,
) -> None:
    """Wrapper component: delegates to the containerised preprocessing script."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "preprocess.py",
            "--input-data-uri", input_data_uri,
            "--train-output-uri", train_output_uri,
            "--test-output-uri", test_output_uri,
            "--target-column", target_column,
            "--test-size", str(test_size),
            "--random-state", str(random_state),
        ],
        check=True,
    )


@dsl.component(base_image="python:3.10-slim")
def train_op(
    train_data_uri: str,
    model_output_uri: str,
    metrics_output_uri: str,
    target_column: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> None:
    """Wrapper component: delegates to the containerised training script."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--train-data-uri", train_data_uri,
            "--model-output-uri", model_output_uri,
            "--metrics-output-uri", metrics_output_uri,
            "--target-column", target_column,
            "--n-estimators", str(n_estimators),
            "--max-depth", str(max_depth),
            "--random-state", str(random_state),
        ],
        check=True,
    )


@dsl.component(base_image="python:3.10-slim")
def evaluate_op(
    model_uri: str,
    test_data_uri: str,
    metrics_output_uri: str,
    target_column: str,
) -> float:
    """Wrapper component: evaluates the model and returns test accuracy."""
    import subprocess
    import sys

    accuracy_path = "/tmp/accuracy.txt"
    subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--model-uri", model_uri,
            "--test-data-uri", test_data_uri,
            "--metrics-output-uri", metrics_output_uri,
            "--accuracy-output-path", accuracy_path,
            "--target-column", target_column,
        ],
        check=True,
    )
    with open(accuracy_path) as f:
        return float(f.read().strip())


@dsl.component(base_image="python:3.10-slim")
def deploy_op(
    project: str,
    location: str,
    model_display_name: str,
    serving_container_image_uri: str,
    artifact_uri: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    endpoint_display_name: str,
) -> None:
    """Wrapper component: deploys the model to a Vertex AI Endpoint."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "deploy.py",
            "--project", project,
            "--location", location,
            "--model-display-name", model_display_name,
            "--serving-container-image-uri", serving_container_image_uri,
            "--artifact-uri", artifact_uri,
            "--machine-type", machine_type,
            "--min-replica-count", str(min_replica_count),
            "--max-replica-count", str(max_replica_count),
            "--endpoint-display-name", endpoint_display_name,
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


@pipeline(
    name="ml-training-pipeline",
    description="End-to-end ML training pipeline using Kubeflow Pipelines on Vertex AI",
)
def ml_pipeline(
    # Data parameters
    input_data_uri: str = "gs://your-bucket/data/raw_data.csv",
    target_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    # Training parameters
    n_estimators: int = 100,
    max_depth: int = 10,
    # Evaluation parameters
    accuracy_threshold: float = 0.80,
    # GCS paths for intermediate artifacts
    pipeline_root: str = "gs://your-bucket/pipeline-root",
    # Deployment parameters
    project: str = "your-gcp-project-id",
    location: str = "us-central1",
    model_display_name: str = "my-ml-model",
    endpoint_display_name: str = "my-ml-model-endpoint",
    serving_container_image_uri: str = (
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest"
    ),
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
) -> None:
    """Define the end-to-end ML pipeline."""

    # Derived GCS paths for intermediate artifacts
    run_id = "{{$.pipeline_job_name}}"
    artifacts_base = f"{pipeline_root}/{run_id}"
    train_data_uri = f"{artifacts_base}/data/train.csv"
    test_data_uri = f"{artifacts_base}/data/test.csv"
    model_uri = f"{artifacts_base}/model/model.pkl"
    train_metrics_uri = f"{artifacts_base}/metrics/train_metrics.json"
    eval_metrics_uri = f"{artifacts_base}/metrics/eval_metrics.json"

    # Step 1: Preprocess data
    preprocess_task = preprocess_op(
        input_data_uri=input_data_uri,
        train_output_uri=train_data_uri,
        test_output_uri=test_data_uri,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 2: Train model (depends on preprocessing)
    train_task = train_op(
        train_data_uri=train_data_uri,
        model_output_uri=model_uri,
        metrics_output_uri=train_metrics_uri,
        target_column=target_column,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).after(preprocess_task)

    # Step 3: Evaluate model (depends on training)
    evaluate_task = evaluate_op(
        model_uri=model_uri,
        test_data_uri=test_data_uri,
        metrics_output_uri=eval_metrics_uri,
        target_column=target_column,
    ).after(train_task)

    # Step 4: Conditionally deploy if accuracy meets threshold
    with dsl.Condition(
        evaluate_task.output >= accuracy_threshold,
        name="accuracy-gate",
    ):
        deploy_op(
            project=project,
            location=location,
            model_display_name=model_display_name,
            serving_container_image_uri=serving_container_image_uri,
            artifact_uri=f"{artifacts_base}/model/",
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            endpoint_display_name=endpoint_display_name,
        )
