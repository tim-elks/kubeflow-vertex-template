"""Submit and monitor a Kubeflow pipeline run on Vertex AI Pipelines.

Usage:
    python scripts/run_pipeline.py --config config/pipeline_config.yaml

Prerequisites:
    - Authenticated via `gcloud auth application-default login` or a
      service account key in GOOGLE_APPLICATION_CREDENTIALS.
    - KFP SDK and google-cloud-aiplatform installed.
    - Pipeline Docker images built and pushed (see scripts/build_components.sh).
"""

import argparse
import logging
import os

import yaml
from google.cloud import aiplatform
from kfp import compiler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pipeline import ml_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML pipeline configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str, pipeline_spec_path: str = "/tmp/pipeline_spec.yaml") -> None:
    """Compile and submit the pipeline to Vertex AI Pipelines.

    Args:
        config_path: Path to pipeline_config.yaml.
        pipeline_spec_path: Temporary path for the compiled pipeline YAML.
    """
    cfg = load_config(config_path)

    project = cfg["gcp"]["project_id"]
    region = cfg["gcp"]["region"]
    pipeline_root = cfg["gcp"]["pipeline_root"]

    # Compile the pipeline
    logger.info("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path=pipeline_spec_path,
    )
    logger.info("Pipeline compiled to %s", pipeline_spec_path)

    # Initialize Vertex AI SDK
    aiplatform.init(project=project, location=region)

    # Build pipeline parameters from config
    pipeline_params = {
        "input_data_uri": cfg["data"]["input_data_uri"],
        "target_column": cfg["data"]["target_column"],
        "test_size": cfg["preprocessing"]["test_size"],
        "random_state": cfg["preprocessing"]["random_state"],
        "n_estimators": cfg["training"]["n_estimators"],
        "max_depth": cfg["training"]["max_depth"],
        "accuracy_threshold": cfg["evaluation"]["accuracy_threshold"],
        "pipeline_root": pipeline_root,
        "project": project,
        "location": region,
        "model_display_name": cfg["deployment"]["model_display_name"],
        "endpoint_display_name": cfg["deployment"]["endpoint_display_name"],
        "serving_container_image_uri": cfg["deployment"]["serving_container_image_uri"],
        "machine_type": cfg["deployment"]["machine_type"],
        "min_replica_count": cfg["deployment"]["min_replica_count"],
        "max_replica_count": cfg["deployment"]["max_replica_count"],
    }

    # Submit the pipeline run
    logger.info("Submitting pipeline to Vertex AI Pipelines in project '%s', region '%s'...", project, region)
    job = aiplatform.PipelineJob(
        display_name=cfg["pipeline"]["name"],
        template_path=pipeline_spec_path,
        pipeline_root=pipeline_root,
        parameter_values=pipeline_params,
    )
    job.submit()
    logger.info("Pipeline job submitted: %s", job.resource_name)
    logger.info(
        "Monitor the run at: https://console.cloud.google.com/vertex-ai/pipelines?project=%s",
        project,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the ML pipeline to Vertex AI Pipelines"
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML (default: config/pipeline_config.yaml)",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
