"""Submit a compiled Kubeflow pipeline to Vertex AI Pipelines.

Prerequisites:
    1. Authenticate with GCP:
           gcloud auth application-default login
    2. Compile the pipeline:
           python pipeline/pipeline.py
    3. Update vertex_ai/config.yaml with your project details.

Usage:
    python vertex_ai/submit_pipeline.py [--config vertex_ai/config.yaml]
"""

import argparse
import os
import sys

import yaml


def load_config(config_path: str) -> dict:
    """Load and return configuration from a YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def submit_pipeline(config: dict) -> None:
    """Compile (if needed) and submit the pipeline to Vertex AI.

    Args:
        config: Parsed configuration dictionary from config.yaml.
    """
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("ERROR: google-cloud-aiplatform is not installed.")
        print("       Run: uv sync  (or: pip install google-cloud-aiplatform)")
        sys.exit(1)

    gcp = config["gcp"]
    pipeline_cfg = config["pipeline"]

    project_id = gcp["project_id"]
    region = gcp["region"]
    pipeline_root = gcp["pipeline_root"]
    spec_path = pipeline_cfg["spec_path"]
    display_name = pipeline_cfg["display_name"]
    parameters = pipeline_cfg.get("parameters", {})
    enable_caching = pipeline_cfg.get("enable_caching", True)

    # Validate that the compiled spec exists
    if not os.path.exists(spec_path):
        print(f"ERROR: Pipeline spec not found at '{spec_path}'.")
        print("       Run 'python pipeline/pipeline.py' to compile it first.")
        sys.exit(1)

    print(f"Initialising Vertex AI SDK (project={project_id}, region={region})")
    aiplatform.init(project=project_id, location=region)

    print(f"Submitting pipeline '{display_name}' ...")
    print(f"  Spec      : {spec_path}")
    print(f"  Root      : {pipeline_root}")
    print(f"  Parameters: {parameters}")
    print(f"  Caching   : {enable_caching}")

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=spec_path,
        pipeline_root=pipeline_root,
        parameter_values=parameters,
        enable_caching=enable_caching,
    )

    job.submit()

    print("\nPipeline submitted successfully!")
    print("Monitor progress in the Vertex AI console:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines?project={project_id}")
    print(f"\nJob resource name: {job.resource_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit pipeline to Vertex AI")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml (default: vertex_ai/config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    submit_pipeline(config)


if __name__ == "__main__":
    main()
