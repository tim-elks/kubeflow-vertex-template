"""Compile the Kubeflow pipeline to a YAML spec for Vertex AI Pipelines.

Run this script to produce the compiled pipeline YAML that can be
submitted directly to Vertex AI or stored as an artifact.

Usage:
    python pipeline/compile_pipeline.py [--output pipeline_spec.yaml]
"""

import argparse
import logging

from kfp import compiler

from pipeline import ml_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_pipeline(output_path: str = "pipeline_spec.yaml") -> None:
    """Compile the ML pipeline to a Vertex AI-compatible YAML spec.

    Args:
        output_path: Local path where the compiled YAML will be written.
    """
    logger.info("Compiling pipeline to %s ...", output_path)
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path=output_path,
    )
    logger.info("Pipeline compiled successfully: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the Kubeflow pipeline to a Vertex AI-compatible YAML spec"
    )
    parser.add_argument(
        "--output",
        default="pipeline_spec.yaml",
        help="Output path for the compiled pipeline YAML (default: pipeline_spec.yaml)",
    )
    args = parser.parse_args()
    compile_pipeline(output_path=args.output)


if __name__ == "__main__":
    main()
