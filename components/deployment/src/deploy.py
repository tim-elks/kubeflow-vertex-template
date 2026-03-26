"""Model deployment component for the ML pipeline.

Deploys a trained model to a Vertex AI Endpoint. If an endpoint already
exists for the model, the new version is deployed there; otherwise a new
endpoint is created.
"""

import argparse
import logging

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deploy(
    project: str,
    location: str,
    model_display_name: str,
    serving_container_image_uri: str,
    artifact_uri: str,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
    endpoint_display_name: str = "",
) -> None:
    """Upload a trained model to Vertex AI Model Registry and deploy it.

    Args:
        project: GCP project ID.
        location: GCP region (e.g. 'us-central1').
        model_display_name: Display name for the Vertex AI Model resource.
        serving_container_image_uri: URI of the Docker image that serves predictions.
        artifact_uri: GCS URI of the directory containing model artifacts.
        machine_type: Compute machine type for the endpoint.
        min_replica_count: Minimum number of serving replicas.
        max_replica_count: Maximum number of serving replicas.
        endpoint_display_name: Display name for the Vertex AI Endpoint resource.
            Defaults to '<model_display_name>-endpoint'.
    """
    aiplatform.init(project=project, location=location)

    if not endpoint_display_name:
        endpoint_display_name = f"{model_display_name}-endpoint"

    # Upload model to Vertex AI Model Registry
    logger.info("Uploading model '%s' to Vertex AI...", model_display_name)
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
    )
    logger.info("Model uploaded: %s", model.resource_name)

    # Create or reuse an endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )

    if endpoints:
        endpoint = endpoints[0]
        logger.info("Reusing existing endpoint: %s", endpoint.resource_name)
    else:
        logger.info("Creating new endpoint '%s'...", endpoint_display_name)
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        logger.info("Endpoint created: %s", endpoint.resource_name)

    # Deploy model to endpoint
    logger.info(
        "Deploying model to endpoint (machine_type=%s, replicas=%d-%d)...",
        machine_type,
        min_replica_count,
        max_replica_count,
    )
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=100,
    )

    logger.info(
        "Deployment complete. Endpoint resource name: %s", endpoint.resource_name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Model deployment component")
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--model-display-name", required=True)
    parser.add_argument("--serving-container-image-uri", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--machine-type", default="n1-standard-4")
    parser.add_argument("--min-replica-count", type=int, default=1)
    parser.add_argument("--max-replica-count", type=int, default=3)
    parser.add_argument("--endpoint-display-name", default="")
    args = parser.parse_args()

    deploy(
        project=args.project,
        location=args.location,
        model_display_name=args.model_display_name,
        serving_container_image_uri=args.serving_container_image_uri,
        artifact_uri=args.artifact_uri,
        machine_type=args.machine_type,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
        endpoint_display_name=args.endpoint_display_name,
    )


if __name__ == "__main__":
    main()
