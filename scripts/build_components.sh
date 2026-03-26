#!/usr/bin/env bash
# build_components.sh
#
# Build and push Docker images for all pipeline components to Google Container Registry.
#
# Usage:
#   export GCP_PROJECT=your-gcp-project-id
#   export IMAGE_TAG=latest          # optional, defaults to 'latest'
#   bash scripts/build_components.sh
#
# Prerequisites:
#   - Docker installed and running
#   - Authenticated with GCP: gcloud auth configure-docker

set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:?Please export GCP_PROJECT=your-gcp-project-id}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="gcr.io/${GCP_PROJECT}"

COMPONENTS=(
  "data_preprocessing:preprocessing"
  "training:training"
  "evaluation:evaluation"
  "deployment:deployment"
)

echo "==> Building and pushing pipeline component images to ${REGISTRY}"
echo "    Tag: ${IMAGE_TAG}"
echo ""

for entry in "${COMPONENTS[@]}"; do
  dir="${entry%%:*}"
  name="${entry##*:}"
  image="${REGISTRY}/${name}:${IMAGE_TAG}"

  echo "--- Building ${image} (context: components/${dir}) ---"
  docker build \
    --platform linux/amd64 \
    -t "${image}" \
    "components/${dir}"

  echo "--- Pushing ${image} ---"
  docker push "${image}"
  echo ""
done

echo "==> All images built and pushed successfully."
echo ""
echo "Update config/pipeline_config.yaml with the following image URIs:"
for entry in "${COMPONENTS[@]}"; do
  dir="${entry%%:*}"
  name="${entry##*:}"
  echo "  ${name}: ${REGISTRY}/${name}:${IMAGE_TAG}"
done
