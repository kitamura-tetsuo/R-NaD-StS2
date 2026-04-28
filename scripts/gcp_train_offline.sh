#!/bin/bash

# Configuration - Update these values!
PROJECT_ID="r-nad-sts2" # Change to your GCP project ID
REGION="us-central1"
REPO_NAME="rnad-containers"
IMAGE_NAME="rnad-sts2-offline"
TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# 1. Build and Push Docker Image
echo "Building Docker image: $IMAGE_URI"
docker build -t "$IMAGE_URI" -f Dockerfile.vertex .

echo "Pushing image to Artifact Registry..."
docker push "$IMAGE_URI"

# 2. Update config_vertex.yaml with the correct image URI
# Note: This uses sed to replace the imageUri line in a temporary config
sed "s|imageUri:.*|imageUri: $IMAGE_URI|" config_vertex.yaml > config_vertex_run.yaml

# 3. Submit Vertex AI Custom Job
echo "Submitting Vertex AI Custom Job..."
JOB_ID=$(gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="rnad-sts2-offline-training" \
  --config=config_vertex_run.yaml \
  --format="value(name)")

if [ -n "$JOB_ID" ]; then
  echo "Job created successfully: $JOB_ID"
  echo "Streaming logs (Ctrl+C to stop streaming, job will continue)..."
  gcloud ai custom-jobs stream-logs "$JOB_ID" --region="$REGION"
else
  echo "Failed to create job."
  exit 1
fi
