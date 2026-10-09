#!/bin/bash
set -e

PROJECT_ID="YOUR_GCP_PROJECT_ID"
REGION="asia-south1"
REPOSITORY="sentiment-repo"
SERVICE="sentiment-api"

gcloud config set project "$PROJECT_ID"

gcloud services enable   artifactregistry.googleapis.com   cloudbuild.googleapis.com   run.googleapis.com

gcloud artifacts repositories create "$REPOSITORY"   --repository-format=docker   --location="$REGION" || true

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/sentiment-api:latest"

gcloud builds submit --tag "$IMAGE" .

gcloud run deploy "$SERVICE"   --image "$IMAGE"   --region "$REGION"   --platform managed   --allow-unauthenticated

echo "Cloud Run deployment complete."
