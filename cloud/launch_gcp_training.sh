#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="round-office-505007-q4"
REGION="us-central1"
BUCKET=""
MAX_IMAGES="100000"
RUN_ID="style-openimages-$(date -u +%Y%m%d-%H%M%S)"
EXECUTE=false
REPOSITORY="furniture-ai-training"
SERVICE_ACCOUNT_NAME="furniture-ml"

usage() {
  cat <<'USAGE'
Create a private GCS dataset bucket, build the NVIDIA training image, and submit
a paid Vertex AI job using one NVIDIA L4 GPU.

Usage:
  bash cloud/launch_gcp_training.sh --execute [options]

Options:
  --project ID         Google Cloud project (default: round-office-505007-q4)
  --region REGION      Vertex/Artifact Registry region (default: us-central1)
  --bucket NAME        Private bucket (default: furniture-ai-PROJECT)
  --run-id ID          Resumable run id (default: timestamped)
  --max-images N       Target accepted images (default: 100000)
  --execute            Acknowledge that billable resources will be created
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --project)
      PROJECT="${2:?missing value for --project}"
      shift 2
      ;;
    --region)
      REGION="${2:?missing value for --region}"
      shift 2
      ;;
    --bucket)
      BUCKET="${2:?missing value for --bucket}"
      BUCKET="${BUCKET#gs://}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="${2:?missing value for --max-images}"
      shift 2
      ;;
    --execute)
      EXECUTE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  BUCKET="furniture-ai-${PROJECT}"
fi

[[ "$PROJECT" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || {
  printf 'Invalid Google Cloud project id: %s\n' "$PROJECT" >&2
  exit 2
}
[[ "$REGION" =~ ^[a-z0-9]+(-[a-z0-9]+)+$ ]] || {
  printf 'Invalid Google Cloud region: %s\n' "$REGION" >&2
  exit 2
}
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$ && "$BUCKET" != *..* ]] || {
  printf 'Invalid Cloud Storage bucket name: %s\n' "$BUCKET" >&2
  exit 2
}
[[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{1,61}[a-z0-9])$ ]] || {
  printf 'Invalid run id: %s\n' "$RUN_ID" >&2
  exit 2
}
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] && ((MAX_IMAGES >= 100 && MAX_IMAGES <= 1000000)) || {
  printf '%s\n' '--max-images must be an integer from 100 to 1000000' >&2
  exit 2
}

if [[ "$EXECUTE" != true ]]; then
  printf '%s\n' 'Dry plan only; no cloud resources were changed.'
  printf 'project=%s region=%s bucket=gs://%s run_id=%s max_images=%s gpu=NVIDIA_L4\n' \
    "$PROJECT" "$REGION" "$BUCKET" "$RUN_ID" "$MAX_IMAGES"
  printf '%s\n' 'Re-run with --execute to create billable resources.'
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || {
  printf '%s\n' 'gcloud is required. Run this script in Google Cloud Shell.' >&2
  exit 1
}

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
[[ -n "$ACTIVE_ACCOUNT" ]] || {
  printf '%s\n' 'No active Google Cloud login. Run: gcloud auth login' >&2
  exit 1
}

case "$ACTIVE_ACCOUNT" in
  *.gserviceaccount.com) ACTIVE_MEMBER="serviceAccount:${ACTIVE_ACCOUNT}" ;;
  *) ACTIVE_MEMBER="user:${ACTIVE_ACCOUNT}" ;;
esac

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT"

if ! gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "$PROJECT" \
    --location "$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi
gcloud storage buckets update "gs://${BUCKET}" \
  --project "$PROJECT" \
  --uniform-bucket-level-access \
  --public-access-prevention

if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --project "$PROJECT" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --project "$PROJECT" \
    --location "$REGION" \
    --repository-format docker \
    --description "Private Furniture AI Vertex training images"
fi

SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" \
  --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --project "$PROJECT" \
    --display-name "Furniture AI managed trainer"
fi

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectAdmin >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.bucketViewer >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" \
  --project "$PROJECT" \
  --location "$REGION" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/artifactregistry.reader >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT" \
  --project "$PROJECT" \
  --member "$ACTIVE_MEMBER" \
  --role roles/iam.serviceAccountUser >/dev/null

BUILD_SERVICE_ACCOUNT="$(
  gcloud builds get-default-service-account --project "$PROJECT" --region "$REGION"
)"
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" \
  --project "$PROJECT" \
  --location "$REGION" \
  --member "serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
  --role roles/artifactregistry.writer >/dev/null

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/openimages-trainer:${RUN_ID}"
gcloud builds submit . \
  --project "$PROJECT" \
  --region "$REGION" \
  --config cloud/cloudbuild.vertex.yaml \
  --substitutions "_IMAGE_URI=${IMAGE_URI}"

JOB_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$JOB_CONFIG"' EXIT
sed \
  -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
  -e "s|__BUCKET__|${BUCKET}|g" \
  -e "s|__RUN_ID__|${RUN_ID}|g" \
  -e "s|__MAX_IMAGES__|${MAX_IMAGES}|g" \
  -e "s|__SERVICE_ACCOUNT__|${SERVICE_ACCOUNT}|g" \
  cloud/vertex_job.template.yaml >"$JOB_CONFIG"

gcloud ai custom-jobs create \
  --project "$PROJECT" \
  --region "$REGION" \
  --display-name "$RUN_ID" \
  --config "$JOB_CONFIG"

printf '\nSubmitted paid Vertex AI job %s.\n' "$RUN_ID"
printf 'Dataset/checkpoints: gs://%s/runs/%s/\n' "$BUCKET" "$RUN_ID"
printf 'Jobs: gcloud ai custom-jobs list --project %s --region %s\n' "$PROJECT" "$REGION"
printf 'Status: gcloud storage cat gs://%s/runs/%s/status.json\n' "$BUCKET" "$RUN_ID"
