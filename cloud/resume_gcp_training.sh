#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="round-office-505007-q4"
REGION="us-central1"
BUCKET=""
BUILD_ID=""
MAX_IMAGES="100000"
POLL_SECONDS="30"
EXECUTE=false
REPOSITORY="furniture-ai-training"
SERVICE_ACCOUNT_NAME="furniture-ml"

usage() {
  cat <<'USAGE'
Wait for an existing Cloud Build without exhausting the status quota, then
submit its image as a paid Vertex AI job using one NVIDIA L4 GPU.

Usage:
  bash cloud/resume_gcp_training.sh --build-id ID --execute [options]

Options:
  --build-id ID        Existing Cloud Build id (required)
  --project ID         Google Cloud project (default: round-office-505007-q4)
  --region REGION      Cloud Build and Vertex region (default: us-central1)
  --bucket NAME        Private bucket (default: furniture-ai-PROJECT)
  --max-images N       Target accepted images (default: 100000)
  --poll-seconds N     Seconds between build checks (default: 30)
  --execute            Acknowledge that a billable Vertex GPU job may start
  -h, --help           Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --build-id)
      BUILD_ID="${2:?missing value for --build-id}"
      shift 2
      ;;
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
    --max-images)
      MAX_IMAGES="${2:?missing value for --max-images}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:?missing value for --poll-seconds}"
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

[[ -n "$BUILD_ID" && "$BUILD_ID" =~ ^[A-Za-z0-9-]+$ ]] || {
  printf '%s\n' '--build-id is required and must contain only letters, numbers, or hyphens' >&2
  exit 2
}
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
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] && ((MAX_IMAGES >= 100 && MAX_IMAGES <= 1000000)) || {
  printf '%s\n' '--max-images must be an integer from 100 to 1000000' >&2
  exit 2
}
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] && ((POLL_SECONDS >= 10 && POLL_SECONDS <= 300)) || {
  printf '%s\n' '--poll-seconds must be an integer from 10 to 300' >&2
  exit 2
}

if [[ "$EXECUTE" != true ]]; then
  printf '%s\n' 'Dry plan only; no Vertex AI job was submitted.'
  printf 'project=%s region=%s build_id=%s bucket=gs://%s max_images=%s poll_seconds=%s\n' \
    "$PROJECT" "$REGION" "$BUILD_ID" "$BUCKET" "$MAX_IMAGES" "$POLL_SECONDS"
  printf '%s\n' 'Re-run with --execute to permit a paid NVIDIA L4 job.'
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || {
  printf '%s\n' 'gcloud is required. Run this script in Google Cloud Shell.' >&2
  exit 1
}

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
[[ -n "$ACTIVE_ACCOUNT" ]] || {
  printf '%s\n' 'No active Google Cloud login.' >&2
  exit 1
}

gcloud config set project "$PROJECT" >/dev/null

printf 'Watching Cloud Build %s every %s seconds.\n' "$BUILD_ID" "$POLL_SECONDS"
while true; do
  if ! STATUS="$(
    gcloud builds describe "$BUILD_ID" \
      --project "$PROJECT" \
      --region "$REGION" \
      --format='value(status)'
  )"; then
    printf 'Status check was rate-limited or interrupted; retrying in %s seconds.\n' "$POLL_SECONDS" >&2
    sleep "$POLL_SECONDS"
    continue
  fi

  printf 'Cloud Build status: %s\n' "$STATUS"
  case "$STATUS" in
    SUCCESS)
      break
      ;;
    QUEUED|WORKING|PENDING|STATUS_UNKNOWN)
      sleep "$POLL_SECONDS"
      ;;
    FAILURE|INTERNAL_ERROR|TIMEOUT|CANCELLED|EXPIRED)
      printf 'Cloud Build ended with status %s; Vertex training was not submitted.\n' "$STATUS" >&2
      exit 1
      ;;
    *)
      printf 'Unexpected Cloud Build status %s; retrying in %s seconds.\n' "$STATUS" "$POLL_SECONDS" >&2
      sleep "$POLL_SECONDS"
      ;;
  esac
done

IMAGE_URI="$(
  gcloud builds describe "$BUILD_ID" \
    --project "$PROJECT" \
    --region "$REGION" \
    --format='value(substitutions._IMAGE_URI)'
)"
EXPECTED_PREFIX="${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/openimages-trainer:"
[[ "$IMAGE_URI" == "${EXPECTED_PREFIX}"* ]] || {
  printf 'Build image %s is not a Furniture AI training image for this project.\n' "$IMAGE_URI" >&2
  exit 1
}

RUN_ID="${IMAGE_URI##*:}"
[[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{1,61}[a-z0-9])$ ]] || {
  printf 'Invalid run id recovered from image: %s\n' "$RUN_ID" >&2
  exit 1
}

EXISTING_JOB="$(
  gcloud ai custom-jobs list \
    --project "$PROJECT" \
    --region "$REGION" \
    --filter="displayName=${RUN_ID}" \
    --limit=1 \
    --format='value(name)' 2>/dev/null || true
)"
if [[ -n "$EXISTING_JOB" ]]; then
  printf 'Vertex AI job already exists: %s\n' "$EXISTING_JOB"
  printf 'No duplicate paid GPU job was submitted.\n'
  exit 0
fi

SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"
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
