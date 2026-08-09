#!/usr/bin/env bash
# =============================================================================
# gcp_bootstrap.sh — one-command GCP setup for furniture-ai-system cloud training
#
# Takes a fresh GCP project to submitted Vertex AI GPU training jobs:
#   1. verify gcloud authentication + project access
#   2. enable required APIs (aiplatform, storage, cloudbuild, artifactregistry)
#   3. create the Artifact Registry repo "furniture-ai"        (idempotent)
#   4. create the GCS bucket gs://furniture-ai-training-$PROJECT (idempotent)
#   5. build + push the training image via Cloud Build          (skippable)
#   6. stage datasets to gs://$BUCKET/datasets/ via
#      python -m training.data_ingest.stage_all                 (skippable)
#   7. submit Vertex AI custom jobs via
#      python -m cloud.vertex_jobs --task all                   (skippable)
#
# The script is safe to re-run: every creation step tolerates "already exists".
#
# Usage:
#   scripts/gcp_bootstrap.sh [--project P] [--region R] [--bucket B]
#                            [--skip-data] [--skip-image] [--skip-jobs]
#
# Defaults: --project from `gcloud config get-value project`,
#           --region us-central1,
#           --bucket furniture-ai-training-<project>.
# Run from the repository root (Cloud Shell: `git clone` then `cd` into it).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
PROJECT=""
REGION="us-central1"
BUCKET=""
SKIP_DATA=0
SKIP_IMAGE=0
SKIP_JOBS=0

usage() {
    cat <<'EOF'
Usage: gcp_bootstrap.sh [options]

Options:
  --project P    GCP project id (default: gcloud configured project)
  --region R     GCP region (default: us-central1)
  --bucket B     GCS staging bucket name, no gs:// (default: furniture-ai-training-<project>)
  --skip-data    Skip dataset staging (step 6)
  --skip-image   Skip container image build (step 5)
  --skip-jobs    Skip Vertex AI job submission (step 7)
  -h, --help     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="${2:?--project requires a value}"; shift 2 ;;
        --region) REGION="${2:?--region requires a value}"; shift 2 ;;
        --bucket) BUCKET="${2:?--bucket requires a value}"; shift 2 ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --skip-image) SKIP_IMAGE=1; shift ;;
        --skip-jobs) SKIP_JOBS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PROJECT" ]]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    echo "error: no GCP project; pass --project or run 'gcloud config set project P'" >&2
    exit 2
fi
if [[ -z "$BUCKET" ]]; then
    BUCKET="furniture-ai-training-${PROJECT}"
fi

AR_REPO="furniture-ai"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/trainer:latest"

echo "==> configuration"
echo "    project : ${PROJECT}"
echo "    region  : ${REGION}"
echo "    bucket  : gs://${BUCKET}"
echo "    image   : ${IMAGE_URI}"

# ---------------------------------------------------------------------------
# Step 1: authentication + project access check
# ---------------------------------------------------------------------------
echo "==> step 1/7: checking gcloud authentication and project access"
# Cloud Shell authenticates implicitly and often reports NO active account in
# `gcloud auth list`, so validate access with a real API call instead of
# requiring a listed account (do NOT run 'gcloud auth login' in Cloud Shell).
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if ! gcloud projects describe "${PROJECT}" >/dev/null 2>&1; then
    echo "error: cannot access project ${PROJECT}" >&2
    echo "       In Cloud Shell you are already authenticated - check the project id." >&2
    echo "       Outside Cloud Shell run: gcloud auth login" >&2
    exit 1
fi
echo "    project access OK${ACTIVE_ACCOUNT:+ (account: ${ACTIVE_ACCOUNT})}"

# ---------------------------------------------------------------------------
# Step 2: enable required APIs (idempotent)
# ---------------------------------------------------------------------------
echo "==> step 2/7: enabling required APIs"
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project "${PROJECT}"

# ---------------------------------------------------------------------------
# Step 3: Artifact Registry repo (idempotent)
# ---------------------------------------------------------------------------
echo "==> step 3/7: ensuring Artifact Registry repo '${AR_REPO}'"
if gcloud artifacts repositories describe "${AR_REPO}" \
    --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
    echo "    repo already exists"
else
    gcloud artifacts repositories create "${AR_REPO}" \
        --repository-format docker \
        --location "${REGION}" \
        --project "${PROJECT}" \
        --description "furniture-ai-system training images"
fi

# ---------------------------------------------------------------------------
# Step 4: GCS bucket (idempotent)
# ---------------------------------------------------------------------------
echo "==> step 4/7: ensuring bucket gs://${BUCKET}"
if gcloud storage ls "gs://${BUCKET}" >/dev/null 2>&1; then
    echo "    bucket already exists"
else
    gcloud storage buckets create "gs://${BUCKET}" \
        --project "${PROJECT}" \
        --location "${REGION}" \
        --uniform-bucket-level-access
fi

# ---------------------------------------------------------------------------
# Step 5: build + push training image via Cloud Build
# ---------------------------------------------------------------------------
if [[ "${SKIP_IMAGE}" -eq 1 ]]; then
    echo "==> step 5/7: SKIPPED image build (--skip-image)"
else
    echo "==> step 5/7: building training image with Cloud Build"
    gcloud builds submit \
        --project "${PROJECT}" \
        --tag "${IMAGE_URI}" \
        -f cloud/Dockerfile.training \
        .
fi

# ---------------------------------------------------------------------------
# Step 6: stage datasets to GCS
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATA}" -eq 1 ]]; then
    echo "==> step 6/7: SKIPPED dataset staging (--skip-data)"
else
    echo "==> step 6/7: staging datasets to gs://${BUCKET}/datasets/"
    python -m training.data_ingest.stage_all --bucket "${BUCKET}"
fi

# ---------------------------------------------------------------------------
# Step 7: submit Vertex AI custom training jobs
# ---------------------------------------------------------------------------
if [[ "${SKIP_JOBS}" -eq 1 ]]; then
    echo "==> step 7/7: SKIPPED job submission (--skip-jobs)"
else
    echo "==> step 7/7: submitting Vertex AI custom jobs (room, segmenter, ranker)"
    python -m cloud.vertex_jobs --task all \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --bucket "${BUCKET}" \
        --image-uri "${IMAGE_URI}"
fi

echo "==> done"
echo "    monitor jobs : gcloud ai custom-jobs list --region ${REGION} --project ${PROJECT}"
echo "    fetch results: gcloud storage cp -r gs://${BUCKET}/runs ."
