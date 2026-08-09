#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="round-office-505007-q4"
REGION="us-central1"
SERVICE="furniture-ai-api"
MODEL="gpt-5-mini"
KEY_FILE="${HOME}/.env.local"
RUNTIME_SERVICE_ACCOUNT_NAME="furniture-api-runtime"
EXECUTE=false
KEEP_KEY_FILE=false

usage() {
  cat <<'USAGE'
Store an OpenAI API key in Google Secret Manager and deploy Furniture AI to
Cloud Run with GPT vision/design assistance enabled.

The key is never committed to GitHub or printed. On the first run, upload a
.env.local file containing OPENAI_API_KEY to Cloud Shell. After the secret is
stored successfully, the transfer file is deleted unless --keep-key-file is set.

Usage:
  bash cloud/deploy_gpt_cloud_run.sh --execute [options]

Options:
  --project ID          Google Cloud project
  --region REGION       Cloud Run region (default: us-central1)
  --service NAME        Cloud Run service name (default: furniture-ai-api)
  --model NAME          OpenAI model (default: gpt-5-mini)
  --key-file PATH       First-run env file (default: $HOME/.env.local)
  --keep-key-file       Do not delete the transfer file after import
  --execute             Create/update billable cloud resources
  -h, --help            Show this help
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
    --service)
      SERVICE="${2:?missing value for --service}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --key-file)
      KEY_FILE="${2:?missing value for --key-file}"
      shift 2
      ;;
    --keep-key-file)
      KEEP_KEY_FILE=true
      shift
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

[[ "$PROJECT" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || {
  printf 'Invalid Google Cloud project id: %s\n' "$PROJECT" >&2
  exit 2
}
[[ "$REGION" =~ ^[a-z0-9]+(-[a-z0-9]+)+$ ]] || {
  printf 'Invalid Google Cloud region: %s\n' "$REGION" >&2
  exit 2
}
[[ "$SERVICE" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || {
  printf 'Invalid Cloud Run service name: %s\n' "$SERVICE" >&2
  exit 2
}
[[ "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'Invalid OpenAI model name: %s\n' "$MODEL" >&2
  exit 2
}

if [[ "$EXECUTE" != true ]]; then
  printf '%s\n' 'Dry plan only; no cloud resources were changed.'
  printf 'project=%s region=%s service=%s model=%s\n' \
    "$PROJECT" "$REGION" "$SERVICE" "$MODEL"
  printf '%s\n' 'Re-run with --execute to store secrets and deploy billable resources.'
  exit 0
fi

for command_name in gcloud openssl python3 curl; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'Required command not found: %s\n' "$command_name" >&2
    exit 1
  }
done

ACTIVE_ACCOUNT="$(
  gcloud auth list --filter='status:ACTIVE' --format='value(account)' 2>/dev/null | head -n 1
)"
[[ -n "$ACTIVE_ACCOUNT" ]] || {
  printf '%s\n' 'No active Google Cloud login. Run: gcloud auth login' >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OPENAI_SECRET="openai-api-key"
SERVICE_SECRET="furniture-service-api-key"
RUNTIME_SERVICE_ACCOUNT="${RUNTIME_SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT"

secret_exists() {
  gcloud secrets describe "$1" --project "$PROJECT" >/dev/null 2>&1
}

latest_enabled_version() {
  local version
  version="$(
    gcloud secrets versions list "$1" \
      --project "$PROJECT" \
      --filter='state:ENABLED' \
      --sort-by='~createTime' \
      --limit=1 \
      --format='value(name)'
  )"
  version="${version##*/}"
  [[ "$version" =~ ^[0-9]+$ ]] || {
    printf 'Could not resolve an enabled version for secret %s\n' "$1" >&2
    return 1
  }
  printf '%s' "$version"
}

if [[ -e "$KEY_FILE" ]]; then
  [[ -f "$KEY_FILE" && ! -L "$KEY_FILE" ]] || {
    printf 'Key file must be a regular, non-symlink file: %s\n' "$KEY_FILE" >&2
    exit 1
  }
  chmod 600 "$KEY_FILE"

  OPENAI_KEY="$(python3 - "$KEY_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
for raw_line in path.read_text(encoding="utf-8").splitlines():
    name, separator, value = raw_line.partition("=")
    if separator and name.strip() == "OPENAI_API_KEY":
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            print(value, end="")
            break
else:
    raise SystemExit("OPENAI_API_KEY is missing from the key file")
PY
)"
  trap 'unset OPENAI_KEY' EXIT
  [[ ${#OPENAI_KEY} -ge 20 ]] || {
    printf '%s\n' 'OPENAI_API_KEY in the key file is not usable.' >&2
    exit 1
  }

  if ! secret_exists "$OPENAI_SECRET"; then
    gcloud secrets create "$OPENAI_SECRET" \
      --project "$PROJECT" \
      --replication-policy=automatic
  fi
  printf '%s' "$OPENAI_KEY" | gcloud secrets versions add "$OPENAI_SECRET" \
    --project "$PROJECT" \
    --data-file=- >/dev/null
  unset OPENAI_KEY
  trap - EXIT

  if [[ "$KEEP_KEY_FILE" != true ]]; then
    rm -f -- "$KEY_FILE"
    printf 'Imported the OpenAI key and removed transfer file: %s\n' "$KEY_FILE"
  fi
elif ! secret_exists "$OPENAI_SECRET"; then
  printf 'No key file at %s and secret %s does not exist.\n' \
    "$KEY_FILE" "$OPENAI_SECRET" >&2
  exit 1
else
  printf 'Reusing existing Secret Manager secret: %s\n' "$OPENAI_SECRET"
fi

if ! secret_exists "$SERVICE_SECRET"; then
  gcloud secrets create "$SERVICE_SECRET" \
    --project "$PROJECT" \
    --replication-policy=automatic
  openssl rand -hex 32 | tr -d '\n' | \
    gcloud secrets versions add "$SERVICE_SECRET" \
      --project "$PROJECT" \
      --data-file=- >/dev/null
fi

if ! gcloud iam service-accounts describe "$RUNTIME_SERVICE_ACCOUNT" \
  --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SERVICE_ACCOUNT_NAME" \
    --project "$PROJECT" \
    --display-name='Furniture AI API runtime'
fi

for secret_name in "$OPENAI_SECRET" "$SERVICE_SECRET"; do
  gcloud secrets add-iam-policy-binding "$secret_name" \
    --project "$PROJECT" \
    --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
    --role='roles/secretmanager.secretAccessor' >/dev/null
done

OPENAI_VERSION="$(latest_enabled_version "$OPENAI_SECRET")"
SERVICE_VERSION="$(latest_enabled_version "$SERVICE_SECRET")"

cd "$REPO_ROOT"
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --service-account "$RUNTIME_SERVICE_ACCOUNT" \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars="ENVIRONMENT=production,OPENAI_MODEL=${MODEL}" \
  --update-secrets="OPENAI_API_KEY=${OPENAI_SECRET}:${OPENAI_VERSION},SERVICE_API_KEY=${SERVICE_SECRET}:${SERVICE_VERSION}" \
  --quiet

SERVICE_URL="$(
  gcloud run services describe "$SERVICE" \
    --project "$PROJECT" \
    --region "$REGION" \
    --format='value(status.url)'
)"
[[ -n "$SERVICE_URL" ]] || {
  printf '%s\n' 'Cloud Run deployed, but no service URL was returned.' >&2
  exit 1
}

printf '\nGPT is connected to Furniture AI on Google Cloud.\n'
printf 'Service URL: %s\n' "$SERVICE_URL"
printf 'Health check: '
curl --fail --silent --show-error "${SERVICE_URL}/health"
printf '\n'
