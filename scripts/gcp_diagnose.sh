#!/usr/bin/env bash
# =============================================================================
# gcp_diagnose.sh — read-only GCP/Cloud Shell diagnostic for furniture-ai-system
#
# Collects the information needed to debug why gcp_bootstrap.sh step 1/7
# (authentication / project access) fails, especially from Google Cloud Shell.
#
# This script is READ-ONLY: it never modifies gcloud config, auth state, or
# any GCP resource. It never prints access tokens (only OK/FAIL).
#
# Usage:
#   scripts/gcp_diagnose.sh [PROJECT_ID]
# If PROJECT_ID is omitted, the gcloud configured project is used.
# Every section degrades gracefully when gcloud/curl are unavailable, so the
# output is always safe to paste into a bug report.
# =============================================================================

set -uo pipefail

section() {
    echo
    echo "==> $1"
}

note_unavailable() {
    echo "    not available: $1"
}

# --- resolve project ---------------------------------------------------------
PROJECT="${1:-}"
if [[ -z "${PROJECT}" ]]; then
    if command -v gcloud >/dev/null 2>&1; then
        PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
    fi
fi
if [[ "${PROJECT}" == "(unset)" ]]; then
    PROJECT=""
fi

echo "gcp_diagnose.sh — read-only diagnostic (no changes will be made)"
echo "date    : $(date -u 2>/dev/null || echo 'not available')"
echo "host    : $(hostname 2>/dev/null || echo 'not available')"
echo "project : ${PROJECT:-<none resolved>}"

# --- a) gcloud config get-value project --------------------------------------
section "a) gcloud config get-value project"
if command -v gcloud >/dev/null 2>&1; then
    gcloud config get-value project 2>&1 || note_unavailable "command failed"
else
    note_unavailable "gcloud CLI not installed"
fi

# --- b) gcloud auth list (all accounts) --------------------------------------
section "b) gcloud auth list"
if command -v gcloud >/dev/null 2>&1; then
    gcloud auth list 2>&1 || note_unavailable "command failed"
else
    note_unavailable "gcloud CLI not installed"
fi

# --- c) gcloud projects describe <project> (full error output) ---------------
section "c) gcloud projects describe ${PROJECT:-<none>}"
if ! command -v gcloud >/dev/null 2>&1; then
    note_unavailable "gcloud CLI not installed"
elif [[ -z "${PROJECT}" ]]; then
    note_unavailable "no project id (pass one as argument or set gcloud config)"
else
    gcloud projects describe "${PROJECT}" 2>&1 || true
fi

# --- d) metadata server identity (Cloud Shell service account) ---------------
section "d) metadata server identity"
if command -v curl >/dev/null 2>&1; then
    META_EMAIL="$(curl -s --max-time 3 \
        -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email \
        2>/dev/null || true)"
    if [[ -n "${META_EMAIL}" ]]; then
        echo "    ${META_EMAIL}"
    else
        note_unavailable "metadata server unreachable (expected outside Cloud Shell/GCE)"
    fi
else
    note_unavailable "curl not installed"
fi

# --- e) token probe: gcloud auth print-access-token (OK/FAIL only) -----------
section "e) access token probe (token value is never printed)"
if command -v gcloud >/dev/null 2>&1; then
    if gcloud auth print-access-token >/dev/null 2>&1; then
        echo "    OK - an access token could be minted"
    else
        echo "    FAIL - no access token (auth is broken or expired)"
    fi
else
    note_unavailable "gcloud CLI not installed"
fi

# --- f) disk usage ------------------------------------------------------------
section "f) disk usage"
if command -v df >/dev/null 2>&1; then
    df -h ~ 2>&1 || note_unavailable "df failed"
else
    note_unavailable "df not installed"
fi
if command -v du >/dev/null 2>&1; then
    du -h --max-depth=1 ~ 2>/dev/null | sort -rh | head -8 \
        || note_unavailable "du failed"
else
    note_unavailable "du not installed"
fi

echo
echo "==> diagnosis complete - paste this output"
