#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_NAME="furnitureai-local-trainer"
VENV="${REPO_ROOT}/.trainer-venv"
PYTHON="${VENV}/bin/python"
RUN_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  printf '%s\n' 'Run this installer once with sudo.' >&2
  exit 1
fi

if ! id "${RUN_USER}" >/dev/null 2>&1; then
  printf 'Cannot resolve service user: %s\n' "${RUN_USER}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  sudo -u "${RUN_USER}" python3 -m venv "${VENV}"
fi

sudo -u "${RUN_USER}" "${PYTHON}" -m pip install --upgrade pip
sudo -u "${RUN_USER}" "${PYTHON}" -m pip install -e "${REPO_ROOT}[training]"

cat >"/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=FurnitureAI autonomous local model training worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
ExecStart=${PYTHON} -m training.local_worker --repo ${REPO_ROOT}
Restart=always
RestartSec=15
TimeoutStopSec=60
KillSignal=SIGTERM
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}.service"

printf 'Installed and started %s.service\n' "${SERVICE_NAME}"
printf 'Worker state: %s/.furnitureai-local/state.json\n' "${REPO_ROOT}"
printf 'Training logs: %s/.furnitureai-local/logs/\n' "${REPO_ROOT}"
printf 'It will start automatically on every Linux boot.\n'
