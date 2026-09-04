# Autonomous local model training

FurnitureAI can dedicate one local computer to model training only. After a one-time installation, the worker starts automatically with the operating system, trains only allow-listed FurnitureAI trainers, writes local checkpoints, and resumes interrupted work after reboot.

## Security boundary

The persistent training worker is intentionally **not** a software-update agent.

- Automatic GitHub code synchronization is disabled in `training/local_training_jobs.json`.
- A remote change to `main` is therefore not fetched and executed by an already-running worker by default.
- Updating code is an explicit local administrative action: rerun the setup/update process or manually review and fast-forward the checkout.
- Windows boot tasks use the restricted Network Service identity (well-known SID `S-1-5-20`) with `RunLevel Limited`; they must not run as SYSTEM or Highest.
- The one-time Windows installer still requires Administrator rights to install dependencies, set ACLs, and register the task. Those installation privileges are not inherited by the persistent worker.
- The worker continues to reject arbitrary shell tasks. Only the allow-listed training task types in `training/local_worker.py` are accepted.

This separation prevents a repository write or merge by itself from immediately becoming privileged code execution on an idle Windows training computer.

## Behavior

- The worker starts at Windows boot or as a Linux systemd service.
- It does not pull new repository code automatically with the committed configuration.
- It never interrupts a running job for code updates.
- It runs only the allow-listed model trainers defined in `training/local_worker.py`.
- Runtime state, logs, caches, and resume checkpoints live under `.furnitureai-local/` and are ignored by Git.
- Model binaries remain local under `models/` and are ignored by Git.
- Missing datasets do not crash the worker. The matching job becomes `blocked_missing_data` and is retried automatically later.
- A completed job is not retrained until its job configuration, trainer source, or dataset identity changes.

## Shutdown and reboot

The local classifier trainer writes a full resume checkpoint every configurable number of optimizer steps. The checkpoint includes model weights, optimizer state, scheduler state, mixed-precision scaler state, epoch, batch position, best validation metric, and dataset identity.

When the operating system delivers a termination signal, the worker asks the trainer to write an immediate pause checkpoint and gives it a grace period before termination. On the next boot, the same job is started with that checkpoint and continues from the saved epoch/batch.

Unexpected power loss cannot be made transactional by software. The periodic checkpoint limits lost work to the interval since the most recent checkpoint. The default classifier interval is 100 optimizer steps and can be lowered in `training/local_training_jobs.json`.

The existing floor-plan segmenter stores model, optimizer, scheduler, epoch, and best-model state after each epoch. The worker automatically passes its local checkpoint back through `--resume` after an interrupted run.

## Training queue

The committed queue is `training/local_training_jobs.json`.

Supported tasks:

- `style_classifier`
- `room_classifier`
- `floorplan_segmenter`

All jobs are enabled by default. A job starts only when its configured local dataset exists. The default paths are:

- style classifier: `data/styles_prepared`
- room classifier: `data/rooms`
- floor-plan segmenter: `data/plans`

Training output defaults to the corresponding directories below `models/`.

The worker deliberately does not expose an arbitrary shell-command queue. A job may select only a supported trainer and its trainer arguments; output and resume paths remain controlled by the worker.

## Windows one-time installation

Open PowerShell as Administrator in the repository and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_local_trainer_windows.ps1
```

The installer creates `.trainer-venv`, installs the training dependencies, grants the restricted Network Service account access to the required repository tree, registers `FurnitureAI-LocalTrainer` as a limited startup task, and starts it.

The standalone `FurnitureAI_GPU_Trainer_Setup.ps1` performs the same persistent-account hardening while also provisioning the dedicated checkout and checking the NVIDIA/PyTorch installation.

State:

```text
.furnitureai-local/state.json
```

Logs:

```text
.furnitureai-local/logs/
```

### Windows GPU service-session note

GPU access from Windows service identities depends on the local NVIDIA/Windows environment. The setup verifies CUDA for the installation account, but that does not prove every driver exposes the GPU identically to Network Service. If the restricted startup task cannot access CUDA, do **not** switch it to SYSTEM/Highest. Use a dedicated restricted training account configured locally, or use the Linux systemd deployment, and verify CUDA from that execution context.

## Linux one-time installation

From the repository:

```bash
sudo bash scripts/install_local_trainer_linux.sh
```

This creates the training virtual environment and enables `furnitureai-local-trainer.service` at boot. The service runs as the configured non-root user rather than root.

Useful inspection commands:

```bash
systemctl status furnitureai-local-trainer
journalctl -u furnitureai-local-trainer -f
```

## Updating training code

The committed queue contains:

```json
"sync": {
  "enabled": false
}
```

Keep this disabled for unattended workers. To apply reviewed repository changes, perform an explicit local update while no training job is running, then restart the worker. The one-click Windows setup treats rerunning the setup as that explicit approval point and uses fast-forward-only Git operations.

The legacy `sync_from_github` helper remains for compatibility with installations that deliberately maintain a local opt-in configuration, but it is not enabled by the repository's unattended default. Enabling it reintroduces a larger supply-chain trust surface and should not be done for a privileged or unattended host.

Dataset and model files are intentionally not pulled from or pushed to GitHub by the worker. GitHub stores code and the training queue; large training data/checkpoints remain on the training computer unless a separate storage workflow is configured.
