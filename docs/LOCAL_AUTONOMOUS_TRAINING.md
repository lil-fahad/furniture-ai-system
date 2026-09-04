# Autonomous local model training

FurnitureAI can dedicate one local computer to model training only. After a one-time installation, the worker starts automatically with the operating system, reads the training queue from GitHub, trains only allow-listed FurnitureAI trainers, writes local checkpoints, and resumes interrupted work after reboot.

## Behavior

- The worker starts at Windows boot or as a Linux systemd service.
- It fast-forwards the local repository from `origin/main` only between training jobs.
- It never interrupts a running job just to pull new code.
- It runs only the allow-listed model trainers defined in `training/local_worker.py`.
- Runtime state, logs, caches, and resume checkpoints live under `.furnitureai-local/` and are ignored by Git.
- Model binaries remain local under `models/` and are ignored by Git.
- Missing datasets do not crash the worker. The matching job becomes `blocked_missing_data` and is retried automatically later.
- A completed job is not retrained until its job configuration, trainer source, or dataset identity changes.

## Shutdown and reboot

The local classifier trainer writes a full resume checkpoint every configurable number of optimizer steps. The checkpoint includes model weights, optimizer state, scheduler state, mixed-precision scaler state, epoch, batch position, best validation metric, and dataset identity.

When the operating system delivers a termination signal, the worker asks the trainer to write an immediate pause checkpoint and gives it a grace period before termination. On the next boot, the same job is started with that checkpoint and continues from the saved epoch/batch.

Unexpected power loss cannot be made transactional by software. The periodic checkpoint limits lost work to the interval since the most recent checkpoint. The default classifier interval is 100 optimizer steps and can be lowered in `training/local_training_jobs.json`.

The existing floor-plan segmenter already stores model, optimizer, scheduler, epoch, and best-model state after each epoch. The worker automatically passes its local checkpoint back through `--resume` after an interrupted run.

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

The worker deliberately does not expose an arbitrary shell-command queue. A GitHub job may select only a supported trainer and its trainer arguments; output and resume paths remain controlled by the worker.

## Windows one-time installation

Open PowerShell as Administrator in the repository and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_local_trainer_windows.ps1
```

The installer creates `.trainer-venv`, installs the training dependencies, registers `FurnitureAI-LocalTrainer` in Task Scheduler as a SYSTEM startup task, and starts it immediately. After that, no terminal command is needed after normal reboots.

State:

```text
.furnitureai-local/state.json
```

Logs:

```text
.furnitureai-local/logs/
```

## Linux one-time installation

From the repository:

```bash
sudo bash scripts/install_local_trainer_linux.sh
```

This creates the training virtual environment and enables `furnitureai-local-trainer.service` at boot.

Useful inspection commands:

```bash
systemctl status furnitureai-local-trainer
journalctl -u furnitureai-local-trainer -f
```

## GitHub synchronization

By default the worker checks `origin/main` every five minutes while idle/between jobs. It uses a fast-forward-only merge. If tracked local source files were edited on the training computer, automatic Git synchronization is skipped instead of overwriting those edits.

Dataset and model files are intentionally not pulled from or pushed to GitHub by the worker. GitHub controls code and the training queue; large training data/checkpoints remain on the training computer unless a separate storage workflow is configured.
