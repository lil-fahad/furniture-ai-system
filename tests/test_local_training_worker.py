from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import local_worker


def write_config(path: Path, jobs: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sync": {"enabled": False},
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )


def test_dataset_identity_prefers_versioned_fingerprint(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "summary.json").write_text(
        json.dumps({"dataset_fingerprint": "abc123"}), encoding="utf-8"
    )
    (data / "image.jpg").write_bytes(b"not-an-image-but-identity-does-not-decode")
    assert local_worker.dataset_identity(data) == "dataset:abc123"


def test_load_config_rejects_non_training_task(tmp_path: Path) -> None:
    config = tmp_path / "jobs.json"
    write_config(
        config,
        [
            {
                "id": "unsafe",
                "task": "shell",
                "data": "data/x",
                "output": "models/x.pth",
                "args": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="unsupported training task"):
        local_worker.load_config(config)


def test_load_config_blocks_worker_managed_path_overrides(tmp_path: Path) -> None:
    config = tmp_path / "jobs.json"
    write_config(
        config,
        [
            {
                "id": "style",
                "task": "style_classifier",
                "data": "data/styles_prepared",
                "output": "models/style.pth",
                "args": ["--output", "/tmp/escape.pth"],
            }
        ],
    )
    with pytest.raises(ValueError, match="worker-managed paths"):
        local_worker.load_config(config)


def test_classifier_command_includes_resume_only_when_allowed(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / ".furnitureai-local" / "checkpoints").mkdir(parents=True)
    job = {
        "id": "style",
        "task": "style_classifier",
        "data": "data/styles_prepared",
        "output": "models/style.pth",
        "checkpoint_every_steps": 25,
        "args": ["--epochs", "3"],
    }
    resume = local_worker.resume_path_for(repo, job)
    resume.write_bytes(b"checkpoint")

    without_resume = local_worker.build_command(repo, job, resume_allowed=False)
    with_resume = local_worker.build_command(repo, job, resume_allowed=True)

    assert "--resume-output" in without_resume
    assert "--checkpoint-every-steps" in without_resume
    assert "--resume" not in without_resume
    assert "--resume" in with_resume
    assert str(resume) in with_resume


def test_segmenter_command_uses_dedicated_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path
    job = {
        "id": "plans",
        "task": "floorplan_segmenter",
        "data": "data/plans",
        "output": "models/segmenter.pt",
        "args": ["--epochs", "2"],
    }
    command = local_worker.build_command(repo, job, resume_allowed=False)
    checkpoint = local_worker.resume_path_for(repo, job)
    assert "--checkpoint" in command
    assert str(checkpoint) in command
    assert "--resume" not in command


def test_successful_matching_job_is_not_retrained(tmp_path: Path) -> None:
    repo = tmp_path
    trainer = repo / "training" / "local_resumable_classifier.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("print('trainer')\n", encoding="utf-8")
    data = repo / "data" / "styles_prepared"
    data.mkdir(parents=True)
    (data / "summary.json").write_text(
        json.dumps({"dataset_fingerprint": "same-data"}), encoding="utf-8"
    )
    output = repo / "models" / "style.pth"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"model")
    job = {
        "id": "style",
        "task": "style_classifier",
        "data": "data/styles_prepared",
        "output": "models/style.pth",
        "args": [],
    }
    fingerprint = local_worker.job_fingerprint(repo, job)
    state = {
        "version": 1,
        "jobs": {
            "style": {
                "status": "succeeded",
                "fingerprint": fingerprint,
            }
        },
    }
    result = local_worker.run_job(
        repo,
        job,
        state,
        repo / ".furnitureai-local" / "state.json",
        grace_seconds=1,
    )
    assert result == "already_current"
