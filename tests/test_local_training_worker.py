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


@pytest.mark.parametrize("argument", ["--output", "--output=/tmp/escape.pth", "--resume=x.pth"])
def test_load_config_blocks_worker_managed_path_overrides(
    tmp_path: Path, argument: str
) -> None:
    config = tmp_path / "jobs.json"
    write_config(
        config,
        [
            {
                "id": "style",
                "task": "style_classifier",
                "data": "data/styles_prepared",
                "output": "models/style.pth",
                "args": [argument],
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

    assert without_resume[1:3] == ["-m", "training.local_resumable_classifier"]
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


def test_sync_skips_when_checkout_is_not_configured_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *arguments: str):
        calls.append(arguments)
        return type("Result", (), {"returncode": 0, "stdout": "feature\n", "stderr": ""})()

    monkeypatch.setattr(local_worker, "git_command", fake_git)
    changed = local_worker.sync_from_github(
        tmp_path, {"enabled": True, "remote": "origin", "branch": "main"}
    )
    assert changed is False
    assert calls == [("branch", "--show-current")]


def test_disabled_sync_never_touches_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_git(*_args, **_kwargs):
        pytest.fail("disabled synchronization must not execute git")

    monkeypatch.setattr(local_worker, "git_command", forbidden_git)
    assert local_worker.sync_from_github(tmp_path, {"enabled": False}) is False


def test_committed_training_queue_disables_code_sync() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "training" / "local_training_jobs.json").read_text(encoding="utf-8")
    )
    assert payload["sync"]["enabled"] is False


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/install_local_trainer_windows.ps1", "FurnitureAI_GPU_Trainer_Setup.ps1"],
)
def test_windows_trainer_installers_use_restricted_service_account(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / relative_path).read_text(encoding="utf-8")
    assert '-UserId "SYSTEM"' not in text
    assert "-RunLevel Highest" not in text
    assert "S-1-5-20" in text
    assert "-RunLevel Limited" in text


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/install_local_trainer_windows.ps1", "FurnitureAI_GPU_Trainer_Setup.ps1"],
)
def test_windows_trainer_acl_keeps_code_read_only(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / relative_path).read_text(encoding="utf-8")
    assert '${WorkerAclSid}:(OI)(CI)RX' in text
    assert '${WorkerAclSid}:(OI)(CI)M' in text
    assert 'Join-Path $RepoRoot ".furnitureai-local"' in text
    assert 'Join-Path $RepoRoot "models"' in text
    assert 'Join-Path $RepoRoot "data"' in text


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
