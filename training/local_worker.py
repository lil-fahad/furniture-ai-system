from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

PAUSED_EXIT_CODE = 75
RESTART_EXIT_CODE = 76
STATE_DIR_NAME = ".furnitureai-local"
DEFAULT_CONFIG = Path("training/local_training_jobs.json")

TASKS: dict[str, dict[str, object]] = {
    "style_classifier": {
        "kind": "classifier",
        "mode": "style",
        "trainer": Path("training/local_resumable_classifier.py"),
    },
    "room_classifier": {
        "kind": "classifier",
        "mode": "room",
        "trainer": Path("training/local_resumable_classifier.py"),
    },
    "floorplan_segmenter": {
        "kind": "segmenter",
        "trainer": Path("training/train_floorplan_segmenter.py"),
    },
}

_STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def _request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_signal_handlers() -> None:
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, _request_stop)


def atomic_json_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_identity(path: Path) -> str:
    summary = path / "summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("dataset_fingerprint"), str):
            return f"dataset:{payload['dataset_fingerprint']}"
    manifest = path / "manifest.jsonl"
    if manifest.is_file():
        return f"manifest:{sha256_file(manifest)}"

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest.update(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return f"tree:{digest.hexdigest()}"


def dataset_identity(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return f"file:{sha256_file(path)}"
    return directory_identity(path)


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("local training config must be a version 1 JSON object")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("local training config must contain a non-empty jobs list")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each local training job must be a JSON object")
        job_id = job.get("id")
        task = job.get("task")
        if not isinstance(job_id, str) or not job_id or job_id in seen:
            raise ValueError("every local training job id must be unique and non-empty")
        seen.add(job_id)
        if task not in TASKS:
            raise ValueError(f"job {job_id!r} uses unsupported training task {task!r}")
        if not isinstance(job.get("data"), str) or not isinstance(job.get("output"), str):
            raise ValueError(f"job {job_id!r} requires string data and output paths")
        arguments = job.get("args", [])
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise ValueError(f"job {job_id!r} args must be a list of strings")
        forbidden = ("--output", "--resume", "--resume-output", "--checkpoint")
        if any(
            item == option or item.startswith(option + "=")
            for item in arguments
            for option in forbidden
        ):
            raise ValueError(f"job {job_id!r} args may not override worker-managed paths")
    return payload


def resolve_repo_path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (repo / candidate).resolve()


def job_fingerprint(repo: Path, job: dict[str, object]) -> str:
    task = str(job["task"])
    trainer = repo / Path(TASKS[task]["trainer"])
    data = resolve_repo_path(repo, str(job["data"]))
    normalized = json.dumps(job, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(normalized.encode())
    digest.update(dataset_identity(data).encode())
    digest.update(sha256_file(trainer).encode())
    return digest.hexdigest()


def resume_path_for(repo: Path, job: dict[str, object]) -> Path:
    state_dir = repo / STATE_DIR_NAME / "checkpoints"
    return state_dir / f"{job['id']}.pth"


def build_command(repo: Path, job: dict[str, object], resume_allowed: bool) -> list[str]:
    task = str(job["task"])
    task_info = TASKS[task]
    data = resolve_repo_path(repo, str(job["data"]))
    output = resolve_repo_path(repo, str(job["output"]))
    resume = resume_path_for(repo, job)
    arguments = [str(item) for item in job.get("args", [])]
    python = sys.executable

    if task_info["kind"] == "classifier":
        command = [
            python,
            "-m",
            "training.local_resumable_classifier",
            str(data),
            "--mode",
            str(task_info["mode"]),
            "--output",
            str(output),
            "--resume-output",
            str(resume),
        ]
        if resume_allowed and resume.is_file():
            command.extend(["--resume", str(resume)])
        checkpoint_steps = int(job.get("checkpoint_every_steps", 100))
        command.extend(["--checkpoint-every-steps", str(checkpoint_steps), *arguments])
        return command

    command = [
        python,
        str(repo / Path(task_info["trainer"])),
        str(data),
        "--output",
        str(output),
        "--checkpoint",
        str(resume),
    ]
    if resume_allowed and resume.is_file():
        command.extend(["--resume", str(resume)])
    command.extend(arguments)
    return command


def load_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"version": 1, "jobs": {}, "updated_at": utc_now()}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "jobs": {}, "updated_at": utc_now()}
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), dict):
        return {"version": 1, "jobs": {}, "updated_at": utc_now()}
    return payload


def update_job_state(
    state: dict[str, object],
    path: Path,
    job_id: str,
    **values: object,
) -> None:
    jobs = state.setdefault("jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError("worker state jobs field is invalid")
    current = jobs.setdefault(job_id, {})
    if not isinstance(current, dict):
        current = {}
        jobs[job_id] = current
    current.update(values)
    current["updated_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json_save(state, path)


def git_command(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={repo}",
        "-C",
        str(repo),
        *arguments,
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def sync_from_github(repo: Path, sync: dict[str, object]) -> bool:
    if not bool(sync.get("enabled", True)):
        return False
    remote = str(sync.get("remote", "origin"))
    branch = str(sync.get("branch", "main"))
    current_branch = git_command(repo, "branch", "--show-current")
    if current_branch.returncode != 0:
        log(f"git branch unavailable; continuing current checkout: {current_branch.stderr.strip()}")
        return False
    if current_branch.stdout.strip() != branch:
        log(
            f"current branch is {current_branch.stdout.strip()!r}; "
            f"automatic GitHub update requires {branch!r}"
        )
        return False
    status = git_command(repo, "status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0:
        log(f"git status unavailable; continuing current checkout: {status.stderr.strip()}")
        return False
    if status.stdout.strip():
        log("tracked working tree has local changes; skipping automatic GitHub update")
        return False
    before = git_command(repo, "rev-parse", "HEAD")
    fetched = git_command(repo, "fetch", "--quiet", remote, branch)
    if fetched.returncode != 0:
        log(f"GitHub fetch failed; retrying later: {fetched.stderr.strip()}")
        return False
    merged = git_command(repo, "merge", "--ff-only", "FETCH_HEAD")
    if merged.returncode != 0:
        log(f"GitHub fast-forward skipped: {merged.stderr.strip()}")
        return False
    after = git_command(repo, "rev-parse", "HEAD")
    changed = before.stdout.strip() != after.stdout.strip()
    if changed:
        log(f"updated repository from GitHub to {after.stdout.strip()[:12]}")
    return changed


@contextmanager
def single_instance_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another FurnitureAI local trainer is already running") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another FurnitureAI local trainer is already running") from exc
        yield
    finally:
        handle.close()


def graceful_child_stop(process: subprocess.Popen[str], grace_seconds: int) -> None:
    if process.poll() is not None:
        return
    log(f"shutdown requested; asking trainer pid={process.pid} to write a pause checkpoint")
    try:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        log("trainer did not stop inside the grace window; forcing termination")
        process.kill()


def run_job(
    repo: Path,
    job: dict[str, object],
    state: dict[str, object],
    state_path: Path,
    grace_seconds: int,
) -> str:
    job_id = str(job["id"])
    data = resolve_repo_path(repo, str(job["data"]))
    output = resolve_repo_path(repo, str(job["output"]))
    fingerprint = job_fingerprint(repo, job)
    jobs_state = state.get("jobs", {})
    previous = jobs_state.get(job_id, {}) if isinstance(jobs_state, dict) else {}
    previous_fingerprint = previous.get("fingerprint") if isinstance(previous, dict) else None
    resume_allowed = previous_fingerprint == fingerprint

    if (
        isinstance(previous, dict)
        and previous.get("status") == "succeeded"
        and previous_fingerprint == fingerprint
        and output.is_file()
    ):
        return "already_current"
    if not data.exists():
        update_job_state(
            state,
            state_path,
            job_id,
            status="blocked_missing_data",
            fingerprint=fingerprint,
            data=str(data),
        )
        return "blocked_missing_data"

    resume = resume_path_for(repo, job)
    command = build_command(repo, job, resume_allowed)
    logs_dir = repo / STATE_DIR_NAME / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job_id}.log"
    update_job_state(
        state,
        state_path,
        job_id,
        status="running",
        fingerprint=fingerprint,
        data=str(data),
        output=str(output),
        resume=str(resume),
        started_at=utc_now(),
    )
    log(f"starting training job={job_id} task={job['task']}")

    creationflags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    environment = os.environ.copy()
    cache_root = repo / STATE_DIR_NAME / "cache"
    environment.setdefault("HF_HOME", str(cache_root / "huggingface"))
    environment.setdefault("TORCH_HOME", str(cache_root / "torch"))
    environment["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND {command!r}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        while process.poll() is None:
            if _STOP_REQUESTED:
                graceful_child_stop(process, grace_seconds)
                break
            time.sleep(1.0)
        return_code = process.wait()

    if _STOP_REQUESTED or return_code == PAUSED_EXIT_CODE:
        update_job_state(
            state,
            state_path,
            job_id,
            status="paused",
            fingerprint=fingerprint,
            exit_code=return_code,
        )
        log(f"training paused job={job_id}; resume checkpoint={resume}")
        return "paused"
    if return_code == 0 and output.is_file():
        update_job_state(
            state,
            state_path,
            job_id,
            status="succeeded",
            fingerprint=fingerprint,
            exit_code=0,
            completed_at=utc_now(),
        )
        log(f"training completed job={job_id} output={output}")
        return "succeeded"

    update_job_state(
        state,
        state_path,
        job_id,
        status="failed",
        fingerprint=fingerprint,
        exit_code=return_code,
        log=str(log_path),
    )
    log(f"training failed job={job_id} exit_code={return_code}; retrying on next cycle")
    return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--once", action="store_true", help="run one scheduler cycle and exit")
    return parser.parse_args()


def main() -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    args = parse_args()
    install_signal_handlers()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    state_dir = repo / STATE_DIR_NAME
    state_path = state_dir / "state.json"
    lock_path = state_dir / "worker.lock"

    with single_instance_lock(lock_path):
        log(f"FurnitureAI local training worker started repo={repo}")
        last_sync = 0.0
        while not _STOP_REQUESTED:
            config = load_config(config_path)
            sync = config.get("sync", {})
            if not isinstance(sync, dict):
                raise ValueError("sync config must be an object")
            sync_interval = max(60, int(sync.get("interval_seconds", 300)))
            if time.monotonic() - last_sync >= sync_interval:
                if sync_from_github(repo, sync):
                    log("new training code pulled from GitHub; restarting worker")
                    return RESTART_EXIT_CODE
                last_sync = time.monotonic()
                config = load_config(config_path)

            state = load_state(state_path)
            jobs = config["jobs"]
            assert isinstance(jobs, list)
            did_work = False
            for item in jobs:
                assert isinstance(item, dict)
                if _STOP_REQUESTED:
                    break
                if not bool(item.get("enabled", True)):
                    continue
                result = run_job(
                    repo,
                    item,
                    state,
                    state_path,
                    int(config.get("shutdown_grace_seconds", 45)),
                )
                if result in {"succeeded", "failed", "paused"}:
                    did_work = True
                if result == "paused":
                    break

            if args.once or _STOP_REQUESTED:
                break
            sleep_seconds = int(config.get("idle_sleep_seconds", 30))
            if did_work:
                sleep_seconds = min(sleep_seconds, 10)
            deadline = time.monotonic() + max(5, sleep_seconds)
            while not _STOP_REQUESTED and time.monotonic() < deadline:
                time.sleep(1.0)

        log("FurnitureAI local training worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
