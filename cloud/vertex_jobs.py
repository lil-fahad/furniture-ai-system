"""Submit Vertex AI custom training jobs for furniture-ai-system.

Stdlib-first module: ``google.cloud.aiplatform`` is imported lazily inside
functions so this module imports cleanly in environments without the Google
Cloud SDK installed (e.g. local CI running ``pytest``).

Usage:
    python -m cloud.vertex_jobs --task all \
        --project my-project --region us-central1 --bucket my-bucket
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASKS: tuple[str, ...] = ("room", "segmenter", "ranker")

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Fallback defaults mirroring cloud/config.yaml (used when the file is absent
# or cannot be parsed).
DEFAULT_CONFIG: dict[str, Any] = {
    "project": "your-gcp-project-id",
    "region": "us-central1",
    "bucket": "",
    "image_uri": "",
    "machine": "g2-standard-4",
    "accelerator": "NVIDIA_L4",
    "replica_count": 1,
    "spot": True,
    "epochs": {"room": 15, "segmenter": 25, "ranker": 1},
}


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load defaults from ``cloud/config.yaml``.

    Uses PyYAML when available; otherwise falls back to a minimal indentation
    aware parse sufficient for the flat keys this config file uses. Any
    failure silently falls back to :data:`DEFAULT_CONFIG`.
    """
    config: dict[str, Any] = {
        **DEFAULT_CONFIG,
        "epochs": dict(DEFAULT_CONFIG["epochs"]),
    }
    if not path.is_file():
        return config
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        parsed = _naive_yaml(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except Exception:
            loaded = None
        parsed = loaded if isinstance(loaded, dict) else {}
    for key in ("project", "region", "bucket", "image_uri", "machine", "accelerator"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            config[key] = value
    for key in ("replica_count",):
        value = parsed.get(key)
        if isinstance(value, int):
            config[key] = value
    if isinstance(parsed.get("spot"), bool):
        config["spot"] = parsed["spot"]
    epochs = parsed.get("epochs")
    if isinstance(epochs, dict):
        for task in TASKS:
            value = epochs.get(task)
            if isinstance(value, int) and value > 0:
                config["epochs"][task] = value
    return config


def _naive_yaml(text: str) -> dict[str, Any]:
    """Parse the small subset of YAML used by ``cloud/config.yaml``.

    Supports top-level ``key: value`` pairs and one level of nested mapping
    (used for ``epochs:``). Values are coerced to int/bool/str. This is NOT a
    general YAML parser.
    """
    result: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indented = raw_line[:1] in (" ", "\t")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            if not indented:
                current_section = key
                result.setdefault(key, {})
            continue
        coerced: Any = _coerce_scalar(value)
        if indented and current_section:
            section = result.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key] = coerced
        else:
            result[key] = coerced
            current_section = None
    return result


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _utc_run_id(task: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{task}"


def _default_image_uri(project: str, region: str) -> str:
    return f"{region}-docker.pkg.dev/{project}/furniture-ai/trainer:latest"


def submit_task(
    task: str,
    project: str,
    region: str,
    bucket: str,
    image_uri: str,
    spot: bool = True,
    machine: str = "g2-standard-4",
    accelerator: str = "NVIDIA_L4",
    replica_count: int = 1,
) -> str:
    """Submit one Vertex AI custom training job and return its resource name.

    The container is invoked with ``--task <task>`` and the environment
    variables ``GCP_PROJECT``, ``GCP_REGION``, ``GCS_BUCKET`` and ``RUN_ID``
    (contract §2.1 of SPEC.md).
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")

    from google.cloud import aiplatform  # lazy import: SDK not required at module import time

    aiplatform.init(project=project, location=region, staging_bucket=f"gs://{bucket}")

    run_id = _utc_run_id(task)
    display_name = f"furniture-ai-{task}-{run_id}"
    env = [
        {"name": "GCP_PROJECT", "value": project},
        {"name": "GCP_REGION", "value": region},
        {"name": "GCS_BUCKET", "value": bucket},
        {"name": "RUN_ID", "value": run_id},
    ]
    worker_pool_spec: dict[str, Any] = {
        "machine_spec": {
            "machine_type": machine,
            "accelerator_type": accelerator,
            "accelerator_count": 1,
        },
        "replica_count": replica_count,
        "container_spec": {
            "image_uri": image_uri,
            "args": ["--task", task],
            "env": env,
        },
    }
    if spot:
        worker_pool_spec["machine_spec"]["scheduling"] = {"strategy": "SPOT"}

    job = aiplatform.CustomJob(display_name=display_name, worker_pool_specs=[worker_pool_spec])
    job.submit()
    return job.resource_name


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cloud.vertex_jobs",
        description="Submit Vertex AI custom training jobs for furniture-ai-system.",
    )
    parser.add_argument(
        "--task",
        choices=(*TASKS, "all"),
        default="all",
        help="Which training job(s) to submit (default: all).",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT") or config["project"],
        help="GCP project id (default: $GCP_PROJECT or cloud/config.yaml).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION") or config["region"],
        help="Vertex AI region (default: $GCP_REGION or cloud/config.yaml).",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET") or config["bucket"] or "",
        help="GCS staging bucket name without gs:// (default: furniture-ai-training-<project>).",
    )
    parser.add_argument(
        "--image-uri",
        default=config["image_uri"] or "",
        help=(
            "Training container image URI "
            "(default: <region>-docker.pkg.dev/<project>/furniture-ai/trainer:latest)."
        ),
    )
    parser.add_argument("--machine", default=config["machine"], help="Vertex machine type.")
    parser.add_argument(
        "--accelerator", default=config["accelerator"], help="GPU accelerator type."
    )
    parser.add_argument(
        "--replica-count",
        type=int,
        default=int(config["replica_count"]),
        help="Number of worker replicas per job.",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Use on-demand VMs instead of spot/preemptible capacity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on successful submission of all jobs."""
    config = _load_config()
    parser = _build_parser(config)
    args = parser.parse_args(argv)

    if not args.project or args.project == DEFAULT_CONFIG["project"]:
        print(
            "error: no GCP project configured; pass --project or set $GCP_PROJECT",
            file=sys.stderr,
        )
        return 2

    bucket = args.bucket or f"furniture-ai-training-{args.project}"
    image_uri = args.image_uri or _default_image_uri(args.project, args.region)
    spot = not args.no_spot and bool(config["spot"])

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for task in tasks:
        print(f"==> submitting Vertex AI job: task={task} image={image_uri} bucket={bucket}")
        resource_name = submit_task(
            task=task,
            project=args.project,
            region=args.region,
            bucket=bucket,
            image_uri=image_uri,
            spot=spot,
            machine=args.machine,
            accelerator=args.accelerator,
            replica_count=args.replica_count,
        )
        print(f"==> submitted {task}: {resource_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
