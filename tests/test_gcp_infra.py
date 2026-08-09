"""Offline tests for the GCP/Vertex AI infrastructure deliverables (WP-A).

These tests must pass with the standard library only: no network access, no
gcloud CLI calls, and no google-cloud SDKs installed (cloud.vertex_jobs uses
lazy imports).
"""

from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CLOUD_DIR = REPO_ROOT / "cloud"
VERTEX_JOBS = CLOUD_DIR / "vertex_jobs.py"
DOCKERFILE = CLOUD_DIR / "Dockerfile.training"
CONFIG_YAML = CLOUD_DIR / "config.yaml"
BOOTSTRAP = REPO_ROOT / "scripts" / "gcp_bootstrap.sh"
CLOUD_README = CLOUD_DIR / "README.md"
AR_GUIDE = REPO_ROOT / "docs" / "GCP_TRAINING_AR.md"

NEW_PY_FILES = [CLOUD_DIR / "__init__.py", VERTEX_JOBS]


@pytest.mark.parametrize("path", NEW_PY_FILES, ids=lambda p: p.name)
def test_new_python_files_compile(path: Path, tmp_path: Path) -> None:
    assert path.is_file(), f"missing {path}"
    py_compile.compile(str(path), cfile=str(tmp_path / f"{path.stem}.pyc"), doraise=True)


def test_vertex_jobs_imports_without_google_cloud_sdk() -> None:
    """The module must import even when google-cloud-aiplatform is absent."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import cloud.vertex_jobs as vertex_jobs
    finally:
        sys.path.remove(str(REPO_ROOT))
    assert vertex_jobs.TASKS == ("room", "segmenter", "ranker")
    assert callable(vertex_jobs.submit_task)
    assert callable(vertex_jobs.main)


def test_vertex_jobs_naive_yaml_parses_config() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from cloud.vertex_jobs import _naive_yaml
    finally:
        sys.path.remove(str(REPO_ROOT))
    parsed = _naive_yaml(CONFIG_YAML.read_text(encoding="utf-8"))
    assert parsed["region"] == "us-central1"
    assert parsed["machine"] == "g2-standard-4"
    assert parsed["accelerator"] == "NVIDIA_L4"
    assert parsed["spot"] is True
    assert parsed["epochs"] == {"room": 15, "segmenter": 25, "ranker": 1}


def test_vertex_jobs_main_rejects_unknown_task() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from cloud.vertex_jobs import main
    finally:
        sys.path.remove(str(REPO_ROOT))
    with pytest.raises(SystemExit) as excinfo:
        main(["--task", "bogus"])
    assert excinfo.value.code == 2


def test_bootstrap_script_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this system")
    assert BOOTSTRAP.is_file(), "missing scripts/gcp_bootstrap.sh"
    result = subprocess.run(
        [bash, "-n", str(BOOTSTRAP)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_bootstrap_script_content() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    for flag in ("--project", "--region", "--bucket", "--skip-data", "--skip-image", "--skip-jobs"):
        assert flag in text, f"missing flag {flag}"
    for api in (
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "cloudbuild.googleapis.com",
        "artifactregistry.googleapis.com",
    ):
        assert api in text, f"missing API {api}"
    assert "furniture-ai-training" in text
    assert "gcloud builds submit" in text
    assert "training.data_ingest.stage_all" in text
    assert "cloud.vertex_jobs" in text
    assert "==>" in text


def test_config_yaml_defaults() -> None:
    assert CONFIG_YAML.is_file(), "missing cloud/config.yaml"
    text = CONFIG_YAML.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        # Plain-text fallback checks (no PyYAML in the minimal environment).
        for needle in (
            "region: us-central1",
            "machine: g2-standard-4",
            "accelerator: NVIDIA_L4",
            "spot: true",
            "room: 15",
            "segmenter: 25",
            "ranker: 1",
        ):
            assert needle in text, f"missing config entry: {needle!r}"
        assert "project:" in text
        assert "epochs:" in text
    else:
        config = yaml.safe_load(text)
        assert config["project"]  # placeholder value, must be set
        assert config["region"] == "us-central1"
        assert config["machine"] == "g2-standard-4"
        assert config["accelerator"] == "NVIDIA_L4"
        assert config["replica_count"] == 1
        assert config["spot"] is True
        assert config["epochs"] == {"room": 15, "segmenter": 25, "ranker": 1}


def test_dockerfile_training_content() -> None:
    assert DOCKERFILE.is_file(), "missing cloud/Dockerfile.training"
    text = DOCKERFILE.read_text(encoding="utf-8")
    # Python 3.11+ CUDA base image (repo requires >= 3.11; uses 3.11-only stdlib).
    assert "FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime" in text
    assert "--ignore-requires-python" not in text
    # Repo installed with the [training] extra plus the cloud SDK clients.
    assert "[training]" in text
    assert "google-cloud-storage" in text
    assert "google-cloud-aiplatform" in text
    # Required COPY sources per spec.
    for needle in ("pyproject.toml", "README.md", "src", "training", "data"):
        assert any(line.startswith("COPY") and needle in line for line in text.splitlines()), (
            f"missing COPY of {needle}"
        )
    # Entrypoint must dispatch to the cloud training entry module.
    assert 'ENTRYPOINT ["python", "-m", "training.cloud_entry"]' in text


def test_docs_exist_and_cover_required_topics() -> None:
    assert CLOUD_README.is_file(), "missing cloud/README.md"
    readme = CLOUD_README.read_text(encoding="utf-8")
    needles = ("gcloud ai custom-jobs list", "stream-logs", "gcloud storage cp", "Prerequisites")
    for needle in needles:
        assert needle in readme, f"cloud/README.md missing: {needle}"

    assert AR_GUIDE.is_file(), "missing docs/GCP_TRAINING_AR.md"
    guide = AR_GUIDE.read_text(encoding="utf-8")
    assert "round-office-505007-q4" in guide
    assert "gcp_bootstrap.sh" in guide
    assert "gcloud storage cp -r" in guide
    # Arabic section markers for cost and troubleshooting.
    assert "التكلفة" in guide
    assert "استكشاف الأخطاء" in guide
