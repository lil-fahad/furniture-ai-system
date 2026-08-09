from __future__ import annotations

import re
import tomllib
from pathlib import Path

import furniture_ai

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_package_version_matches_pyproject() -> None:
    assert furniture_ai.__version__ == _pyproject()["project"]["version"]


def test_python_multipart_floor_excludes_cve_2024_53981() -> None:
    dependencies = _pyproject()["project"]["dependencies"]
    (requirement,) = [d for d in dependencies if d.startswith("python-multipart")]
    match = re.search(r">=(\d+)\.(\d+)\.(\d+)", requirement)
    assert match is not None, "python-multipart must declare a lower bound"
    floor = tuple(int(part) for part in match.groups())
    assert floor >= (0, 0, 18), "CVE-2024-53981 is fixed in python-multipart 0.0.18"


def test_dockerfile_matches_release_metadata() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert f'org.opencontainers.image.version="{furniture_ai.__version__}"' in dockerfile
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "libgl1" not in dockerfile
