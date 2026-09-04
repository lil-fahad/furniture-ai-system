from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nvidia_runtime_probe.py"
spec = importlib.util.spec_from_file_location("nvidia_runtime_probe", SCRIPT)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_cpu_report_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    report = probe.build_report()
    assert report["cuda_available"] is False
    assert report["device_count"] == 0
    assert report["devices"] == []


def test_require_cuda_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        probe.build_report(require_cuda=True)
