from __future__ import annotations

import pytest
import torch

from furniture_ai.nvidia_acceleration import (
    NvidiaAccelerationUnavailable,
    inference_context,
    resolve_nvidia_runtime,
)


def test_explicit_cpu_runtime_is_safe_and_fp32() -> None:
    profile = resolve_nvidia_runtime("cpu")
    assert profile.device == "cpu"
    assert profile.precision == "fp32"
    assert profile.autocast_enabled is False
    assert profile.tf32_enabled is False
    assert profile.torch_compile_enabled is False
    with inference_context(profile):
        result = torch.tensor([1.0]) + 1
    assert result.item() == 2.0


def test_cpu_rejects_cuda_only_precision() -> None:
    with pytest.raises(NvidiaAccelerationUnavailable, match="requires an NVIDIA CUDA device"):
        resolve_nvidia_runtime("cpu", precision="fp16")


def test_explicit_cuda_fails_closed_without_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(NvidiaAccelerationUnavailable, match="no NVIDIA CUDA device"):
        resolve_nvidia_runtime("cuda")


def test_auto_cuda_prefers_bf16_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    profile = resolve_nvidia_runtime("cuda", precision="auto", enable_torch_compile=True)

    assert profile.device == "cuda"
    assert profile.gpu_name == "NVIDIA Test GPU"
    assert profile.compute_capability == (9, 0)
    assert profile.precision == "bf16"
    assert profile.autocast_enabled is True
    assert profile.tf32_enabled is False
    assert profile.torch_compile_enabled is True


def test_cuda_fp32_enables_tf32_only_on_ampere_or_newer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    profile = resolve_nvidia_runtime("cuda", precision="fp32")

    assert profile.precision == "fp32"
    assert profile.autocast_enabled is False
    assert profile.tf32_enabled is True
