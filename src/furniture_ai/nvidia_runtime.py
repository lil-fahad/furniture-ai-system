"""NVIDIA-aware inference policy with deterministic, fail-closed selection.

This module never starts paid compute. It only inspects the local PyTorch runtime and
selects an execution policy that callers may use for inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceRuntime:
    backend: str
    device: str
    precision: str
    cuda_available: bool
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    tf32: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_inference_runtime(*, prefer_gpu: bool = True) -> InferenceRuntime:
    """Return a measured local runtime; CPU is the explicit safe fallback."""
    try:
        import torch
    except ImportError:
        return InferenceRuntime(
            backend="cpu",
            device="cpu",
            precision="fp32",
            cuda_available=False,
        )

    if not prefer_gpu or not torch.cuda.is_available():
        return InferenceRuntime(
            backend="pytorch",
            device="cpu",
            precision="fp32",
            cuda_available=False,
        )

    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    major = capability[0]
    precision = "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"
    tf32 = major >= 8
    return InferenceRuntime(
        backend="pytorch-cuda",
        device=f"cuda:{index}",
        precision=precision,
        cuda_available=True,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(index),
        compute_capability=capability,
        tf32=tf32,
    )


def autocast_kwargs(runtime: InferenceRuntime) -> dict[str, Any]:
    """Translate a runtime decision into torch.autocast kwargs."""
    if not runtime.cuda_available:
        return {"enabled": False, "device_type": "cpu"}
    import torch

    dtype = torch.bfloat16 if runtime.precision == "bf16" else torch.float16
    return {"enabled": True, "device_type": "cuda", "dtype": dtype}
