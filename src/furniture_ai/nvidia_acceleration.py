from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal

Precision = Literal["auto", "fp32", "fp16", "bf16"]


class NvidiaAccelerationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class NvidiaRuntimeProfile:
    device: str
    cuda_available: bool
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    precision: Literal["fp32", "fp16", "bf16"]
    autocast_enabled: bool
    tf32_enabled: bool
    torch_compile_enabled: bool

    def as_public_dict(self) -> dict[str, object]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise NvidiaAccelerationUnavailable("PyTorch is required for accelerated inference") from exc
    return torch


def resolve_nvidia_runtime(
    requested_device: str | None = None,
    *,
    precision: Precision = "auto",
    enable_torch_compile: bool = False,
) -> NvidiaRuntimeProfile:
    torch = _torch()
    cuda_available = bool(torch.cuda.is_available())

    if requested_device and requested_device.startswith("cuda") and not cuda_available:
        raise NvidiaAccelerationUnavailable("CUDA was requested but no NVIDIA CUDA device is available")

    device = requested_device or ("cuda" if cuda_available else "cpu")
    if not device.startswith("cuda"):
        if precision not in {"auto", "fp32"}:
            raise NvidiaAccelerationUnavailable(
                f"Precision {precision!r} requires an NVIDIA CUDA device"
            )
        return NvidiaRuntimeProfile(
            device=device,
            cuda_available=cuda_available,
            gpu_name=None,
            compute_capability=None,
            precision="fp32",
            autocast_enabled=False,
            tf32_enabled=False,
            torch_compile_enabled=False,
        )

    index = torch.device(device).index
    if index is None:
        index = int(torch.cuda.current_device())
    gpu_name = str(torch.cuda.get_device_name(index))
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))

    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    if precision == "auto":
        resolved_precision: Literal["fp32", "fp16", "bf16"] = (
            "bf16" if bf16_supported else "fp16"
        )
    else:
        resolved_precision = precision

    if resolved_precision == "bf16" and not bf16_supported:
        raise NvidiaAccelerationUnavailable("BF16 was requested but is unsupported by this CUDA device")

    return NvidiaRuntimeProfile(
        device=device,
        cuda_available=True,
        gpu_name=gpu_name,
        compute_capability=(capability[0], capability[1]),
        precision=resolved_precision,
        autocast_enabled=resolved_precision in {"fp16", "bf16"},
        tf32_enabled=resolved_precision == "fp32" and capability[0] >= 8,
        torch_compile_enabled=bool(enable_torch_compile),
    )


def prepare_model(model: Any, profile: NvidiaRuntimeProfile) -> Any:
    torch = _torch()
    model = model.to(profile.device)
    model.eval()

    if profile.device.startswith("cuda") and profile.tf32_enabled:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    if profile.torch_compile_enabled:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            raise NvidiaAccelerationUnavailable("torch.compile is unavailable in this PyTorch build")
        model = compile_fn(model, mode="reduce-overhead", fullgraph=False)
    return model


@contextmanager
def inference_context(profile: NvidiaRuntimeProfile) -> Iterator[None]:
    torch = _torch()
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        if profile.autocast_enabled:
            dtype = torch.bfloat16 if profile.precision == "bf16" else torch.float16
            stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
        yield
