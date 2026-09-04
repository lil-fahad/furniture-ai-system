#!/usr/bin/env python3
"""Fail-closed NVIDIA/CUDA capability probe for FurnitureAI.

This utility performs no training and starts no cloud resources. It emits a
machine-readable report that can be attached to benchmark/training provenance.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


def build_report(require_cuda: bool = False) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but no NVIDIA CUDA device is visible to PyTorch")

    report: dict[str, object] = {
        "schema_version": 1,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }
    if not cuda_available:
        return report

    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        bf16 = bool(torch.cuda.is_bf16_supported())
        devices.append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": f"{major}.{minor}",
                "total_memory_bytes": props.total_memory,
                "multiprocessors": props.multi_processor_count,
                "fp16_supported": major >= 5,
                "bf16_supported": bf16,
                "tf32_supported": major >= 8,
            }
        )
    report["devices"] = devices
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(require_cuda=args.require_cuda)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
