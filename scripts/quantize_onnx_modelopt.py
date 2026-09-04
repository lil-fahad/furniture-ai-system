from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

QUANTIZE_MODES = ("fp8", "int8")
CALIBRATION_METHODS = ("entropy", "max")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_modelopt_command(
    python_executable: str,
    onnx: Path,
    calibration_data: Path,
    output: Path,
    *,
    quantize_mode: str,
    calibration_method: str,
) -> list[str]:
    if quantize_mode not in QUANTIZE_MODES:
        raise ValueError(f"unsupported quantize mode: {quantize_mode}")
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"unsupported calibration method: {calibration_method}")
    return [
        python_executable,
        "-m",
        "modelopt.onnx.quantization",
        f"--onnx_path={onnx}",
        f"--quantize_mode={quantize_mode}",
        f"--calibration_data={calibration_data}",
        f"--calibration_method={calibration_method}",
        f"--output_path={output}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize an ONNX model with NVIDIA ModelOpt before TensorRT 11 engine building. "
            "Calibration data is mandatory; random calibration is intentionally forbidden."
        )
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quantize-mode", choices=QUANTIZE_MODES, required=True)
    parser.add_argument(
        "--calibration-method",
        choices=CALIBRATION_METHODS,
        default="entropy",
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if importlib.util.find_spec("modelopt") is None:
        raise RuntimeError(
            "NVIDIA ModelOpt is not installed. Install nvidia-modelopt from NVIDIA's index."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_modelopt_command(
        sys.executable,
        args.onnx.resolve(),
        args.calibration_data.resolve(),
        args.output.resolve(),
        quantize_mode=args.quantize_mode,
        calibration_method=args.calibration_method,
    )
    subprocess.run(command, check=True)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("ModelOpt did not produce a non-empty ONNX model")

    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": "NVIDIA TensorRT Model Optimizer",
        "quantize_mode": args.quantize_mode,
        "calibration_method": args.calibration_method,
        "random_calibration_allowed": False,
        "source_onnx": {
            "path": str(args.onnx.resolve()),
            "sha256": sha256_file(args.onnx),
            "size_bytes": args.onnx.stat().st_size,
        },
        "calibration_data": {
            "path": str(args.calibration_data.resolve()),
            "sha256": sha256_file(args.calibration_data),
            "size_bytes": args.calibration_data.stat().st_size,
        },
        "quantized_onnx": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "size_bytes": args.output.stat().st_size,
        },
        "next_step": "Build a strongly typed TensorRT 11 engine from this ONNX artifact.",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
