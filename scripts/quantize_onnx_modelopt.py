from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

QUANTIZE_MODES = ("fp8", "int8")
CALIBRATION_METHODS = ("entropy", "max")

CalibrationData = np.ndarray | dict[str, np.ndarray]
QuantizeFunction = Callable[..., Any]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_array(array: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype.hasobject:
        raise ValueError(f"{label} must not contain Python objects")
    if result.size == 0:
        raise ValueError(f"{label} must not be empty")
    return result


def load_calibration_data(path: Path) -> tuple[CalibrationData, dict[str, object]]:
    """Load non-pickled calibration arrays and return auditable metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise ValueError("Calibration data file must not be empty")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        try:
            data = _validate_array(
                np.load(source, allow_pickle=False),
                label="Calibration array",
            )
        except ValueError as exc:
            raise ValueError(f"Unsafe or invalid calibration data: {source}") from exc
        metadata = {
            "format": "npy",
            "arrays": {
                "input": {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "elements": int(data.size),
                }
            },
        }
        return data, metadata

    if suffix == ".npz":
        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(source, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError("Calibration archive must contain at least one array")
                for key in sorted(archive.files):
                    arrays[key] = _validate_array(
                        archive[key],
                        label=f"Calibration array {key!r}",
                    )
        except ValueError as exc:
            raise ValueError(f"Unsafe or invalid calibration data: {source}") from exc
        metadata = {
            "format": "npz",
            "arrays": {
                key: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "elements": int(array.size),
                }
                for key, array in arrays.items()
            },
        }
        return arrays, metadata

    raise ValueError("Calibration data must be a .npy or .npz file")


def _resolve_quantize_function() -> QuantizeFunction:
    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA ModelOpt ONNX support is not installed. "
            'Install it on the controlled optimization host with '
            '`pip install "nvidia-modelopt[onnx]"`.'
        ) from exc
    return quantize


def run_quantization(
    *,
    onnx: Path,
    calibration_data: Path,
    output: Path,
    quantize_mode: str,
    calibration_method: str = "entropy",
    manifest: Path | None = None,
    quantize_fn: QuantizeFunction | None = None,
) -> dict[str, object]:
    """Quantize one ONNX model with explicit, real calibration data."""
    if quantize_mode not in QUANTIZE_MODES:
        raise ValueError(f"unsupported quantize mode: {quantize_mode}")
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"unsupported calibration method: {calibration_method}")

    source = Path(onnx)
    calibration = Path(calibration_data)
    destination = Path(output)
    manifest_path = Path(manifest) if manifest else destination.with_suffix(destination.suffix + ".json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise ValueError("Source ONNX model must not be empty")
    if not calibration.is_file():
        raise FileNotFoundError(calibration)

    resolved_source = source.resolve()
    resolved_calibration = calibration.resolve()
    resolved_output = destination.resolve()
    resolved_manifest = manifest_path.resolve()
    protected_inputs = {resolved_source, resolved_calibration}
    if resolved_output in protected_inputs:
        raise ValueError("Output path must differ from source ONNX and calibration data")
    if resolved_manifest in protected_inputs | {resolved_output}:
        raise ValueError("Manifest path must differ from source, calibration, and output paths")
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse existing output: {destination}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    loaded_calibration, calibration_metadata = load_calibration_data(calibration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    executor = quantize_fn or _resolve_quantize_function()
    executor(
        str(resolved_source),
        quantize_mode=quantize_mode,
        calibration_data=loaded_calibration,
        calibration_method=calibration_method,
        output_path=str(resolved_output),
    )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("ModelOpt did not produce a non-empty ONNX model")

    payload: dict[str, object] = {
        "schema_version": 1,
        "optimizer": "NVIDIA Model Optimizer",
        "api": "modelopt.onnx.quantization.quantize",
        "quantize_mode": quantize_mode,
        "calibration_method": calibration_method,
        "random_calibration_allowed": False,
        "source_onnx": {
            "path": str(resolved_source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "calibration_data": {
            "path": str(resolved_calibration),
            "size_bytes": calibration.stat().st_size,
            "sha256": sha256_file(calibration),
            **calibration_metadata,
        },
        "quantized_onnx": {
            "path": str(resolved_output),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        },
        "promotion_policy": (
            "Re-evaluate the quantized model on the same pinned real-data benchmark "
            "before building or promoting a TensorRT engine."
        ),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize an ONNX model with NVIDIA ModelOpt using explicit calibration arrays."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_quantization(
        onnx=args.onnx,
        calibration_data=args.calibration_data,
        output=args.output,
        quantize_mode=args.quantize_mode,
        calibration_method=args.calibration_method,
        manifest=args.manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
