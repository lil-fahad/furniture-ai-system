from __future__ import annotations

from pathlib import Path

import pytest

from scripts.quantize_onnx_modelopt import build_modelopt_command


def test_modelopt_command_requires_explicit_real_calibration(tmp_path: Path) -> None:
    command = build_modelopt_command(
        "/usr/bin/python3",
        tmp_path / "model.onnx",
        tmp_path / "calibration.npz",
        tmp_path / "model.fp8.onnx",
        quantize_mode="fp8",
        calibration_method="entropy",
    )

    assert command == [
        "/usr/bin/python3",
        "-m",
        "modelopt.onnx.quantization",
        f"--onnx_path={tmp_path / 'model.onnx'}",
        "--quantize_mode=fp8",
        f"--calibration_data={tmp_path / 'calibration.npz'}",
        "--calibration_method=entropy",
        f"--output_path={tmp_path / 'model.fp8.onnx'}",
    ]
    assert not any("random" in argument.lower() for argument in command)


@pytest.mark.parametrize("mode", ["fp4", "int4", "best", "fp16"])
def test_modelopt_command_rejects_unapproved_quantization_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported quantize mode"):
        build_modelopt_command(
            "python",
            tmp_path / "model.onnx",
            tmp_path / "calibration.npz",
            tmp_path / "output.onnx",
            quantize_mode=mode,
            calibration_method="entropy",
        )


def test_modelopt_command_rejects_unknown_calibration_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported calibration method"):
        build_modelopt_command(
            "python",
            tmp_path / "model.onnx",
            tmp_path / "calibration.npz",
            tmp_path / "output.onnx",
            quantize_mode="int8",
            calibration_method="random",
        )
