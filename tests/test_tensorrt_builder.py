from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_tensorrt_engine import build_command, shape_flags


def test_tensorrt_builder_uses_strongly_typed_compatible_flags(tmp_path: Path) -> None:
    command = build_command(
        "/usr/bin/trtexec",
        tmp_path / "model.onnx",
        tmp_path / "model.engine",
    )

    joined = " ".join(command)
    assert "--skipInference" in command
    assert "--profilingVerbosity=detailed" in command
    assert "--fp16" not in joined
    assert "--bf16" not in joined
    assert "--int8" not in joined
    assert "--best" not in joined


def test_tensorrt_shape_profile_requires_complete_triplet() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        shape_flags("input:1x3x640x640", None, "input:4x3x640x640")


def test_tensorrt_shape_profile_is_forwarded() -> None:
    flags = shape_flags(
        "pixel_values:1x3x512x512",
        "pixel_values:1x3x640x640",
        "pixel_values:4x3x960x960",
    )
    assert flags == [
        "--minShapes=pixel_values:1x3x512x512",
        "--optShapes=pixel_values:1x3x640x640",
        "--maxShapes=pixel_values:4x3x960x960",
    ]
