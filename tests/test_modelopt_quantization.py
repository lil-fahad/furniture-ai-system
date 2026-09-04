from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.quantize_onnx_modelopt import load_calibration_data, run_quantization


def test_load_npy_calibration_returns_array_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npy"
    np.save(path, np.ones((2, 3), dtype=np.float32))

    data, metadata = load_calibration_data(path)

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)
    assert metadata["format"] == "npy"
    assert metadata["arrays"]["input"]["dtype"] == "float32"


def test_load_npz_calibration_returns_named_arrays(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npz"
    np.savez(
        path,
        image=np.ones((1, 3, 8, 8), dtype=np.float32),
        scale=np.ones((1,), dtype=np.float32),
    )

    data, metadata = load_calibration_data(path)

    assert isinstance(data, dict)
    assert sorted(data) == ["image", "scale"]
    assert metadata["format"] == "npz"
    assert metadata["arrays"]["image"]["shape"] == [1, 3, 8, 8]


def test_object_calibration_is_rejected_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npy"
    np.save(path, np.array([{"secret": "object"}], dtype=object), allow_pickle=True)

    with pytest.raises(ValueError, match="Unsafe or invalid calibration data"):
        load_calibration_data(path)


def test_empty_calibration_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.npy"
    np.save(path, np.array([], dtype=np.float32))

    with pytest.raises(ValueError, match="Unsafe or invalid calibration data"):
        load_calibration_data(path)


@pytest.mark.parametrize("mode", ["fp4", "int4", "best", "fp16"])
def test_quantization_rejects_modes_outside_policy(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="unsupported quantize mode"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=tmp_path / "output.onnx",
            quantize_mode=mode,
            quantize_fn=lambda *args, **kwargs: None,
        )


def test_quantization_rejects_unknown_calibration_method(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="unsupported calibration method"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=tmp_path / "output.onnx",
            quantize_mode="int8",
            calibration_method="random",
            quantize_fn=lambda *args, **kwargs: None,
        )


def test_output_cannot_overwrite_source_or_calibration(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Output path must differ"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=source,
            quantize_mode="fp8",
            quantize_fn=lambda *args, **kwargs: None,
        )


def test_manifest_cannot_overwrite_source_or_output(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))
    output = tmp_path / "model.fp8.onnx"

    with pytest.raises(ValueError, match="Manifest path must differ"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=output,
            manifest=source,
            quantize_mode="fp8",
            quantize_fn=lambda *args, **kwargs: None,
        )


def test_existing_output_is_rejected_before_quantization(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))
    output = tmp_path / "model.fp8.onnx"
    output.write_bytes(b"stale-output")
    called = False

    def fake_quantize(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    with pytest.raises(FileExistsError, match="existing output"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=output,
            quantize_mode="fp8",
            quantize_fn=fake_quantize,
        )
    assert called is False


def test_quantization_passes_arrays_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"source-onnx")
    calibration = tmp_path / "calibration.npz"
    np.savez(calibration, input=np.ones((2, 4), dtype=np.float32))
    output = tmp_path / "model.fp8.onnx"
    manifest = tmp_path / "model.fp8.manifest.json"
    observed: dict[str, object] = {}

    def fake_quantize(onnx_path: str, **kwargs: object) -> None:
        observed["onnx_path"] = onnx_path
        observed.update(kwargs)
        output.write_bytes(b"quantized-onnx")

    payload = run_quantization(
        onnx=source,
        calibration_data=calibration,
        output=output,
        manifest=manifest,
        quantize_mode="fp8",
        calibration_method="max",
        quantize_fn=fake_quantize,
    )

    assert observed["onnx_path"] == str(source.resolve())
    assert isinstance(observed["calibration_data"], dict)
    assert observed["quantize_mode"] == "fp8"
    assert observed["calibration_method"] == "max"
    assert observed["output_path"] == str(output.resolve())
    assert payload["random_calibration_allowed"] is False
    assert payload["source_onnx"]["sha256"]
    assert payload["calibration_data"]["sha256"]
    assert payload["quantized_onnx"]["sha256"]
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload


def test_quantization_requires_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"source-onnx")
    calibration = tmp_path / "calibration.npy"
    np.save(calibration, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(RuntimeError, match="did not produce"):
        run_quantization(
            onnx=source,
            calibration_data=calibration,
            output=tmp_path / "missing.onnx",
            quantize_mode="int8",
            quantize_fn=lambda *args, **kwargs: None,
        )
