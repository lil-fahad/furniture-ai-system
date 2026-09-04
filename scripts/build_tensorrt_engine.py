from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_flags(
    min_shapes: str | None,
    opt_shapes: str | None,
    max_shapes: str | None,
) -> list[str]:
    provided = (min_shapes, opt_shapes, max_shapes)
    if any(provided) and not all(provided):
        raise ValueError("min/opt/max shapes must be provided together")
    if not all(provided):
        return []
    return [
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
    ]


def build_command(
    trtexec: str,
    onnx: Path,
    engine: Path,
    *,
    min_shapes: str | None = None,
    opt_shapes: str | None = None,
    max_shapes: str | None = None,
) -> list[str]:
    return [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--skipInference",
        "--profilingVerbosity=detailed",
        *shape_flags(min_shapes, opt_shapes, max_shapes),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strongly typed TensorRT engine from an ONNX model. "
            "Mixed precision must be encoded into the ONNX graph before this step."
        )
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--min-shapes")
    parser.add_argument("--opt-shapes")
    parser.add_argument("--max-shapes")
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec is not installed or not on PATH")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        trtexec,
        args.onnx.resolve(),
        args.engine.resolve(),
        min_shapes=args.min_shapes,
        opt_shapes=args.opt_shapes,
        max_shapes=args.max_shapes,
    )
    version = subprocess.run(
        [trtexec, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(command, check=True)
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise RuntimeError("TensorRT did not produce a non-empty engine")

    manifest = args.manifest or args.engine.with_suffix(args.engine.suffix + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "builder": "NVIDIA TensorRT trtexec",
        "trtexec_version": version,
        "onnx_path": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "engine_path": str(args.engine.resolve()),
        "engine_sha256": sha256_file(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "strongly_typed_expected": True,
        "precision_policy": "encoded_in_onnx_before_engine_build",
        "cuda_graph_policy": "TensorRT runtime default; benchmark on target GPU",
        "shape_profile": {
            "min": args.min_shapes,
            "opt": args.opt_shapes,
            "max": args.max_shapes,
        },
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
