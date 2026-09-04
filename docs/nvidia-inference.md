# NVIDIA inference path

FurnitureAI keeps deterministic geometry authoritative while allowing the local vision layer to use NVIDIA GPU acceleration when CUDA is available.

## Runtime policy

- CPU remains a supported fallback and uses FP32.
- CUDA uses automatic mixed precision by default: BF16 when the installed GPU/PyTorch runtime reports BF16 support, otherwise FP16.
- Explicit CUDA requests fail closed when CUDA is unavailable.
- Explicit BF16 fails closed when the device does not report BF16 support.
- TF32 is enabled only for explicit FP32 inference on compute capability 8.0 or newer.
- `torch.compile(mode="reduce-overhead")` is opt-in because model- and driver-specific benchmarking is required before enabling it globally.
- Model weights are still loaded only from verified local files; acceleration never enables implicit model downloads.

Use `python scripts/nvidia_runtime_probe.py` to record the runtime profile on the target host.

## ModelOpt FP8 / INT8 policy

TensorRT 11 expects precision and quantization to be encoded into the model before engine construction. `scripts/quantize_onnx_modelopt.py` provides FurnitureAI's controlled NVIDIA ModelOpt preprocessing step for FP8 and INT8 ONNX models.

The wrapper intentionally requires an explicit calibration-data file. FurnitureAI does not accept random calibration as production evidence because it would make the resulting quantized artifact unrelated to the real furniture-image distribution. The wrapper records SHA-256 and byte size for the source ONNX, calibration data, and quantized ONNX output.

ModelOpt remains an optional deployment tool rather than a base runtime dependency. Install it only on the controlled optimization host using NVIDIA's supported package source, then retain the generated manifest together with the evaluation report.

A quantized model is still a candidate until its accuracy is re-evaluated against the same pinned real-data benchmark used for the unquantized baseline. Faster latency alone is not a promotion criterion.

## TensorRT 11 path

`scripts/build_tensorrt_engine.py` builds a serialized engine from an ONNX model with a reproducible SHA-256 manifest. The builder intentionally does not pass legacy `--fp16`, `--bf16`, `--int8`, or `--best` flags. TensorRT 11 uses strongly typed networks, so mixed precision/quantization must be encoded into the ONNX model before engine construction, for example with NVIDIA ModelOpt.

The builder supports explicit min/opt/max dynamic-shape profiles and uses `--skipInference` so engine construction and performance benchmarking remain separate evidence-producing steps. Benchmark the resulting engine on the exact deployment GPU before any production promotion. CUDA Graph behavior is left to the target TensorRT runtime default and must be measured on that hardware.

No TensorRT engine is portable evidence by itself: retain the ONNX SHA-256, engine SHA-256, TensorRT version, target GPU model/compute capability, driver/CUDA versions, shape profile, and measured latency/throughput together.
