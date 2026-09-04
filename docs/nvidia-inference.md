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

## TensorRT 11 path

`scripts/build_tensorrt_engine.py` builds a serialized engine from an ONNX model with a reproducible SHA-256 manifest. The builder intentionally does not pass legacy `--fp16`, `--bf16`, `--int8`, or `--best` flags. TensorRT 11 uses strongly typed networks, so mixed precision/quantization must be encoded into the ONNX model before engine construction (for example with NVIDIA ModelOpt).

The builder supports explicit min/opt/max dynamic-shape profiles and uses `--skipInference` so engine construction and performance benchmarking remain separate evidence-producing steps. Benchmark the resulting engine on the exact deployment GPU before any production promotion. CUDA Graph behavior is left to the target TensorRT runtime default and must be measured on that hardware.

No TensorRT engine is portable evidence by itself: retain the ONNX SHA-256, engine SHA-256, TensorRT version, target GPU model/compute capability, driver/CUDA versions, shape profile, and measured latency/throughput together.
