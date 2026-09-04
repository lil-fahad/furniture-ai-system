from furniture_ai.nvidia_runtime import InferenceRuntime, autocast_kwargs, detect_inference_runtime


def test_runtime_can_be_forced_to_cpu() -> None:
    runtime = detect_inference_runtime(prefer_gpu=False)
    assert runtime.device == "cpu"
    assert runtime.precision == "fp32"
    assert runtime.cuda_available is False


def test_cpu_autocast_is_disabled() -> None:
    runtime = InferenceRuntime(
        backend="cpu", device="cpu", precision="fp32", cuda_available=False
    )
    assert autocast_kwargs(runtime) == {"enabled": False, "device_type": "cpu"}


def test_runtime_serialization_contains_measured_fields() -> None:
    payload = detect_inference_runtime(prefer_gpu=False).to_dict()
    assert payload["device"] == "cpu"
    assert payload["precision"] == "fp32"
    assert payload["cuda_available"] is False
