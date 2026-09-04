from __future__ import annotations

import pytest
from pydantic import ValidationError

from furniture_ai.config import Settings


def test_nvidia_vision_defaults_are_auto_and_compile_off() -> None:
    settings = Settings(environment="test")
    assert settings.professional_vision_device == "auto"
    assert settings.professional_vision_precision == "auto"
    assert settings.professional_vision_torch_compile is False


def test_nvidia_vision_device_normalizes_cuda_index() -> None:
    settings = Settings(environment="test", professional_vision_device=" CUDA:2 ")
    assert settings.professional_vision_device == "cuda:2"


def test_nvidia_vision_device_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError, match="PROFESSIONAL_VISION_DEVICE"):
        Settings(environment="test", professional_vision_device="gpu")


def test_nvidia_vision_precision_is_strict_literal() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="test", professional_vision_precision="fp8")
