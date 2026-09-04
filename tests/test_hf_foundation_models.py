from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.transforms import InterpolationMode

from training import local_resumable_classifier as trainer


class TinyClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(2, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


def test_adaptive_transforms_use_backbone_preprocessing() -> None:
    preprocessing = {
        "image_size": 224,
        "native_image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "crop_pct": 0.9,
        "interpolation": "bicubic",
    }
    training, evaluation = trainer.build_adaptive_transforms("style", preprocessing)

    crop = training.transforms[0]
    normalize = training.transforms[-1]
    resize = evaluation.transforms[0]

    assert crop.size == (224, 224)
    assert crop.interpolation == InterpolationMode.BICUBIC
    assert list(normalize.mean) == [0.5, 0.5, 0.5]
    assert list(normalize.std) == [0.5, 0.5, 0.5]
    assert resize.size == 249


def test_pretrained_failure_is_fail_closed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(trainer.timm, "create_model", fail)
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        trainer.build_model("hf_hub:example/model", 12, True, False)


def test_random_initialization_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def create_model(_name: str, *, pretrained: bool, num_classes: int):
        nonlocal calls
        calls += 1
        if pretrained:
            raise OSError("offline")
        return TinyClassifier(num_classes)

    monkeypatch.setattr(trainer.timm, "create_model", create_model)
    model = trainer.build_model("hf_hub:example/model", 12, True, True)

    assert isinstance(model, TinyClassifier)
    assert calls == 2


def test_preprocessing_signature_changes_with_backbone_contract() -> None:
    first = {
        "image_size": 224,
        "native_image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "crop_pct": 0.9,
        "interpolation": "bicubic",
    }
    second = dict(first)
    second["mean"] = [0.485, 0.456, 0.406]

    assert trainer.preprocessing_signature(first) != trainer.preprocessing_signature(second)
