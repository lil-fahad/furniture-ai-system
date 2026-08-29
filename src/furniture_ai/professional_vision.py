from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from furniture_ai.contracts import (
    BoundingBox,
    RelativeDepthSummary,
    SceneAnalysis,
    SceneObject,
)


class ProfessionalVisionUnavailable(RuntimeError):
    pass


def _require_model_dir(root: Path, name: str) -> Path:
    model_dir = root / name
    required = (
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "preprocessor_config.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ProfessionalVisionUnavailable(
            f"Professional model {name!r} is not installed completely: missing {', '.join(missing)}"
        )
    return model_dir


def _resolve_device(requested: str | None = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise ProfessionalVisionUnavailable(
            "Install the professional extra to use local Hugging Face vision models"
        ) from exc
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ProfessionalVisionUnavailable("CUDA was requested but is not available")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_detector(model_dir: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        raise ProfessionalVisionUnavailable(
            "Install the professional extra to use DETR object detection"
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=4)
def _load_depth_model(model_dir: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise ProfessionalVisionUnavailable(
            "Install the professional extra to use Depth Anything V2"
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model


def _move_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def relative_depth_summary(depth: np.ndarray) -> RelativeDepthSummary:
    """Return stable per-image normalized percentiles for relative depth.

    Depth Anything V2 is a relative-depth model. Normalizing each image keeps
    the API bounded and explicit about the fact that values are not metric and
    are not directly comparable across different images.
    """
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Depth prediction contains no finite values")
    minimum = float(finite.min())
    maximum = float(finite.max())
    if maximum - minimum <= 1e-12:
        normalized = np.zeros_like(finite)
    else:
        normalized = (finite - minimum) / (maximum - minimum)
    p10, median, p90 = np.quantile(normalized, [0.1, 0.5, 0.9])
    return RelativeDepthSummary(p10=float(p10), median=float(median), p90=float(p90))


class ProfessionalVisionService:
    """Offline scene analysis using the verified local Hugging Face bundle."""

    DETECTOR_ID = "facebook/detr-resnet-50"
    DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, models_root: Path, *, device: str | None = None) -> None:
        self.models_root = Path(models_root)
        self.detector_dir = _require_model_dir(self.models_root, "detr_resnet50")
        self.depth_dir = _require_model_dir(self.models_root, "depth_anything_v2_small")
        self.device = _resolve_device(device)

    def _detect(self, image: Image.Image, threshold: float) -> list[SceneObject]:
        try:
            import torch
        except ImportError as exc:
            raise ProfessionalVisionUnavailable("PyTorch is required for object detection") from exc

        processor, model = _load_detector(str(self.detector_dir), self.device)
        inputs = _move_inputs(processor(images=image.convert("RGB"), return_tensors="pt"), self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        result = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        labels = getattr(model.config, "id2label", {})
        objects: list[SceneObject] = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"], strict=True
        ):
            label_id = int(label.detach().cpu().item())
            label_name = str(labels.get(label_id, labels.get(str(label_id), label_id)))
            x_min, y_min, x_max, y_max = [float(value) for value in box.detach().cpu().tolist()]
            x_min = min(max(x_min, 0.0), float(image.width))
            x_max = min(max(x_max, x_min), float(image.width))
            y_min = min(max(y_min, 0.0), float(image.height))
            y_max = min(max(y_max, y_min), float(image.height))
            objects.append(
                SceneObject(
                    label=label_name,
                    confidence=float(score.detach().cpu().item()),
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                )
            )
        return objects

    def _depth(self, image: Image.Image) -> RelativeDepthSummary:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise ProfessionalVisionUnavailable("PyTorch is required for depth estimation") from exc

        processor, model = _load_depth_model(str(self.depth_dir), self.device)
        inputs = _move_inputs(processor(images=image.convert("RGB"), return_tensors="pt"), self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        predicted = outputs.predicted_depth
        resized = functional.interpolate(
            predicted.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth = resized[0].detach().float().cpu().numpy()
        return relative_depth_summary(depth)

    def analyze(
        self,
        image: Image.Image,
        *,
        detection_threshold: float = 0.55,
        include_depth: bool = True,
    ) -> SceneAnalysis:
        if not 0 <= detection_threshold <= 1:
            raise ValueError("detection_threshold must be between 0 and 1")

        warnings: list[str] = []
        objects = self._detect(image, detection_threshold)
        depth: RelativeDepthSummary | None = None
        model_ids = [self.DETECTOR_ID]
        if include_depth:
            try:
                depth = self._depth(image)
                model_ids.append(self.DEPTH_ID)
            except (ProfessionalVisionUnavailable, RuntimeError, ValueError) as exc:
                warnings.append(f"Relative depth unavailable: {exc}")

        return SceneAnalysis(
            source_width=image.width,
            source_height=image.height,
            objects=objects,
            relative_depth=depth,
            model_ids=model_ids,
            warnings=warnings,
        )
