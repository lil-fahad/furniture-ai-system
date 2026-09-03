"""Reproducible evaluation utilities for FurnitureAI models and pipelines."""

from furniture_ai.evaluation.detection import (
    DetectionEvaluationReport,
    DetectionRecord,
    GroundTruthRecord,
    NormalizedBox,
    evaluate_detections,
)

__all__ = [
    "DetectionEvaluationReport",
    "DetectionRecord",
    "GroundTruthRecord",
    "NormalizedBox",
    "evaluate_detections",
]
