from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedBox:
    """Axis-aligned box in normalized image coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Bounding-box coordinates must be normalized to [0, 1]")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding-box maximums must exceed minimums")


@dataclass(frozen=True)
class GroundTruthRecord:
    image_id: str
    label: str
    box: NormalizedBox

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not self.label.strip():
            raise ValueError("Ground-truth image_id and label must be non-empty")


@dataclass(frozen=True)
class DetectionRecord:
    image_id: str
    label: str
    score: float
    box: NormalizedBox

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not self.label.strip():
            raise ValueError("Detection image_id and label must be non-empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be finite and within [0, 1]")


@dataclass(frozen=True)
class DetectionClassMetrics:
    label: str
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    average_precision: float
    mean_matched_iou: float | None


@dataclass(frozen=True)
class DetectionEvaluationReport:
    iou_threshold: float
    classes: tuple[DetectionClassMetrics, ...]
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    map_at_iou: float | None


def intersection_over_union(left: NormalizedBox, right: NormalizedBox) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    intersection = width * height
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _average_precision(tp_flags: list[int], fp_flags: list[int], gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for true_positive, false_positive in zip(tp_flags, fp_flags, strict=True):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / gt_count)
        precisions.append(_safe_ratio(cumulative_tp, cumulative_tp + cumulative_fp))
    recalls.append(1.0)
    precisions.append(0.0)

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    area = 0.0
    for index in range(1, len(recalls)):
        if recalls[index] != recalls[index - 1]:
            area += (recalls[index] - recalls[index - 1]) * precisions[index]
    return area


def _evaluate_class(
    label: str,
    ground_truth: list[GroundTruthRecord],
    detections: list[DetectionRecord],
    iou_threshold: float,
) -> DetectionClassMetrics:
    gt_by_image: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    for record in ground_truth:
        gt_by_image[record.image_id].append(record)

    matched: dict[str, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    matched_ious: list[float] = []

    ranked = sorted(
        detections,
        key=lambda item: (-item.score, item.image_id, item.box.x_min, item.box.y_min),
    )
    for detection in ranked:
        candidates = gt_by_image.get(detection.image_id, [])
        best_index: int | None = None
        best_iou = 0.0
        for index, truth in enumerate(candidates):
            if index in matched[detection.image_id]:
                continue
            candidate_iou = intersection_over_union(detection.box, truth.box)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched[detection.image_id].add(best_index)
            tp_flags.append(1)
            fp_flags.append(0)
            matched_ious.append(best_iou)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    true_positives = sum(tp_flags)
    false_positives = sum(fp_flags)
    false_negatives = max(0, len(ground_truth) - true_positives)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, len(ground_truth))
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else None
    return DetectionClassMetrics(
        label=label,
        ground_truth=len(ground_truth),
        predictions=len(detections),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        average_precision=_average_precision(tp_flags, fp_flags, len(ground_truth)),
        mean_matched_iou=mean_iou,
    )


def evaluate_detections(
    ground_truth: list[GroundTruthRecord],
    detections: list[DetectionRecord],
    *,
    iou_threshold: float = 0.5,
) -> DetectionEvaluationReport:
    """Evaluate ranked detections with one-to-one class/image matching.

    This is a deterministic FurnitureAI benchmark metric, not an implementation
    of the full Open Images challenge evaluator. Group-of semantics and other
    dataset-specific policies must be handled during benchmark manifest build.
    """
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and within (0, 1]")

    gt_by_label: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    detections_by_label: dict[str, list[DetectionRecord]] = defaultdict(list)
    for record in ground_truth:
        gt_by_label[record.label].append(record)
    for detection in detections:
        detections_by_label[detection.label].append(detection)

    labels = sorted(set(gt_by_label) | set(detections_by_label))
    class_metrics = tuple(
        _evaluate_class(
            label,
            gt_by_label.get(label, []),
            detections_by_label.get(label, []),
            iou_threshold,
        )
        for label in labels
    )
    true_positives = sum(item.true_positives for item in class_metrics)
    false_positives = sum(item.false_positives for item in class_metrics)
    false_negatives = sum(item.false_negatives for item in class_metrics)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    ap_values = [item.average_precision for item in class_metrics if item.ground_truth > 0]
    return DetectionEvaluationReport(
        iou_threshold=iou_threshold,
        classes=class_metrics,
        ground_truth=len(ground_truth),
        predictions=len(detections),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=_f1(precision, recall),
        map_at_iou=sum(ap_values) / len(ap_values) if ap_values else None,
    )
