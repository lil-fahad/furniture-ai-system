from __future__ import annotations

from furniture_ai.evaluation.detection import (
    DetectionRecord,
    GroundTruthRecord,
    NormalizedBox,
    evaluate_detections,
    intersection_over_union,
)


def box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBox:
    return NormalizedBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)


def test_iou_exact_match_is_one() -> None:
    candidate = box(0.1, 0.1, 0.4, 0.4)
    assert intersection_over_union(candidate, candidate) == 1.0


def test_perfect_detection_report_is_perfect() -> None:
    truth = [GroundTruthRecord("img-1", "Chair", box(0.1, 0.1, 0.4, 0.4))]
    predictions = [
        DetectionRecord("img-1", "Chair", 0.9, box(0.1, 0.1, 0.4, 0.4))
    ]
    report = evaluate_detections(truth, predictions)
    assert report.true_positives == 1
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.micro_precision == 1.0
    assert report.micro_recall == 1.0
    assert report.map_at_iou == 1.0


def test_duplicate_detection_counts_as_false_positive() -> None:
    truth = [GroundTruthRecord("img-1", "Chair", box(0.1, 0.1, 0.4, 0.4))]
    predictions = [
        DetectionRecord("img-1", "Chair", 0.9, box(0.1, 0.1, 0.4, 0.4)),
        DetectionRecord("img-1", "Chair", 0.8, box(0.1, 0.1, 0.4, 0.4)),
    ]
    report = evaluate_detections(truth, predictions)
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.micro_precision == 0.5
    # Ranked AP remains perfect because the correct detection appears first.
    assert report.map_at_iou == 1.0


def test_wrong_class_does_not_match_ground_truth() -> None:
    truth = [GroundTruthRecord("img-1", "Chair", box(0.1, 0.1, 0.4, 0.4))]
    predictions = [
        DetectionRecord("img-1", "Table", 0.9, box(0.1, 0.1, 0.4, 0.4))
    ]
    report = evaluate_detections(truth, predictions)
    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.micro_f1 == 0.0


def test_iou_threshold_controls_matching() -> None:
    truth = [GroundTruthRecord("img-1", "Chair", box(0.0, 0.0, 0.5, 0.5))]
    predictions = [
        DetectionRecord("img-1", "Chair", 0.9, box(0.2, 0.0, 0.7, 0.5))
    ]
    loose = evaluate_detections(truth, predictions, iou_threshold=0.4)
    strict = evaluate_detections(truth, predictions, iou_threshold=0.6)
    assert loose.true_positives == 1
    assert strict.true_positives == 0


def test_invalid_threshold_fails_closed() -> None:
    try:
        evaluate_detections([], [], iou_threshold=0)
    except ValueError as exc:
        assert "iou_threshold" in str(exc)
    else:
        raise AssertionError("invalid IoU threshold must fail")
