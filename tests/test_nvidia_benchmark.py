from __future__ import annotations

from scripts.benchmark_nvidia_vision import percentile


def test_percentile_handles_empty_and_single_value() -> None:
    assert percentile([], 0.95) is None
    assert percentile([12.5], 0.95) == 12.5


def test_percentile_interpolates_sorted_values() -> None:
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 1.0) == 40.0
