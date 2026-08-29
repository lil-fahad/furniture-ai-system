from __future__ import annotations

import numpy as np
import pytest

from furniture_ai.professional_vision import (
    ProfessionalVisionService,
    ProfessionalVisionUnavailable,
    relative_depth_summary,
)


def test_relative_depth_summary_is_bounded_and_monotonic() -> None:
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    summary = relative_depth_summary(depth)
    assert 0 <= summary.p10 <= summary.median <= summary.p90 <= 1
    assert summary.p10 == pytest.approx(0.1, abs=0.02)
    assert summary.median == pytest.approx(0.5, abs=0.02)
    assert summary.p90 == pytest.approx(0.9, abs=0.02)
    assert "not physical dimensions" in summary.note


def test_relative_depth_constant_prediction_is_safe() -> None:
    summary = relative_depth_summary(np.full((4, 4), 7.0, dtype=np.float32))
    assert summary.p10 == 0
    assert summary.median == 0
    assert summary.p90 == 0


def test_relative_depth_rejects_non_finite_prediction() -> None:
    with pytest.raises(ValueError, match="finite"):
        relative_depth_summary(np.full((2, 2), np.nan, dtype=np.float32))


def test_professional_service_fails_cleanly_when_bundle_is_missing(tmp_path) -> None:
    with pytest.raises(ProfessionalVisionUnavailable, match="not installed completely"):
        ProfessionalVisionService(tmp_path)
