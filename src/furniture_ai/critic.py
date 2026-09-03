from __future__ import annotations

from dataclasses import dataclass

from furniture_ai.constraints import validate_layout_constraints
from furniture_ai.contracts import DesignResult, LayoutValidationReport


class DesignCriticRejected(RuntimeError):
    """Raised when a generated design violates deterministic hard constraints."""

    def __init__(self, report: LayoutValidationReport) -> None:
        self.report = report
        codes = sorted({issue.code for issue in report.issues})
        summary = ", ".join(codes) if codes else "unknown constraint failure"
        super().__init__(f"Generated layout failed spatial validation: {summary}")


@dataclass(frozen=True)
class SpatialDesignCritic:
    """Independent deterministic critic for generated furniture layouts.

    The default clearance remains zero. This critic enforces only geometry rules
    whose values are already present in the layout; sourced code/user-specific
    clearances can be supplied explicitly by future policy layers.
    """

    minimum_clearance: float = 0.0

    def inspect(self, result: DesignResult) -> LayoutValidationReport:
        return validate_layout_constraints(
            result.floor_plan,
            minimum_clearance=self.minimum_clearance,
        )

    def require_valid(self, result: DesignResult) -> DesignResult:
        report = self.inspect(result)
        if not report.valid:
            raise DesignCriticRejected(report)
        return result
