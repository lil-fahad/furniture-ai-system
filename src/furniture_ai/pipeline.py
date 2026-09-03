from __future__ import annotations

from PIL import Image

from furniture_ai.config import Settings
from furniture_ai.contracts import DesignResult
from furniture_ai.critic import SpatialDesignCritic
from furniture_ai.floorplan import FloorPlanAnalyzer
from furniture_ai.layout import furnish_floor_plan
from furniture_ai.openai_service import OpenAIDesignService, OpenAIUnavailable


class DesignPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = FloorPlanAnalyzer()
        self.critic = SpatialDesignCritic()

    def run(
        self,
        image: Image.Image,
        *,
        pixels_per_cm: float | None = None,
        use_openai: bool = False,
        preferences: str = "",
    ) -> DesignResult:
        floor_plan = self.analyzer.analyze(image, pixels_per_cm=pixels_per_cm)
        service: OpenAIDesignService | None = None
        if use_openai:
            try:
                service = OpenAIDesignService(self.settings)
                refinements = service.refine_room_types(image, floor_plan)
                for room in floor_plan.rooms:
                    if room.id in refinements:
                        room.room_type, room.confidence = refinements[room.id]
                floor_plan.analysis_method += "+openai-vision"
            except (OpenAIUnavailable, ValueError, RuntimeError) as exc:
                floor_plan.warnings.append(f"OpenAI refinement unavailable: {exc}")

        # Generated geometry crosses an independent deterministic trust boundary
        # before any optional design brief is requested or returned to callers.
        result = self.critic.require_valid(furnish_floor_plan(floor_plan))
        if use_openai and preferences and service is not None:
            try:
                result.design_brief = service.create_design_brief(result.floor_plan, preferences)
            except (ValueError, RuntimeError) as exc:
                result.warnings.append(f"OpenAI design brief unavailable: {exc}")
        return result
