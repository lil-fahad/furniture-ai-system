from __future__ import annotations

from typing import Any

from furniture_ai.config import Settings
from furniture_ai.rendering.backend import get_renderer
from furniture_ai.rendering.contracts import RenderPreviewRequest, RenderPreviewResult
from furniture_ai.rendering.prompt import PromptCompiler
from furniture_ai.rendering.scene import SceneCompiler


class RenderingService:
    """Renderer-neutral orchestration entrypoint used by the API and future job workers."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        renderer_client: Any | None = None,
        scene_compiler: SceneCompiler | None = None,
        prompt_compiler: PromptCompiler | None = None,
    ) -> None:
        self.settings = settings
        self.renderer_client = renderer_client
        self.scene_compiler = scene_compiler or SceneCompiler()
        self.prompt_compiler = prompt_compiler or PromptCompiler()

    def preview(self, request: RenderPreviewRequest) -> RenderPreviewResult:
        scene = self.scene_compiler.compile(
            request.design,
            style=request.style,
            room_id=request.room_id,
        )
        prompt = self.prompt_compiler.compile(scene)
        renderer = get_renderer(
            request.backend,
            settings=self.settings,
            client=self.renderer_client,
        )
        artifact = renderer.render(scene, prompt, seed=request.seed)
        warnings: list[str] = []
        if not renderer.photorealistic:
            warnings.append(
                "The selected backend is a deterministic grounding preview, "
                "not a photorealistic renderer."
            )
        elif prompt.reference_urls:
            warnings.append(
                "Product reference URLs are recorded for provenance but are not yet sent as "
                "image inputs; this render is grounded by scene geometry and catalog text."
            )
        return RenderPreviewResult(
            photorealistic=renderer.photorealistic,
            scene=scene,
            prompt=prompt,
            artifact=artifact,
            warnings=warnings,
        )
