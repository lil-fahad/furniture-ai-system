from __future__ import annotations

from furniture_ai.rendering.backend import get_renderer
from furniture_ai.rendering.contracts import RenderPreviewRequest, RenderPreviewResult
from furniture_ai.rendering.prompt import PromptCompiler
from furniture_ai.rendering.scene import SceneCompiler


class RenderingService:
    """Renderer-neutral orchestration entrypoint used by the API and future job workers."""

    def __init__(
        self,
        *,
        scene_compiler: SceneCompiler | None = None,
        prompt_compiler: PromptCompiler | None = None,
    ) -> None:
        self.scene_compiler = scene_compiler or SceneCompiler()
        self.prompt_compiler = prompt_compiler or PromptCompiler()

    def preview(self, request: RenderPreviewRequest) -> RenderPreviewResult:
        scene = self.scene_compiler.compile(
            request.design,
            style=request.style,
            room_id=request.room_id,
        )
        prompt = self.prompt_compiler.compile(scene)
        renderer = get_renderer(request.backend)
        artifact = renderer.render(scene, prompt, seed=request.seed)
        warnings: list[str] = []
        if not renderer.photorealistic:
            warnings.append(
                "The selected backend is a deterministic grounding preview, "
                "not a photorealistic renderer."
            )
        return RenderPreviewResult(
            photorealistic=renderer.photorealistic,
            scene=scene,
            prompt=prompt,
            artifact=artifact,
            warnings=warnings,
        )
