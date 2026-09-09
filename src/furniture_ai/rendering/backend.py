from __future__ import annotations

import base64
import binascii
import shutil
import subprocess
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from furniture_ai.config import Settings
from furniture_ai.rendering.contracts import (
    RenderArtifact,
    RenderGenerationConfig,
    RendererKind,
    RenderPromptPackage,
    SceneSpec,
)

MAX_RENDER_BYTES = 25 * 1024 * 1024
_OPENART_MODEL_BY_KIND = {
    RendererKind.OPENART_GPT_IMAGE_25_FLARE: "gpt-image-2-5-flare",
    RendererKind.OPENART_GPT_IMAGE_25_SUNBURST: "gpt-image-2-5-sunburst",
}


class RenderBackendUnavailable(RuntimeError):
    """Raised when a requested renderer cannot be used in the current environment."""


class RenderBackendError(RuntimeError):
    """Raised for sanitized external-renderer failures."""


class RendererBackend(Protocol):
    kind: RendererKind
    photorealistic: bool

    def render(
        self,
        scene: SceneSpec,
        prompt: RenderPromptPackage,
        *,
        seed: int,
        generation: RenderGenerationConfig,
    ) -> RenderArtifact: ...


class OpenArtCommandRunner(Protocol):
    def run(self, args: list[str], *, timeout: float) -> None: ...


class SubprocessOpenArtRunner:
    """Execute the official OpenArt CLI without invoking a shell."""

    def run(self, args: list[str], *, timeout: float) -> None:
        try:
            subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise RenderBackendUnavailable("OpenArt CLI is unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise RenderBackendError("OpenArt image generation timed out") from exc
        except subprocess.CalledProcessError as exc:
            raise RenderBackendError("OpenArt image generation failed") from exc
        except OSError as exc:
            raise RenderBackendUnavailable("OpenArt CLI is unavailable") from exc


def _render_prompt(prompt: RenderPromptPackage) -> str:
    return (
        f"{prompt.positive_prompt}\n\n"
        f"Hard constraints: {prompt.negative_prompt}. "
        "Render one finished photorealistic interior image only."
    )


def _verified_png(
    image_bytes: bytes,
    *,
    max_pixels: int,
) -> tuple[str, int, int]:
    if not image_bytes or len(image_bytes) > MAX_RENDER_BYTES:
        raise ValueError("Image output size is invalid")

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError("Image output dimensions are invalid")
        normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        buffer = BytesIO()
        normalized.save(buffer, format="PNG")

    png_bytes = buffer.getvalue()
    if not png_bytes or len(png_bytes) > MAX_RENDER_BYTES:
        raise ValueError("Normalized image output size is invalid")
    return base64.b64encode(png_bytes).decode("ascii"), width, height


def _resolve_openart_cli(cli_path: str) -> str | None:
    candidate = Path(cli_path).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate) if candidate.is_file() else None
    return shutil.which(cli_path)


class DeterministicMockRenderer:
    """Produce a deterministic top-down SVG preview without claiming photorealism."""

    kind = RendererKind.MOCK
    photorealistic = False

    def render(
        self,
        scene: SceneSpec,
        prompt: RenderPromptPackage,
        *,
        seed: int,
        generation: RenderGenerationConfig,
    ) -> RenderArtifact:
        del generation
        width, height, margin = 1024, 768, 40.0
        scale = min(
            (width - 2 * margin) / scene.source_width,
            (height - 2 * margin) / scene.source_height,
        )

        def sx(value: float) -> float:
            return margin + value * scale

        def sy(value: float) -> float:
            return margin + value * scale

        room_shapes: list[str] = []
        furniture_shapes: list[str] = []
        opening_shapes: list[str] = []
        for room in scene.rooms:
            points = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in room.polygon)
            room_shapes.append(
                f'<polygon points="{points}" fill="#f7f7f7" stroke="#222" stroke-width="3" />'
            )
            for item in room.furniture:
                item_width = item.width * scale
                item_depth = item.depth * scale
                center_x = sx(item.center.x)
                center_y = sy(item.center.y)
                x = center_x - item_width / 2
                y = center_y - item_depth / 2
                label = escape(item.product_name)
                rotation = (
                    f"rotate({item.rotation_degrees:.2f} "
                    f"{center_x:.2f} {center_y:.2f})"
                )
                furniture_shapes.append(
                    f'<g transform="{rotation}">'
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{item_width:.2f}" '
                    f'height="{item_depth:.2f}" rx="6" fill="#dedede" stroke="#555" '
                    f'stroke-width="2" />'
                    f'<text x="{center_x:.2f}" y="{center_y:.2f}" text-anchor="middle" '
                    f'dominant-baseline="middle" font-size="14" fill="#111">{label}</text>'
                    "</g>"
                )

        for opening in scene.openings:
            opening_shapes.append(
                f'<line x1="{sx(opening.start.x):.2f}" y1="{sy(opening.start.y):.2f}" '
                f'x2="{sx(opening.end.x):.2f}" y2="{sy(opening.end.y):.2f}" '
                f'stroke="#1677ff" stroke-width="7" stroke-linecap="round" '
                f'data-kind="{escape(opening.kind.value)}" />'
            )

        title = escape(scene.style)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white" />'
            f'<text x="40" y="28" font-size="18" fill="#111">'
            f"Scene preview — {title}</text>"
            + "".join(room_shapes)
            + "".join(opening_shapes)
            + "".join(furniture_shapes)
            + "</svg>"
        )
        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return RenderArtifact(
            backend=self.kind,
            media_type="image/svg+xml",
            data_uri=f"data:image/svg+xml;base64,{encoded}",
            width=width,
            height=height,
            metadata={
                "seed": seed,
                "scene_fingerprint": prompt.scene_fingerprint,
                "preview_kind": "top_down_grounding_preview",
            },
        )


class OpenAIGPTImageRenderer:
    """Generate a photorealistic room image through OpenAI's GPT-Image-2 Image API."""

    kind = RendererKind.OPENAI_GPT_IMAGE_2
    photorealistic = True

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if not settings.openai_configured:
            raise RenderBackendUnavailable("Photorealistic renderer is not configured")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RenderBackendUnavailable("Photorealistic renderer is unavailable") from exc
            key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
            client = OpenAI(
                api_key=key,
                timeout=settings.openai_image_timeout_seconds,
                max_retries=1,
            )

        self.client = client
        self.model = settings.openai_image_model
        self.quality = settings.openai_image_quality
        self.size = settings.openai_image_size
        self.max_pixels = settings.max_image_pixels

    def render(
        self,
        scene: SceneSpec,
        prompt: RenderPromptPackage,
        *,
        seed: int,
        generation: RenderGenerationConfig,
    ) -> RenderArtifact:
        del scene, generation
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=_render_prompt(prompt),
                quality=self.quality,
                size=self.size,
                background="opaque",
            )
        except Exception as exc:
            raise RenderBackendError("Photorealistic image generation failed") from exc

        try:
            data = response.data
            raw_encoded = data[0].b64_json.strip()
            image_bytes = base64.b64decode(raw_encoded, validate=True)
            encoded, width, height = _verified_png(
                image_bytes,
                max_pixels=self.max_pixels,
            )
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            binascii.Error,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise RenderBackendError("Photorealistic renderer returned an invalid image") from exc

        return RenderArtifact(
            backend=self.kind,
            media_type="image/png",
            data_uri=f"data:image/png;base64,{encoded}",
            width=width,
            height=height,
            metadata={
                "model": self.model,
                "quality": self.quality,
                "requested_size": self.size,
                "scene_fingerprint": prompt.scene_fingerprint,
                "seed_requested": seed,
                "seed_applied": False,
                "reference_images_applied": 0,
            },
        )


class OpenArtGPTImageRenderer:
    """Render GPT Image 2.5 through the authenticated official OpenArt CLI."""

    photorealistic = True

    def __init__(
        self,
        settings: Settings,
        *,
        kind: RendererKind,
        runner: OpenArtCommandRunner | None = None,
    ) -> None:
        if kind not in _OPENART_MODEL_BY_KIND:
            raise ValueError(f"Unsupported OpenArt renderer backend: {kind}")
        if not settings.openart_configured:
            raise RenderBackendUnavailable("OpenArt renderer is not configured")

        cli_path = settings.openart_cli_path
        if runner is None:
            resolved = _resolve_openart_cli(cli_path)
            if resolved is None:
                raise RenderBackendUnavailable("OpenArt CLI is unavailable")
            cli_path = resolved
            runner = SubprocessOpenArtRunner()

        self.kind = kind
        self.model = _OPENART_MODEL_BY_KIND[kind]
        self.cli_path = cli_path
        self.runner = runner
        self.timeout = settings.openart_image_timeout_seconds
        self.max_pixels = settings.max_image_pixels

    @staticmethod
    def _validate_supported_controls(generation: RenderGenerationConfig) -> None:
        if generation.quality != "medium":
            raise RenderBackendUnavailable(
                "OpenArt CLI integration currently guarantees the medium quality tier only"
            )
        if not generation.lock_aspect_ratio:
            raise RenderBackendUnavailable(
                "OpenArt CLI integration currently requires a locked aspect ratio"
            )
        if generation.auto_enhance_prompt:
            raise RenderBackendUnavailable(
                "OpenArt CLI integration does not enable provider-side prompt enhancement"
            )

    def _command(
        self,
        prompt: RenderPromptPackage,
        generation: RenderGenerationConfig,
        output_dir: Path,
    ) -> list[str]:
        args = [
            self.cli_path,
            "generate",
            "image",
            _render_prompt(prompt),
            "--model",
            self.model,
            "--aspect-ratio",
            generation.aspect_ratio,
            "--resolution",
            generation.resolution_tier,
            "--timeout",
            str(int(self.timeout)),
            "--json",
            "-o",
            str(output_dir),
        ]
        for reference in generation.visual_references:
            args.extend(["--image", reference.url])
        return args

    @staticmethod
    def _find_generated_image(output_dir: Path) -> Path:
        allowed = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        if len(candidates) != 1:
            raise RenderBackendError("OpenArt renderer returned an invalid image set")
        return candidates[0]

    def render(
        self,
        scene: SceneSpec,
        prompt: RenderPromptPackage,
        *,
        seed: int,
        generation: RenderGenerationConfig,
    ) -> RenderArtifact:
        del scene
        self._validate_supported_controls(generation)
        try:
            with TemporaryDirectory(prefix="furniture-openart-") as temp_dir:
                output_dir = Path(temp_dir)
                self.runner.run(
                    self._command(prompt, generation, output_dir),
                    timeout=self.timeout,
                )
                output_path = self._find_generated_image(output_dir)
                image_bytes = output_path.read_bytes()
                encoded, width, height = _verified_png(
                    image_bytes,
                    max_pixels=self.max_pixels,
                )
        except (RenderBackendUnavailable, RenderBackendError):
            raise
        except (ValueError, UnidentifiedImageError, OSError) as exc:
            raise RenderBackendError("OpenArt renderer returned an invalid image") from exc

        return RenderArtifact(
            backend=self.kind,
            media_type="image/png",
            data_uri=f"data:image/png;base64,{encoded}",
            width=width,
            height=height,
            metadata={
                "provider": "openart_cli",
                "model": self.model,
                "mode": generation.mode,
                "quality": generation.quality,
                "resolution_tier": generation.resolution_tier,
                "aspect_ratio": generation.aspect_ratio,
                "scene_fingerprint": prompt.scene_fingerprint,
                "seed_requested": seed,
                "seed_applied": False,
                "reference_images_applied": len(generation.visual_references),
            },
        )


def get_renderer(
    kind: RendererKind,
    *,
    settings: Settings | None = None,
    client: Any | None = None,
) -> RendererBackend:
    if kind is RendererKind.MOCK:
        return DeterministicMockRenderer()
    if kind is RendererKind.OPENAI_GPT_IMAGE_2:
        if settings is None:
            raise RenderBackendUnavailable("Photorealistic renderer is not configured")
        return OpenAIGPTImageRenderer(settings, client=client)
    if kind in _OPENART_MODEL_BY_KIND:
        if settings is None:
            raise RenderBackendUnavailable("OpenArt renderer is not configured")
        return OpenArtGPTImageRenderer(settings, kind=kind, runner=client)
    raise ValueError(f"Unsupported renderer backend: {kind}")
