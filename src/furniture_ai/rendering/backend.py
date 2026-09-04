from __future__ import annotations

import base64
from html import escape
from typing import Protocol

from furniture_ai.rendering.contracts import (
    RenderArtifact,
    RendererKind,
    RenderPromptPackage,
    SceneSpec,
)


class RendererBackend(Protocol):
    kind: RendererKind
    photorealistic: bool

    def render(
        self,
        scene: SceneSpec,
        prompt: RenderPromptPackage,
        *,
        seed: int,
    ) -> RenderArtifact: ...


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
    ) -> RenderArtifact:
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

        title = escape(scene.style)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white" />'
            f'<text x="40" y="28" font-size="18" fill="#111">'
            f"Scene preview — {title}</text>"
            + "".join(room_shapes)
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


def get_renderer(kind: RendererKind) -> RendererBackend:
    if kind is RendererKind.MOCK:
        return DeterministicMockRenderer()
    raise ValueError(f"Unsupported renderer backend: {kind}")
