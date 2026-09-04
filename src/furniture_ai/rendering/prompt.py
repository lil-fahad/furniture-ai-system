from __future__ import annotations

import hashlib
import json

from furniture_ai.rendering.contracts import RenderPromptPackage, SceneSpec


class PromptCompiler:
    """Compile a deterministic, grounding-first prompt package from a SceneSpec."""

    def compile(self, scene: SceneSpec) -> RenderPromptPackage:
        room_lines: list[str] = []
        reference_urls: set[str] = set()
        for room in scene.rooms:
            item_lines: list[str] = []
            for item in room.furniture:
                item_lines.append(
                    f"{item.product_name} ({item.category}) at "
                    f"({item.center.x:.2f}, {item.center.y:.2f}), "
                    f"footprint {item.width:.2f} x {item.depth:.2f}, "
                    f"rotation {item.rotation_degrees:.1f} degrees"
                )
                if item.reference_url:
                    reference_urls.add(item.reference_url)
            furniture_text = "; ".join(item_lines) if item_lines else "no grounded furniture"
            room_lines.append(
                f"Room {room.id}: type={room.room_type}, area={room.area:.2f}, "
                f"grounded furniture=[{furniture_text}]"
            )

        positive_prompt = (
            "Photorealistic interior architecture photograph. "
            f"Design style: {scene.style}. "
            f"Camera: {scene.camera.preset}, {scene.camera.lens_mm:.1f}mm lens, "
            f"camera height {scene.camera.height_cm:.1f}cm. "
            "Preserve the supplied room geometry and every grounded furniture placement exactly. "
            "Keep furniture scale, orientation, and product identity consistent with the scene. "
            "Use physically plausible materials, realistic daylight and practical interior "
            "lighting, natural shadows, correct perspective, and high-detail architectural "
            "photography. "
            + " ".join(room_lines)
        )
        negative_prompt = "; ".join(scene.negative_constraints)
        canonical = json.dumps(
            scene.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        return RenderPromptPackage(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_urls=sorted(reference_urls),
            scene_fingerprint=fingerprint,
        )
