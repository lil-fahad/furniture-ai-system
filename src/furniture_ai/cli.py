from __future__ import annotations

import argparse
import json
from pathlib import Path

from furniture_ai.config import get_settings
from furniture_ai.image_io import load_validated_image
from furniture_ai.pipeline import DesignPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Furniture AI design pipeline")
    parser.add_argument("image", type=Path)
    parser.add_argument("--pixels-per-cm", type=float, default=None)
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--preferences", default="")
    args = parser.parse_args()

    settings = get_settings()
    image = load_validated_image(args.image.read_bytes(), None, settings)
    result = DesignPipeline(settings).run(
        image,
        pixels_per_cm=args.pixels_per_cm,
        use_openai=args.openai,
        preferences=args.preferences,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
