from __future__ import annotations

import argparse
import json
from pathlib import Path

from furniture_ai.config import get_settings
from furniture_ai.image_io import ImageValidationError, load_validated_image
from furniture_ai.pipeline import DesignPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Furniture AI design pipeline")
    parser.add_argument("image", type=Path)
    parser.add_argument("--pixels-per-cm", type=float, default=None)
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--preferences", default="")
    args = parser.parse_args()

    settings = get_settings()
    try:
        file_size = args.image.stat().st_size
    except OSError as exc:
        parser.error(f"cannot read image {args.image}: {exc.strerror or exc}")
        return  # unreachable; parser.error exits
    if file_size > settings.max_upload_bytes:
        parser.error(
            f"image {args.image} is {file_size} bytes, exceeding the configured "
            f"limit of {settings.max_upload_bytes} bytes"
        )
    try:
        data = args.image.read_bytes()
    except OSError as exc:
        parser.error(f"cannot read image {args.image}: {exc.strerror or exc}")
        return  # unreachable; parser.error exits
    try:
        image = load_validated_image(data, None, settings)
    except ImageValidationError as exc:
        parser.error(str(exc))
        return  # unreachable; parser.error exits
    result = DesignPipeline(settings).run(
        image,
        pixels_per_cm=args.pixels_per_cm,
        use_openai=args.openai,
        preferences=args.preferences,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
