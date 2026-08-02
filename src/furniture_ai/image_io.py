from __future__ import annotations

import warnings
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from furniture_ai.config import Settings

ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}
_MEDIA_TYPE_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidationError(ValueError):
    pass


def load_validated_image(data: bytes, media_type: str | None, settings: Settings) -> Image.Image:
    if not data:
        raise ImageValidationError("The uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError("The uploaded image exceeds the configured byte limit")

    declared_type = media_type.lower() if media_type else None
    if declared_type and declared_type not in ALLOWED_MEDIA_TYPES:
        raise ImageValidationError("Only PNG, JPEG, and WebP images are accepted")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                actual_type = _MEDIA_TYPE_BY_FORMAT.get(probe.format or "")
                if actual_type is None:
                    raise ImageValidationError("Only PNG, JPEG, and WebP images are accepted")
                if declared_type and actual_type != declared_type:
                    raise ImageValidationError(
                        "The declared media type does not match the image data"
                    )
                width, height = probe.size
                if width * height > settings.max_image_pixels:
                    raise ImageValidationError(
                        "The image dimensions exceed the configured pixel limit"
                    )
                if width < 64 or height < 64:
                    raise ImageValidationError("The image is too small for floor-plan analysis")
                probe.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The image dimensions exceed the safe pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image") from exc
    return image
