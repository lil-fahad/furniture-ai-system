from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from furniture_ai.config import Settings

ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}


class ImageValidationError(ValueError):
    pass


def load_validated_image(data: bytes, media_type: str | None, settings: Settings) -> Image.Image:
    if not data:
        raise ImageValidationError("The uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError("The uploaded image exceeds the configured byte limit")
    if media_type and media_type.lower() not in ALLOWED_MEDIA_TYPES:
        raise ImageValidationError("Only PNG, JPEG, and WebP images are accepted")

    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image") from exc

    if image.width * image.height > settings.max_image_pixels:
        raise ImageValidationError("The image dimensions exceed the configured pixel limit")
    if image.width < 64 or image.height < 64:
        raise ImageValidationError("The image is too small for floor-plan analysis")
    return image
