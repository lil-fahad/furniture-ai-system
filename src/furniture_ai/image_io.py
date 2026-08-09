from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from furniture_ai.config import Settings

ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}


class ImageValidationError(ValueError):
    pass


def load_validated_image(data: bytes, media_type: str | None, settings: Settings) -> Image.Image:
    """Validate and decode an uploaded image.

    The declared media type is only a hint (parameters after ``;`` are
    ignored); the actual bytes sniffed by PIL are authoritative. Pixel-count
    limits are enforced from the container dimensions immediately after
    ``Image.open`` — before any pixel data is decoded — so oversized images are
    rejected without a decompression window. All failures raise
    ``ImageValidationError`` (mapped to HTTP 422 by the API layer).
    """
    if not data:
        raise ImageValidationError("The uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError("The uploaded image exceeds the configured byte limit")
    if media_type:
        declared = media_type.split(";", 1)[0].strip().lower()
        if declared not in ALLOWED_MEDIA_TYPES:
            raise ImageValidationError("Only PNG, JPEG, and WebP images are accepted")

    # Keep PIL's own decompression guard aligned with the configured limit as
    # defense in depth; the explicit dimension check below rejects first.
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    try:
        # Single-pass open + decode: the pixel-count limit is enforced from the
        # container dimensions immediately after ``Image.open`` (before any
        # pixel data is decoded), and ``convert("RGB")`` below forces a full
        # decode that validates the payload — corrupt or truncated data raises
        # here, so the old probe/verify/reopen cycle is unnecessary.
        image = Image.open(BytesIO(data))
        if image.width * image.height > settings.max_image_pixels:
            image.close()
            raise ImageValidationError("The image dimensions exceed the configured pixel limit")
        image = image.convert("RGB")
    except ImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "The image dimensions exceed the configured pixel limit"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image") from exc

    if image.width < 64 or image.height < 64:
        raise ImageValidationError("The image is too small for floor-plan analysis")
    return image
