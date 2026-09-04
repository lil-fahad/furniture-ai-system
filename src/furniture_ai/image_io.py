from __future__ import annotations

import warnings
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from furniture_ai.config import Settings

ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}


class ImageValidationError(ValueError):
    pass


def load_validated_image(data: bytes, media_type: str | None, settings: Settings) -> Image.Image:
    """Validate and decode one still PNG, JPEG, or WebP image.

    The declared media type is only a hint (parameters after ``;`` are
    ignored); the format sniffed from the bytes is authoritative. Pixel-count
    limits are enforced from container dimensions immediately after
    ``Image.open`` and before pixel decoding. The function never mutates
    Pillow's process-global decompression settings, so concurrent requests with
    different Settings objects cannot change each other's validation policy.
    """
    if not data:
        raise ImageValidationError("The uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError("The uploaded image exceeds the configured byte limit")
    if media_type:
        declared = media_type.split(";", 1)[0].strip().lower()
        if declared not in ALLOWED_MEDIA_TYPES:
            raise ImageValidationError("Only PNG, JPEG, and WebP images are accepted")

    try:
        # Pillow's default bomb threshold is process-global. Do not rewrite it
        # per request. Its warning can fire below FurnitureAI's configured
        # maximum (which is capped at 100M pixels), so suppress only the warning
        # locally and enforce the application limit explicitly before decode.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                actual_format = (source.format or "").upper()
                if actual_format not in ALLOWED_IMAGE_FORMATS:
                    raise ImageValidationError(
                        "The uploaded bytes are not a supported PNG, JPEG, or WebP image"
                    )
                if getattr(source, "n_frames", 1) != 1:
                    raise ImageValidationError("Animated or multi-frame images are not accepted")
                if source.width * source.height > settings.max_image_pixels:
                    raise ImageValidationError(
                        "The image dimensions exceed the configured pixel limit"
                    )
                # ``convert`` forces a full decode, validating corrupt or
                # truncated payloads, and returns a detached image before the
                # source decoder is deterministically closed by the context.
                image = source.convert("RGB")
    except ImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "The image dimensions exceed the configured pixel limit"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image") from exc

    if image.width < 64 or image.height < 64:
        image.close()
        raise ImageValidationError("The image is too small for floor-plan analysis")
    return image
