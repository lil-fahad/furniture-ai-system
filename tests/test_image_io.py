from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from furniture_ai.config import Settings
from furniture_ai.image_io import ImageValidationError, load_validated_image


def _png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def settings() -> Settings:
    return Settings(environment="test")


def test_valid_image_loads(settings: Settings) -> None:
    image = load_validated_image(_png_bytes(128, 128), "image/png", settings)
    assert image.size == (128, 128)
    assert image.mode == "RGB"


def test_media_type_parameters_are_tolerated(settings: Settings) -> None:
    image = load_validated_image(_png_bytes(128, 128), "image/png; charset=binary", settings)
    assert image.size == (128, 128)


def test_disallowed_media_type_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError):
        load_validated_image(_png_bytes(128, 128), "image/gif", settings)


def test_oversized_pixels_rejected_before_decode(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1500x1000 = 1.5M pixels > the 1M floor allowed by Settings validation,
    # but below PIL's hard error threshold (2x MAX), so the explicit
    # pre-decode dimension check is what rejects it.
    limited = settings.model_copy(update={"max_image_pixels": 1_000_000})
    data = _png_bytes(1500, 1000)

    decoded = False
    real_convert = Image.Image.convert

    def spy_convert(self, *args, **kwargs):
        nonlocal decoded
        decoded = True
        return real_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", spy_convert)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(data, "image/png", limited)
    assert decoded is False, "pixel data must not be decoded before the limit check"


def test_decompression_bomb_error_mapped_to_validation_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bomb(self, *args, **kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(Image.Image, "convert", bomb)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(_png_bytes(128, 128), "image/png", settings)


def test_pil_hard_bomb_error_also_mapped(settings: Settings) -> None:
    # 3000x3000 = 9M pixels > 2x the configured 1M limit: PIL itself raises
    # DecompressionBombError at open time and it must surface as 422, not 500.
    limited = settings.model_copy(update={"max_image_pixels": 1_000_000})
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(_png_bytes(3000, 3000), "image/png", limited)


def test_max_image_pixels_aligned_with_settings(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_image_pixels": 1_000_000})
    load_validated_image(_png_bytes(128, 128), "image/png", limited)
    assert Image.MAX_IMAGE_PIXELS == 1_000_000


def test_garbage_bytes_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError):
        load_validated_image(b"not an image at all", "image/png", settings)


def test_too_small_image_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError, match="too small"):
        load_validated_image(_png_bytes(32, 32), None, settings)


def test_empty_upload_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError, match="empty"):
        load_validated_image(b"", None, settings)
