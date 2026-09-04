from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from furniture_ai.config import Settings
from furniture_ai.image_io import ImageValidationError, load_validated_image


def _image_bytes(width: int, height: int, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format=image_format)
    return buffer.getvalue()


def _png_bytes(width: int, height: int) -> bytes:
    return _image_bytes(width, height, "PNG")


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


def test_actual_unsupported_format_rejected_even_when_declared_png(settings: Settings) -> None:
    gif = _image_bytes(128, 128, "GIF")

    with pytest.raises(ImageValidationError, match="supported PNG, JPEG, or WebP"):
        load_validated_image(gif, "image/png", settings)


def test_actual_unsupported_format_rejected_without_declared_type(settings: Settings) -> None:
    gif = _image_bytes(128, 128, "GIF")

    with pytest.raises(ImageValidationError, match="supported PNG, JPEG, or WebP"):
        load_validated_image(gif, None, settings)


def test_multiframe_allowed_format_is_rejected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MultiFramePng:
        format = "PNG"
        n_frames = 2
        width = 128
        height = 128

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: MultiFramePng())

    with pytest.raises(ImageValidationError, match="multi-frame"):
        load_validated_image(_png_bytes(128, 128), "image/png", settings)


def test_oversized_pixels_rejected_before_decode(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_pillow_open_bomb_error_also_mapped(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(Image, "open", bomb)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(_png_bytes(128, 128), "image/png", settings)


def test_validation_does_not_mutate_pillow_global_limit(settings: Settings) -> None:
    original_limit = Image.MAX_IMAGE_PIXELS
    limited = settings.model_copy(update={"max_image_pixels": 1_000_000})

    load_validated_image(_png_bytes(128, 128), "image/png", limited)

    assert original_limit == Image.MAX_IMAGE_PIXELS


def test_garbage_bytes_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError):
        load_validated_image(b"not an image at all", "image/png", settings)


def test_too_small_image_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError, match="too small"):
        load_validated_image(_png_bytes(32, 32), None, settings)


def test_empty_upload_rejected(settings: Settings) -> None:
    with pytest.raises(ImageValidationError, match="empty"):
        load_validated_image(b"", None, settings)
