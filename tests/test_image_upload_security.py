"""Enhanced security tests for image upload validation."""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from furniture_ai.api import app
from furniture_ai.config import Settings, get_settings
from furniture_ai.image_io import ImageValidationError, load_validated_image


class TestUploadSizeLimit:
    """Test that upload size limits prevent DoS attacks."""

    def test_oversized_upload_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that files exceeding max_upload_bytes are rejected with 413."""
        # Set a small limit for testing
        monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")  # 1 KB
        get_settings.cache_clear()

        # Create a fake image larger than 1 KB
        fake_large_data = b"X" * 2048  # 2 KB

        with pytest.raises(ImageValidationError, match="exceeds the configured byte limit"):
            settings = get_settings()
            load_validated_image(fake_large_data, "image/png", settings)

    def test_endpoint_rejects_oversized_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test /analyze endpoint returns 413 for files exceeding limit."""
        monkeypatch.setenv("MAX_UPLOAD_BYTES", "512")  # 512 bytes
        get_settings.cache_clear()

        # Create a valid PNG larger than 512 bytes
        img = Image.new("RGB", (200, 200), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        oversized_data = buf.getvalue()

        assert len(oversized_data) > 512, "Test image must exceed 512 bytes"

        client = TestClient(app)
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("large.png", oversized_data, "image/png")},
            data={"use_openai": "false", "preferences": ""},
        )
        # Should be 413 (Payload Too Large) or 422 (validation error)
        assert response.status_code in [413, 422], f"Got {response.status_code}: {response.text}"


class TestMimeTypeValidation:
    """Test MIME type validation prevents invalid file uploads."""

    def test_text_file_rejected_by_mime_check(self) -> None:
        """Verify that text files are rejected even if renamed to .png."""
        text_data = b"This is not an image!"
        settings = get_settings()

        with pytest.raises(ImageValidationError, match="not a valid image"):
            load_validated_image(text_data, "text/plain", settings)

    def test_empty_file_rejected(self) -> None:
        """Verify that empty uploads are rejected."""
        settings = get_settings()

        with pytest.raises(ImageValidationError, match="empty"):
            load_validated_image(b"", "image/png", settings)

    def test_invalid_media_type_header_rejected(self) -> None:
        """Verify that disallowed media types (declared) are rejected."""
        settings = get_settings()

        # Valid PNG bytes, but declared as WebP (which is not in ALLOWED_MEDIA_TYPES)
        img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Temporarily patch allowed types to exclude WebP for this test
        with patch("furniture_ai.image_io.ALLOWED_MEDIA_TYPES", {"image/png", "image/jpeg"}):
            # If we declare it as WebP, it should be rejected at the media_type check
            with pytest.raises(ImageValidationError, match="Only PNG, JPEG"):
                load_validated_image(png_bytes, "image/webp", settings)


class TestPixelLimitValidation:
    """Test that oversized images (by pixel count) are rejected."""

    def test_oversized_image_by_pixels_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify images exceeding max_image_pixels are rejected."""
        # Set a very low pixel limit
        monkeypatch.setenv("MAX_IMAGE_PIXELS", "1000")  # 1000 pixels total
        get_settings.cache_clear()
        settings = get_settings()

        # Create a 100x100 image = 10,000 pixels (exceeds limit)
        img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        with pytest.raises(ImageValidationError, match="dimensions exceed"):
            load_validated_image(image_bytes, "image/png", settings)

    def test_minimum_image_size_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that images smaller than 64x64 are rejected."""
        settings = get_settings()

        # Create a 32x32 image (below minimum)
        img = Image.new("RGB", (32, 32), color="yellow")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        small_image_bytes = buf.getvalue()

        with pytest.raises(ImageValidationError, match="too small"):
            load_validated_image(small_image_bytes, "image/png", settings)


class TestValidImageAcceptance:
    """Test that legitimate images are accepted."""

    def test_valid_png_accepted(self) -> None:
        """Verify that valid PNG images are accepted."""
        settings = get_settings()

        img = Image.new("RGB", (256, 256), color="purple")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        valid_png = buf.getvalue()

        result = load_validated_image(valid_png, "image/png", settings)
        assert result.width == 256
        assert result.height == 256
        assert result.mode == "RGB"

    def test_valid_jpeg_accepted(self) -> None:
        """Verify that valid JPEG images are accepted."""
        settings = get_settings()

        img = Image.new("RGB", (200, 200), color="orange")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        valid_jpeg = buf.getvalue()

        result = load_validated_image(valid_jpeg, "image/jpeg", settings)
        assert result.width == 200
        assert result.height == 200
        assert result.mode == "RGB"

    def test_no_media_type_uses_sniffing(self) -> None:
        """Verify that PIL sniffing works when media_type is None."""
        settings = get_settings()

        img = Image.new("RGB", (100, 100), color="cyan")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # When media_type is None, PIL should sniff and accept valid images
        result = load_validated_image(png_bytes, None, settings)
        assert result.width == 100
        assert result.height == 100


class TestEndpointSecurityIntegration:
    """Integration tests for the /analyze endpoint security."""

    def test_health_endpoint_accessible_without_auth(self) -> None:
        """Verify /health does not require API key."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_ready_endpoint_accessible_without_auth(self) -> None:
        """Verify /ready does not require API key."""
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code in [200, 503]  # 503 if deps missing is OK

    def test_analyze_endpoint_accepts_valid_image(self) -> None:
        """Integration test: /analyze accepts a valid image."""
        img = Image.new("RGB", (256, 256), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        valid_image = buf.getvalue()

        client = TestClient(app)
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.png", valid_image, "image/png")},
            data={"use_openai": "false", "preferences": ""},
        )
        # 200 if model available, 503 if dependencies missing (still OK)
        assert response.status_code in [200, 503], f"Got {response.status_code}: {response.text}"

    def test_analyze_endpoint_rejects_missing_image(self) -> None:
        """Integration test: /analyze rejects request without image."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/analyze",
            data={"use_openai": "false", "preferences": ""},
        )
        assert response.status_code == 422  # Unprocessable entity
