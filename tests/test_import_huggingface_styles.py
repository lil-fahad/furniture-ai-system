from __future__ import annotations

import importlib.util
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "import_huggingface_styles.py"
SPEC = importlib.util.spec_from_file_location("import_huggingface_styles", MODULE_PATH)
assert SPEC and SPEC.loader
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def test_unlicensed_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="no declared license"):
        importer.require_allowed_license({"tags": ["modality:image"]}, ("cc0",))


def test_declared_license_must_be_allowed() -> None:
    metadata = {"cardData": {"license": "other"}, "tags": ["license:other"]}
    with pytest.raises(ValueError, match="outside --allowed-licenses"):
        importer.require_allowed_license(metadata, ("cc0",))


def test_declared_license_is_deduplicated() -> None:
    metadata = {"cardData": {"license": "CC-BY-4.0"}, "tags": ["license:cc-by-4.0"]}
    assert importer.declared_licenses(metadata) == ("cc-by-4.0",)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("A Scandinavian interior", "scandinavian"),
        ("mid-century modern living room", "mid_century_modern"),
        ("JAPANDI apartment", "japandi"),
        ("A photo of Modern style interior design", None),
        ("A photo of Japanese style interior design", None),
    ],
)
def test_style_mapping_is_conservative(label: str, expected: str | None) -> None:
    assert importer.style_from_text(label) == expected


def test_exact_user_mapping_can_handle_source_taxonomy() -> None:
    mapping = {importer.normalize_text("A photo of Japanese style interior design"): "japandi"}
    assert (
        importer.style_from_text("A photo of Japanese style interior design", mapping)
        == "japandi"
    )


def test_asset_url_rejects_untrusted_hosts() -> None:
    assert importer.safe_asset_url("https://datasets-server.huggingface.co/cached-assets/x")
    assert importer.safe_asset_url("http://datasets-server.huggingface.co/cached-assets/x") is None
    assert importer.safe_asset_url("https://example.test/image.jpg") is None


def test_normalized_jpeg_is_rgb_and_bounded() -> None:
    source = Image.new("RGBA", (2_000, 1_000), (12, 34, 56, 128))
    payload = BytesIO()
    source.save(payload, format="PNG")

    normalized = importer.normalized_jpeg(payload.getvalue())
    with Image.open(BytesIO(normalized)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert max(image.size) == importer.OUTPUT_MAX_EDGE


def test_existing_hashes_ignores_blank_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text('{"sha256": "first"}\n\n' + json.dumps({"sha256": "second"}))
    assert importer.existing_hashes(manifest) == {"first", "second"}
