from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fetch_openverse_styles.py"
SPEC = importlib.util.spec_from_file_location("fetch_openverse_styles", MODULE_PATH)
assert SPEC and SPEC.loader
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


def test_normalized_jpeg_is_rgb_and_bounded() -> None:
    from PIL import Image
    from io import BytesIO

    source = Image.new("RGBA", (2_000, 1_000), (12, 34, 56, 128))
    payload = BytesIO()
    source.save(payload, format="PNG")

    normalized = fetcher.normalized_jpeg(payload.getvalue())
    with Image.open(BytesIO(normalized)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert max(image.size) == fetcher.OUTPUT_MAX_EDGE


def test_existing_openverse_ids_ignores_blank_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text('{"openverse_id": "first"}\n\n' + json.dumps({"openverse_id": "second"}), encoding="utf-8")

    assert fetcher.existing_openverse_ids(manifest) == {"first", "second"}


def test_make_record_uses_relative_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    record = fetcher.make_record(
        {"id": "record-id", "license": "cc0", "url": "https://example.test/image.jpg"},
        "minimalist",
        tmp_path / "data/styles/minimalist/example.jpg",
        "minimalist interior design living room",
    )

    assert record["openverse_id"] == "record-id"
    assert record["local_path"] == "data/styles/minimalist/example.jpg"
