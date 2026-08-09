from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from furniture_ai import cli


def synthetic_plan(path: Path) -> Path:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 570, 370), outline="black", width=12)
    draw.line((300, 30, 300, 370), fill="black", width=12)
    image.save(path)
    return path


def run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["furniture-ai", *argv])
    cli.main()


def test_cli_outputs_parseable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = synthetic_plan(tmp_path / "plan.png")
    run_cli(monkeypatch, [str(plan)])
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["floor_plan"]["source_width"] == 600
    assert isinstance(payload["placed_items"], int)
    assert isinstance(payload["warnings"], list)


def test_cli_missing_file_exits_with_code_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nonexistent" / "مخطط.png"
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, [str(missing)])
    assert excinfo.value.code == 2
    assert "cannot read image" in capsys.readouterr().err


def test_cli_rejects_non_image_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"this is not an image")
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, [str(junk)])
    assert excinfo.value.code == 2
    assert "not a valid image" in capsys.readouterr().err
