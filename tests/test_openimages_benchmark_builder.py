from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_openimages_builder_selects_real_target_schema(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    boxes = tmp_path / "boxes.csv"
    output = tmp_path / "manifest.json"

    classes.write_text(
        "/m/chair,Chair\n/m/couch,Couch\n/m/table,Table\n/m/lamp,Lamp\n",
        encoding="utf-8",
    )
    with boxes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ImageID",
                "Source",
                "LabelName",
                "Confidence",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
                "IsOccluded",
                "IsTruncated",
                "IsGroupOf",
                "IsDepiction",
                "IsInside",
            ]
        )
        writer.writerow(
            [
                "img1",
                "xclick",
                "/m/chair",
                "1",
                "0.1",
                "0.4",
                "0.2",
                "0.8",
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
        )
        writer.writerow(
            [
                "img1",
                "xclick",
                "/m/table",
                "1",
                "0.2",
                "0.8",
                "0.5",
                "0.9",
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
        )
        writer.writerow(
            [
                "img2",
                "xclick",
                "/m/lamp",
                "1",
                "0.1",
                "0.2",
                "0.1",
                "0.2",
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_openimages_furniture_benchmark.py",
            "--classes",
            str(classes),
            "--boxes",
            str(boxes),
            "--output",
            str(output),
            "--max-images-per-class",
            "1",
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"] == "Open Images V7"
    assert payload["split"] == "validation"
    assert payload["counts"]["Chair"] == 1
    assert payload["counts"]["Table"] == 1
    assert payload["counts"]["Couch"] == 0
    assert len(payload["images"]) == 1
    assert payload["images"][0]["image_id"] == "img1"
    labels = {row["label"] for row in payload["images"][0]["annotations"]}
    assert labels == {"Chair", "Table"}
