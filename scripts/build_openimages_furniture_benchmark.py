from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TARGET_NAMES = {"Chair", "Couch", "Table"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real Open Images V7 furniture benchmark manifest from official CSVs."
    )
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images-per-class", type=int, default=200)
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def main() -> None:
    args = parse_args()
    if args.max_images_per_class < 1:
        raise ValueError("--max-images-per-class must be positive")

    class_map = load_class_map(args.classes)
    target_ids = {mid: name for mid, name in class_map.items() if name in TARGET_NAMES}
    missing = TARGET_NAMES.difference(target_ids.values())
    if missing:
        raise RuntimeError(f"target classes missing from class map: {sorted(missing)}")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    selected_ids: dict[str, set[str]] = {name: set() for name in TARGET_NAMES}

    with args.boxes.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Open Images box CSV is missing required columns")

        for row in reader:
            label_name = row["LabelName"]
            if label_name not in target_ids:
                continue
            display_name = target_ids[label_name]
            image_id = row["ImageID"]
            if (
                image_id not in selected_ids[display_name]
                and len(selected_ids[display_name]) >= args.max_images_per_class
            ):
                continue
            selected_ids[display_name].add(image_id)
            grouped[image_id].append(
                {
                    "label": display_name,
                    "label_id": label_name,
                    "box": [
                        float(row["XMin"]),
                        float(row["YMin"]),
                        float(row["XMax"]),
                        float(row["YMax"]),
                    ],
                    "is_occluded": int(row.get("IsOccluded", "0") or 0),
                    "is_truncated": int(row.get("IsTruncated", "0") or 0),
                    "is_group_of": int(row.get("IsGroupOf", "0") or 0),
                }
            )

    manifest = [
        {
            "image_id": image_id,
            "split": "validation",
            "image": f"images/{image_id}.jpg",
            "source": "open-images-v7",
            "annotations": annotations,
        }
        for image_id, annotations in sorted(grouped.items())
    ]

    output = {
        "schema_version": 1,
        "dataset": "Open Images V7",
        "split": "validation",
        "target_classes": sorted(TARGET_NAMES),
        "images": manifest,
        "counts": {
            name: sum(
                1
                for item in manifest
                if any(annotation["label"] == name for annotation in item["annotations"])
            )
            for name in sorted(TARGET_NAMES)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["counts"], indent=2))


if __name__ == "__main__":
    main()
