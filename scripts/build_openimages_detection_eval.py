from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from furniture_ai.evaluation.openimages import (
    OPENIMAGES_VALIDATION_BBOX_URL,
    OpenImagesBenchmarkPolicy,
    build_openimages_furniture_ground_truth,
    write_benchmark_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance-tracked FurnitureAI detection benchmark from an "
            "Open Images bounding-box CSV."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-url",
        default=None,
        help=(
            "Optional provenance URL for the exact local annotations file. "
            f"Official validation URL: {OPENIMAGES_VALIDATION_BBOX_URL}"
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Optional expected SHA-256; mismatch fails closed.",
    )
    parser.add_argument("--include-group-of", action="store_true")
    parser.add_argument("--include-depictions", action="store_true")
    args = parser.parse_args(argv)

    records, metadata = build_openimages_furniture_ground_truth(
        args.annotations,
        policy=OpenImagesBenchmarkPolicy(
            include_group_of=args.include_group_of,
            include_depictions=args.include_depictions,
        ),
        source_url=args.source_url,
        expected_sha256=args.expected_sha256,
    )
    manifest_path, metadata_path = write_benchmark_manifest(records, metadata, args.output)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "metadata": str(metadata_path),
                "summary": asdict(metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
