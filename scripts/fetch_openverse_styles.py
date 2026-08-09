#!/usr/bin/env python3
"""Build a small, reviewable interior-style dataset from Openverse.

The downloader intentionally defaults to CC0 records only.  It stores source
and license metadata beside the images, but Openverse metadata is not a legal
guarantee: review the manifest before any commercial use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError


API_URL = "https://api.openverse.org/v1/images/"
DEFAULT_STYLES = {
    "minimalist": "minimalist interior design living room",
    "scandinavian": "scandinavian interior design living room",
    "industrial": "industrial interior design living room",
    "bohemian": "bohemian interior design living room",
    "luxury": "luxury interior design living room",
    "mid_century_modern": "mid century modern interior design living room",
    "japandi": "japandi interior design living room",
}
DEFAULT_LICENSES = ("cc0",)
MAX_SOURCE_PIXELS = 25_000_000
OUTPUT_MAX_EDGE = 1_024
USER_AGENT = "furniture-ai-system-style-dataset/1.0"


def csv_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def style_names(raw: str) -> tuple[str, ...]:
    styles = csv_values(raw)
    unknown = sorted(set(styles) - set(DEFAULT_STYLES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown style(s): {', '.join(unknown)}")
    return styles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/styles"),
        help="ImageFolder root; one directory is created per style.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/style_sources.jsonl"),
        help="JSON Lines source, attribution, and license record.",
    )
    parser.add_argument(
        "--per-style",
        type=int,
        default=20,
        help="Target image count per style, including existing files (default: 20).",
    )
    parser.add_argument(
        "--styles",
        type=style_names,
        default=tuple(DEFAULT_STYLES),
        help="Comma-separated subset of styles.",
    )
    parser.add_argument(
        "--licenses",
        type=csv_values,
        default=DEFAULT_LICENSES,
        help="Openverse license codes, default: cc0. Example: cc0,by,by-sa.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Openverse candidates requested per style (1-50, default: 50).",
    )
    parser.add_argument(
        "--max-download-mb",
        type=int,
        default=15,
        help="Reject source files larger than this number of MiB (default: 15).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between Openverse searches (default: 1).",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search and report candidates without downloading or writing files.",
    )
    args = parser.parse_args()
    if args.per_style < 1:
        parser.error("--per-style must be at least 1")
    if not 1 <= args.page_size <= 50:
        parser.error("--page-size must be between 1 and 50")
    if args.max_download_mb < 1:
        parser.error("--max-download-mb must be at least 1")
    if args.pause < 0:
        parser.error("--pause cannot be negative")
    return args


def request_bytes(url: str, timeout: float, limit: int | None = None) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - URL comes from Openverse.
        if limit is None:
            return response.read()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > limit:
            raise ValueError(f"source is larger than {limit // (1024 * 1024)} MiB")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"source is larger than {limit // (1024 * 1024)} MiB")
    return payload


def openverse_results(query: str, licenses: Iterable[str], page_size: int, timeout: float) -> list[dict[str, Any]]:
    parameters = {
        "q": query,
        "license": ",".join(licenses),
        "category": "photograph",
        "mature": "false",
        "page_size": str(page_size),
    }
    payload = request_bytes(f"{API_URL}?{urlencode(parameters)}", timeout)
    response = json.loads(payload)
    results = response.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Openverse returned an unexpected response")
    return [item for item in results if isinstance(item, dict)]


def normalized_jpeg(payload: bytes) -> bytes:
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    try:
        with Image.open(BytesIO(payload)) as source:
            source.load()
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError("source image has too many pixels")
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            image.thumbnail((OUTPUT_MAX_EDGE, OUTPUT_MAX_EDGE), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc


def existing_openverse_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    ids: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {manifest} line {number}") from exc
        identifier = record.get("openverse_id")
        if isinstance(identifier, str):
            ids.add(identifier)
    return ids


def count_images(directory: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def output_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def safe_identifier(result: dict[str, Any]) -> str | None:
    identifier = result.get("id") or result.get("identifier")
    if not isinstance(identifier, str) or not identifier:
        return None
    return re.sub(r"[^A-Za-z0-9_-]", "_", identifier)


def source_url(result: dict[str, Any]) -> str | None:
    for key in ("url", "thumbnail"):
        value = result.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return None


def make_record(result: dict[str, Any], style: str, local_path: Path, query: str) -> dict[str, Any]:
    fields = (
        "title",
        "creator",
        "creator_url",
        "license",
        "license_version",
        "license_url",
        "attribution",
        "source",
        "foreign_landing_url",
        "url",
        "thumbnail",
    )
    record = {field: result.get(field) for field in fields}
    record.update(
        {
            "openverse_id": result.get("id") or result.get("identifier"),
            "style": style,
            "query": query,
            "local_path": output_path(local_path),
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
    )
    return record


def append_manifest(manifest: Path, record: dict[str, Any]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    seen_ids = existing_openverse_ids(args.manifest)
    max_download_bytes = args.max_download_mb * 1024 * 1024
    total_downloaded = 0

    for position, style in enumerate(args.styles):
        query = DEFAULT_STYLES[style]
        style_directory = args.target / style
        if not args.dry_run:
            style_directory.mkdir(parents=True, exist_ok=True)
        current = count_images(style_directory) if style_directory.exists() else 0
        needed = max(0, args.per_style - current)
        print(f"{style}: {current}/{args.per_style} images; query={query!r}", flush=True)
        if needed == 0:
            continue

        try:
            candidates = openverse_results(query, args.licenses, args.page_size, args.timeout)
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f"  search failed: {exc}", file=sys.stderr, flush=True)
            continue

        accepted = 0
        for result in candidates:
            identifier = safe_identifier(result)
            url = source_url(result)
            if identifier is None or url is None or identifier in seen_ids:
                continue
            if args.dry_run:
                print(
                    f"  candidate {identifier}: license={result.get('license')} title={result.get('title')!r}",
                    flush=True,
                )
                accepted += 1
                if accepted >= min(needed, 5):
                    break
                continue

            filename = f"{style}_{current + accepted + 1:04d}_{identifier[:16]}.jpg"
            local_path = style_directory / filename
            try:
                jpeg = normalized_jpeg(request_bytes(url, args.timeout, max_download_bytes))
                local_path.write_bytes(jpeg)
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                print(f"  skipped {identifier}: {exc}", file=sys.stderr, flush=True)
                continue
            append_manifest(args.manifest, make_record(result, style, local_path, query))
            seen_ids.add(identifier)
            accepted += 1
            total_downloaded += 1
            print(f"  downloaded {local_path.name}", flush=True)
            if accepted >= needed:
                break

        if accepted < needed:
            print(
                f"  found {accepted}/{needed} new images. Review results and rerun later if needed.",
                file=sys.stderr,
                flush=True,
            )
        if position < len(args.styles) - 1 and args.pause:
            time.sleep(args.pause)

    if args.dry_run:
        print("Dry run complete: no images or manifest records were written.")
    else:
        print(f"Downloaded {total_downloaded} images. Review {args.manifest} before training.")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
