#!/usr/bin/env python3
"""Import a licensed Hugging Face image dataset into a style ImageFolder.

The importer uses the read-only Dataset Viewer API and refuses datasets whose
Hub card does not declare a license. Labels are mapped conservatively: broad
labels such as ``Modern`` or ``Japanese`` are not silently relabeled as one of
the project's seven narrower styles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

HUB_API = "https://huggingface.co/api/datasets"
VIEWER_API = "https://datasets-server.huggingface.co"
USER_AGENT = "furniture-ai-system-hf-importer/1.0"
ALLOWED_ASSET_HOSTS = {
    "datasets-server.huggingface.co",
    "cdn-lfs.huggingface.co",
    "huggingface.co",
}
DEFAULT_ALLOWED_LICENSES = (
    "apache-2.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0",
    "cc0-1.0",
    "mit",
    "odc-pddl",
    "pddl",
)
STYLES = (
    "minimalist",
    "scandinavian",
    "industrial",
    "bohemian",
    "luxury",
    "mid_century_modern",
    "japandi",
)
STYLE_TERMS = {
    "minimalist": ("minimalist", "minimalism"),
    "scandinavian": ("scandinavian", "nordic"),
    "industrial": ("industrial",),
    "bohemian": ("bohemian", "boho"),
    "luxury": ("luxury", "luxurious"),
    "mid_century_modern": ("mid century modern", "mid-century modern"),
    "japandi": ("japandi",),
}
MAX_SOURCE_PIXELS = 30_000_000
OUTPUT_MAX_EDGE = 1_024


def csv_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def request_bytes(
    url: str,
    timeout: float,
    limit: int | None = None,
    allowed_hosts: set[str] | None = None,
) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - callers validate hosts.
        final_host = urlparse(response.geturl()).hostname
        if allowed_hosts is not None and final_host not in allowed_hosts:
            raise ValueError("download redirected to an untrusted host")
        if limit is None:
            return response.read()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > limit:
            raise ValueError(f"source is larger than {limit // (1024 * 1024)} MiB")
        payload = response.read(limit + 1)
    if limit is not None and len(payload) > limit:
        raise ValueError(f"source is larger than {limit // (1024 * 1024)} MiB")
    return payload


def request_json(url: str, timeout: float) -> dict[str, Any]:
    payload = json.loads(request_bytes(url, timeout))
    if not isinstance(payload, dict):
        raise ValueError("Hugging Face returned an unexpected response")
    return payload


def dataset_metadata(dataset_id: str, timeout: float) -> dict[str, Any]:
    return request_json(f"{HUB_API}/{quote(dataset_id, safe='/')}", timeout)


def declared_licenses(metadata: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    card_data = metadata.get("cardData")
    if isinstance(card_data, dict):
        declared = card_data.get("license")
        if isinstance(declared, str):
            values.append(declared)
        elif isinstance(declared, list):
            values.extend(value for value in declared if isinstance(value, str))
    tags = metadata.get("tags")
    if isinstance(tags, list):
        values.extend(
            tag.split(":", 1)[1]
            for tag in tags
            if isinstance(tag, str) and tag.startswith("license:")
        )
    return tuple(dict.fromkeys(value.strip().lower() for value in values if value.strip()))


def require_allowed_license(
    metadata: dict[str, Any], allowed_licenses: tuple[str, ...]
) -> tuple[str, ...]:
    licenses = declared_licenses(metadata)
    if not licenses:
        raise ValueError(
            "Dataset card has no declared license; import refused before downloading images"
        )
    disallowed = sorted(set(licenses) - set(allowed_licenses))
    if disallowed:
        raise ValueError(
            "Dataset license is outside --allowed-licenses: " + ", ".join(disallowed)
        )
    return licenses


def viewer_splits(dataset_id: str, timeout: float) -> list[dict[str, Any]]:
    query = urlencode({"dataset": dataset_id})
    response = request_json(f"{VIEWER_API}/splits?{query}", timeout)
    splits = response.get("splits", [])
    if not isinstance(splits, list) or not splits:
        raise ValueError("Dataset Viewer has no ready subsets/splits for this dataset")
    return [item for item in splits if isinstance(item, dict)]


def resolve_subset_split(
    splits: list[dict[str, Any]], subset: str | None, split: str | None
) -> tuple[str, str]:
    available = {
        (str(item.get("config")), str(item.get("split")))
        for item in splits
        if item.get("config") is not None and item.get("split") is not None
    }
    if subset is not None and split is not None:
        if (subset, split) not in available:
            raise ValueError(
                f"Unknown subset/split {subset!r}/{split!r}; available: {sorted(available)}"
            )
        return subset, split
    if len(available) == 1:
        return next(iter(available))
    train_options = sorted(pair for pair in available if pair[1] == "train")
    if subset is None and split is None and len(train_options) == 1:
        return train_options[0]
    raise ValueError(
        "Dataset has multiple subsets/splits; pass both --subset and --split. "
        f"Available: {sorted(available)}"
    )


def load_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--label-map must contain a JSON object")
    mapping: dict[str, str] = {}
    for source, target in payload.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("--label-map keys and values must be strings")
        if target not in STYLES:
            raise ValueError(f"Unknown target style in --label-map: {target}")
        mapping[normalize_text(source)] = target
    return mapping


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def style_from_text(value: object, label_map: dict[str, str] | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_text(value)
    mapping = label_map or {}
    if normalized in mapping:
        return mapping[normalized]
    for style, terms in STYLE_TERMS.items():
        if any(normalize_text(term) in normalized for term in terms):
            return style
    return None


def safe_asset_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_ASSET_HOSTS:
        return None
    return value


def normalized_jpeg(payload: bytes) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                if source.width * source.height > MAX_SOURCE_PIXELS:
                    raise ValueError("source image has too many pixels")
                source.load()
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
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(f"invalid image: {exc}") from exc


def existing_hashes(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    hashes: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {manifest} line {number}") from exc
        digest = record.get("sha256")
        if isinstance(digest, str):
            hashes.add(digest)
    return hashes


def append_manifest(manifest: Path, record: dict[str, Any]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Public Hugging Face dataset id (namespace/name)")
    parser.add_argument("--subset")
    parser.add_argument("--split")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="text")
    parser.add_argument(
        "--label-map",
        type=Path,
        help="JSON object mapping exact source labels to the project's seven style names",
    )
    parser.add_argument("--target", type=Path, default=Path("data/styles"))
    parser.add_argument("--manifest", type=Path, default=Path("data/style_sources.jsonl"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-download-mb", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allowed-licenses",
        type=csv_values,
        default=DEFAULT_ALLOWED_LICENSES,
        help="Declared dataset licenses accepted by policy",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be at least 1")
    if not 1 <= args.page_size <= 100:
        parser.error("--page-size must be between 1 and 100")
    if args.max_download_mb < 1:
        parser.error("--max-download-mb must be at least 1")
    return args


def run(args: argparse.Namespace) -> int:
    metadata = dataset_metadata(args.dataset, args.timeout)
    licenses = require_allowed_license(metadata, args.allowed_licenses)
    subset, split = resolve_subset_split(
        viewer_splits(args.dataset, args.timeout), args.subset, args.split
    )
    label_map = load_label_map(args.label_map)
    revision = metadata.get("sha")
    max_download_bytes = args.max_download_mb * 1024 * 1024
    seen_hashes = existing_hashes(args.manifest)
    offset = imported = matched = skipped = 0
    total_rows: int | None = None
    print(
        f"dataset={args.dataset} revision={revision} license={','.join(licenses)} "
        f"subset={subset} split={split}",
        flush=True,
    )

    while total_rows is None or offset < total_rows:
        if args.max_rows is not None and offset >= args.max_rows:
            break
        length = args.page_size
        if args.max_rows is not None:
            length = min(length, args.max_rows - offset)
        query = urlencode(
            {
                "dataset": args.dataset,
                "config": subset,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        response = request_json(f"{VIEWER_API}/rows?{query}", args.timeout)
        rows = response.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("Dataset Viewer returned an unexpected rows response")
        total_rows = int(response.get("num_rows_total", len(rows)))
        if not rows:
            break

        for item in rows:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                skipped += 1
                continue
            row = item["row"]
            source_label = row.get(args.label_column)
            style = style_from_text(source_label, label_map)
            image_value = row.get(args.image_column)
            image_url = (
                safe_asset_url(image_value.get("src")) if isinstance(image_value, dict) else None
            )
            if style is None or image_url is None:
                skipped += 1
                continue
            matched += 1
            row_index = int(item.get("row_idx", offset))
            if args.dry_run:
                if matched <= 10:
                    print(f"candidate row={row_index} style={style} label={source_label!r}")
                continue
            try:
                jpeg = normalized_jpeg(
                    request_bytes(
                        image_url,
                        args.timeout,
                        limit=max_download_bytes,
                        allowed_hosts=ALLOWED_ASSET_HOSTS,
                    )
                )
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                print(f"skipped row={row_index}: {exc}", file=sys.stderr, flush=True)
                skipped += 1
                continue
            digest = hashlib.sha256(jpeg).hexdigest()
            if digest in seen_hashes:
                skipped += 1
                continue
            style_directory = args.target / style
            style_directory.mkdir(parents=True, exist_ok=True)
            local_path = style_directory / f"hf_{row_index:07d}_{digest[:12]}.jpg"
            temporary = local_path.with_suffix(".jpg.part")
            temporary.write_bytes(jpeg)
            temporary.replace(local_path)
            append_manifest(
                args.manifest,
                {
                    "source": "huggingface",
                    "dataset_id": args.dataset,
                    "dataset_revision": revision,
                    "dataset_license": list(licenses),
                    "subset": subset,
                    "split": split,
                    "row_index": row_index,
                    "source_label": source_label,
                    "style": style,
                    "sha256": digest,
                    "local_path": relative_path(local_path),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                },
            )
            seen_hashes.add(digest)
            imported += 1
        offset += len(rows)
        print(
            f"progress={min(offset, total_rows)}/{total_rows} "
            f"matched={matched} imported={imported}",
            flush=True,
        )

    action = "would import" if args.dry_run else "imported"
    print(f"{action}={matched if args.dry_run else imported} skipped={skipped}", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
