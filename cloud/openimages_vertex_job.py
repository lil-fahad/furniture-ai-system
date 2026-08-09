#!/usr/bin/env python3
"""Build a licensed Open Images style dataset and train it on Vertex AI.

The job deliberately keeps source selection, weak-label provenance, images,
manifests, and the resulting checkpoint together under one private GCS prefix.
Open Images object annotations are used only to find likely indoor scenes. A
pinned SigLIP model supplies weak style labels; those labels are not a
substitute for a human-reviewed validation and test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

OPENIMAGES_VERSION = "v7"
OPENIMAGES_PAGE = "https://storage.googleapis.com/openimages/web/download_v7.html"
CLASS_DESCRIPTIONS_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
BOX_ANNOTATIONS_URL = (
    "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
)
TRAIN_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
)
IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
ALLOWED_DOWNLOAD_HOSTS = {
    "open-images-dataset.s3.amazonaws.com",
    "storage.googleapis.com",
}
USER_AGENT = "furniture-ai-system-openimages/1.0"

SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SIGLIP_LICENSE = "apache-2.0"

STYLES = (
    "minimalist",
    "scandinavian",
    "industrial",
    "bohemian",
    "luxury",
    "mid_century_modern",
    "japandi",
)
STYLE_PROMPTS = (
    "This is a photo of a minimalist interior room with clean lines, neutral colors, "
    "open space, and very little decoration.",
    "This is a photo of a Scandinavian interior room with pale wood, soft natural "
    "light, practical furniture, and light neutral colors.",
    "This is a photo of an industrial interior room with exposed brick or concrete, "
    "dark metal, visible structure, and loft-style furniture.",
    "This is a photo of a bohemian interior room with layered textiles, plants, warm "
    "earthy colors, handcrafted objects, and eclectic furniture.",
    "This is a photo of a luxury interior room with premium materials, elegant lighting, "
    "refined finishes, and sophisticated furniture.",
    "This is a photo of a mid-century modern interior room with 1950s-inspired furniture, "
    "tapered legs, warm wood, simple geometry, and retro accents.",
    "This is a photo of a Japandi interior room combining Japanese simplicity with "
    "Scandinavian warmth, natural wood, calm colors, and minimal furniture.",
)
NON_INTERIOR_PROMPT = (
    "This is not an interior room: it is an outdoor scene, a drawing, a person, or an "
    "isolated furniture product without a designed room around it."
)

FURNITURE_CLASSES = (
    "Bed",
    "Bookcase",
    "Cabinetry",
    "Chair",
    "Closet",
    "Coffee table",
    "Couch",
    "Desk",
    "Drawer",
    "Fireplace",
    "Kitchen & dining room table",
    "Lamp",
    "Refrigerator",
    "Shelf",
    "Sink",
    "Table",
    "Wardrobe",
)

CANONICAL_LICENSES = {
    "http://creativecommons.org/licenses/by/2.0/": "https://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/licenses/by/2.0/": "https://creativecommons.org/licenses/by/2.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/": (
        "https://creativecommons.org/publicdomain/zero/1.0/"
    ),
    "https://creativecommons.org/publicdomain/zero/1.0/": (
        "https://creativecommons.org/publicdomain/zero/1.0/"
    ),
}

IMAGE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
MAX_SOURCE_PIXELS = 40_000_000
OUTPUT_MAX_EDGE = 1_024
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class SourceRecord:
    image_id: str
    license_url: str
    landing_url: str
    author: str
    title: str


@dataclass(frozen=True)
class StylePrediction:
    style: str
    confidence: float
    margin: float
    non_interior_probability: float


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def clean_text(value: object, limit: int = 1_000) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value if character >= " " or character == "\t")[
        :limit
    ]


def normalize_bucket_name(value: str) -> str:
    name = value[5:] if value.startswith("gs://") else value
    name = name.rstrip("/")
    if not BUCKET_RE.fullmatch(name) or ".." in name:
        raise ValueError("--bucket must be a valid private Cloud Storage bucket name")
    return name


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError("--run-id must contain 3-63 lowercase letters, numbers, or hyphens")
    return value


def canonical_license(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return CANONICAL_LICENSES.get(value.strip().lower())


def validated_image_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if IMAGE_ID_RE.fullmatch(normalized) else None


def safe_source_url(image_id: str) -> str:
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError("unsafe Open Images identifier")
    return IMAGE_URL_TEMPLATE.format(image_id=image_id)


def safe_request(url: str, *, headers: Mapping[str, str] | None = None) -> Request:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"download host is not allowlisted: {parsed.hostname!r}")
    request_headers = {"Accept-Encoding": "identity", "User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    return Request(url, headers=request_headers)


def download_file(url: str, destination: Path, *, max_bytes: int, timeout: float) -> Path:
    """Download one trusted metadata file atomically, resuming a partial transfer."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and 0 < destination.stat().st_size <= max_bytes:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(safe_request(url, headers=headers), timeout=timeout) as response:  # nosec B310
                final = urlparse(response.geturl())
                if final.scheme != "https" or final.hostname not in ALLOWED_DOWNLOAD_HOSTS:
                    raise ValueError("metadata download redirected to an untrusted host")
                append = offset > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                written = offset if append else 0
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(f"metadata exceeds the {max_bytes} byte safety limit")
                        handle.write(chunk)
            os.replace(partial, destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def load_furniture_class_ids(path: Path) -> dict[str, int]:
    by_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                by_name[row[1].strip().casefold()] = row[0].strip()
    matched = {
        by_name[name.casefold()]: position
        for position, name in enumerate(FURNITURE_CLASSES)
        if name.casefold() in by_name
    }
    if len(matched) < 8:
        raise ValueError(
            "Open Images class metadata changed unexpectedly; fewer than eight furniture "
            "classes were found"
        )
    return matched


def candidate_ids_from_rows(
    rows: Iterable[Mapping[str, object]],
    class_ids: Mapping[str, int],
    *,
    min_distinct_classes: int,
    min_boxes: int,
) -> set[str]:
    states: dict[str, tuple[int, int]] = {}
    for row in rows:
        image_id = validated_image_id(row.get("ImageID"))
        label = row.get("LabelName")
        if (
            image_id is None
            or not isinstance(label, str)
            or label not in class_ids
            or str(row.get("IsDepiction", "0")) != "0"
        ):
            continue
        mask, count = states.get(image_id, (0, 0))
        states[image_id] = (mask | (1 << class_ids[label]), count + 1)
    return {
        image_id
        for image_id, (mask, count) in states.items()
        if mask.bit_count() >= min_distinct_classes and count >= min_boxes
    }


def reservoir_select_metadata(
    rows: Iterable[Mapping[str, object]],
    candidate_ids: set[str],
    *,
    limit: int,
    seed: int,
) -> list[SourceRecord]:
    rng = random.Random(seed)
    selected: list[SourceRecord] = []
    eligible_seen = 0
    for row in rows:
        image_id = validated_image_id(row.get("ImageID"))
        license_url = canonical_license(row.get("License"))
        if image_id is None or image_id not in candidate_ids or license_url is None:
            continue
        record = SourceRecord(
            image_id=image_id,
            license_url=license_url,
            landing_url=clean_text(row.get("OriginalLandingURL")),
            author=clean_text(row.get("Author")),
            title=clean_text(row.get("Title")),
        )
        eligible_seen += 1
        if len(selected) < limit:
            selected.append(record)
            continue
        replacement = rng.randrange(eligible_seen)
        if replacement < limit:
            selected[replacement] = record
    return selected


def normalized_jpeg(payload: bytes) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
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
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=90, optimize=True)
                return output.getvalue()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(f"invalid source image: {exc}") from exc


def request_image(record: SourceRecord, timeout: float) -> tuple[SourceRecord, bytes] | None:
    url = safe_source_url(record.image_id)
    for attempt in range(3):
        try:
            with urlopen(safe_request(url), timeout=timeout) as response:  # nosec B310
                final = urlparse(response.geturl())
                if final.scheme != "https" or final.hostname not in ALLOWED_DOWNLOAD_HOSTS:
                    raise ValueError("image download redirected to an untrusted host")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_IMAGE_BYTES:
                    raise ValueError("source image is larger than 20 MiB")
                payload = response.read(MAX_IMAGE_BYTES + 1)
            if len(payload) > MAX_IMAGE_BYTES:
                raise ValueError("source image is larger than 20 MiB")
            return record, normalized_jpeg(payload)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            if attempt == 2:
                print(f"skip image={record.image_id} reason={exc}", file=sys.stderr, flush=True)
                return None
            time.sleep(2**attempt)
    return None


class SiglipStyleLabeler:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.prompts = (*STYLE_PROMPTS, NON_INTERIOR_PROMPT)
        self.processor = AutoProcessor.from_pretrained(
            SIGLIP_MODEL,
            revision=SIGLIP_REVISION,
        )
        self.model = AutoModel.from_pretrained(
            SIGLIP_MODEL,
            revision=SIGLIP_REVISION,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, jpeg_images: Sequence[bytes]) -> list[StylePrediction]:
        torch = self.torch
        images: list[Image.Image] = []
        for payload in jpeg_images:
            with Image.open(io.BytesIO(payload)) as source:
                images.append(source.convert("RGB"))
        inputs = self.processor(
            text=list(self.prompts),
            images=images,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ),
        ):
            logits = self.model(**inputs).logits_per_image
        probabilities = torch.sigmoid(logits.float()).cpu()
        predictions: list[StylePrediction] = []
        for row in probabilities:
            style_probabilities = row[: len(STYLES)]
            values, indices = torch.topk(style_probabilities, k=2)
            predictions.append(
                StylePrediction(
                    style=STYLES[int(indices[0])],
                    confidence=float(values[0]),
                    margin=float(values[0] - values[1]),
                    non_interior_probability=float(row[-1]),
                )
            )
        return predictions


class GCSStore:
    def __init__(self, bucket_name: str, prefix: str) -> None:
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        if not self.bucket.exists():
            raise ValueError(f"Cloud Storage bucket does not exist: gs://{bucket_name}")
        self.prefix = prefix.strip("/")

    def name(self, relative: str) -> str:
        relative = relative.strip("/")
        if not relative or ".." in relative.split("/"):
            raise ValueError("unsafe Cloud Storage object name")
        return f"{self.prefix}/{relative}"

    def upload_bytes(self, relative: str, payload: bytes, content_type: str) -> str:
        blob = self.bucket.blob(self.name(relative))
        blob.upload_from_string(payload, content_type=content_type, timeout=180)
        return f"gs://{self.bucket.name}/{blob.name}"

    def upload_file(self, relative: str, path: Path, content_type: str) -> str:
        blob = self.bucket.blob(self.name(relative))
        blob.upload_from_filename(str(path), content_type=content_type, timeout=600)
        return f"gs://{self.bucket.name}/{blob.name}"

    def exists(self, relative: str) -> bool:
        return self.bucket.blob(self.name(relative)).exists(timeout=60)

    def download_file(self, relative: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        self.bucket.blob(self.name(relative)).download_to_filename(str(temporary), timeout=300)
        os.replace(temporary, path)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON on line {number}") from exc
        image_id = validated_image_id(record.get("image_id")) if isinstance(record, dict) else None
        license_url = canonical_license(record.get("license")) if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or image_id is None
            or image_id in seen_ids
            or record.get("style") not in STYLES
            or license_url != record.get("license")
            or record.get("weak_label_revision") != SIGLIP_REVISION
            or not SHA256_RE.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError(f"invalid manifest record on line {number}")
        seen_ids.add(image_id)
        records.append(record)
    return records


def append_manifest(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def local_image_path(data_root: Path, style: str, image_id: str) -> Path:
    if style not in STYLES or not IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError("unsafe local dataset path")
    return data_root / style / f"{image_id}.jpg"


def rehydrate_existing(
    store: GCSStore,
    records: Sequence[Mapping[str, Any]],
    data_root: Path,
    workers: int,
) -> None:
    def restore(record: Mapping[str, Any]) -> None:
        style = str(record["style"])
        image_id = str(record["image_id"])
        expected_sha256 = str(record["sha256"])
        destination = local_image_path(data_root, style, image_id)
        if destination.exists() and sha256_file(destination) == expected_sha256:
            return
        store.download_file(f"dataset/{style}/{image_id}.jpg", destination)
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"GCS image checksum mismatch: {image_id}")

    if records:
        print(f"rehydrating={len(records)} existing images from GCS", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(restore, records))


def persist_labeled_image(
    store: GCSStore,
    data_root: Path,
    record: SourceRecord,
    prediction: StylePrediction,
    jpeg: bytes,
) -> dict[str, Any]:
    destination = local_image_path(data_root, prediction.style, record.image_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    temporary.write_bytes(jpeg)
    os.replace(temporary, destination)
    digest = hashlib.sha256(jpeg).hexdigest()
    gcs_uri = store.upload_bytes(
        f"dataset/{prediction.style}/{record.image_id}.jpg",
        jpeg,
        "image/jpeg",
    )
    return {
        "source": "open_images",
        "openimages_version": OPENIMAGES_VERSION,
        "image_id": record.image_id,
        "source_url": safe_source_url(record.image_id),
        "original_landing_url": record.landing_url,
        "license": record.license_url,
        "author": record.author,
        "title": record.title,
        "style": prediction.style,
        "weak_label_model": SIGLIP_MODEL,
        "weak_label_revision": SIGLIP_REVISION,
        "weak_label_confidence": round(prediction.confidence, 8),
        "weak_label_margin": round(prediction.margin, 8),
        "non_interior_probability": round(prediction.non_interior_probability, 8),
        "review_required": prediction.confidence < 0.35 or prediction.margin < 0.05,
        "sha256": digest,
        "gcs_uri": gcs_uri,
        "retrieved_at": utc_now(),
    }


def build_dataset(
    args: argparse.Namespace,
    store: GCSStore,
    candidates: list[SourceRecord],
    data_root: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    if store.exists("manifest.jsonl") and not manifest_path.exists():
        store.download_file("manifest.jsonl", manifest_path)
    records = load_manifest(manifest_path)
    existing_ids = {str(record["image_id"]) for record in records}
    rehydrate_existing(store, records, data_root, args.download_workers)
    counts = Counter(str(record["style"]) for record in records)
    if len(records) >= args.max_images:
        return records[: args.max_images]

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = [record for record in candidates if record.image_id not in existing_ids]
    labeler = SiglipStyleLabeler("cuda")
    accepted = len(records)
    attempted = 0
    for offset in range(0, len(candidates), args.label_batch_size):
        if accepted >= args.max_images:
            break
        batch_records = candidates[offset : offset + args.label_batch_size]
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            downloaded = list(
                executor.map(lambda item: request_image(item, args.timeout), batch_records)
            )
        successful = [item for item in downloaded if item is not None]
        attempted += len(batch_records)
        if not successful:
            continue
        predictions = labeler.predict([item[1] for item in successful])
        persist_items: list[tuple[SourceRecord, StylePrediction, bytes]] = []
        for (record, jpeg), prediction in zip(successful, predictions, strict=True):
            if prediction.non_interior_probability >= prediction.confidence:
                continue
            persist_items.append((record, prediction, jpeg))
            if accepted + len(persist_items) >= args.max_images:
                break

        def persist(item: tuple[SourceRecord, StylePrediction, bytes]) -> dict[str, Any]:
            return persist_labeled_image(store, data_root, *item)

        with ThreadPoolExecutor(max_workers=args.upload_workers) as executor:
            new_records = list(executor.map(persist, persist_items))
        for record in new_records:
            append_manifest(manifest_path, record)
            records.append(record)
            counts[str(record["style"])] += 1
            accepted += 1
        if new_records and (
            accepted % args.manifest_checkpoint < len(new_records) or accepted >= args.max_images
        ):
            store.upload_file("manifest.jsonl", manifest_path, "application/x-ndjson")
        if attempted % max(args.label_batch_size * 10, 1) == 0 or accepted >= args.max_images:
            print(
                f"dataset accepted={accepted}/{args.max_images} attempted={attempted} "
                f"classes={dict(sorted(counts.items()))}",
                flush=True,
            )

    if accepted < args.max_images:
        if manifest_path.exists():
            store.upload_file("manifest.jsonl", manifest_path, "application/x-ndjson")
        raise RuntimeError(
            f"Only {accepted} eligible interior images were accepted from the candidate pool; "
            "increase --candidate-multiplier and rerun with the same --run-id"
        )
    missing = [style for style in STYLES if counts[style] < 2]
    if missing:
        raise RuntimeError(
            "Weak labeling produced fewer than two images for: " + ", ".join(missing)
        )
    return records


def train_classifier(args: argparse.Namespace, data_root: Path, output: Path) -> None:
    command = [
        sys.executable,
        "training/train_room_classifier.py",
        str(data_root),
        "--output",
        str(output),
        "--device",
        "cuda",
        "--amp",
        "--pretrained",
        "--class-balance",
        "sampler",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.train_batch_size),
        "--num-workers",
        str(args.train_workers),
        "--validation-fraction",
        "0.1",
        "--early-stopping-patience",
        "5",
    ]
    print("training command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)  # noqa: S603 - fixed executable and arguments.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-images", type=int, default=100_000)
    parser.add_argument("--candidate-multiplier", type=float, default=4.0)
    parser.add_argument("--min-distinct-classes", type=int, default=1)
    parser.add_argument("--min-boxes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--label-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--upload-workers", type=int, default=16)
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--manifest-checkpoint", type=int, default=2_000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/job"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.bucket = normalize_bucket_name(args.bucket)
        args.run_id = validate_run_id(args.run_id)
    except ValueError as exc:
        parser.error(str(exc))
    if not 100 <= args.max_images <= 1_000_000:
        parser.error("--max-images must be between 100 and 1,000,000")
    if not 1.0 <= args.candidate_multiplier <= 10.0:
        parser.error("--candidate-multiplier must be between 1 and 10")
    for name in (
        "min_distinct_classes",
        "min_boxes",
        "epochs",
        "label_batch_size",
        "train_batch_size",
        "download_workers",
        "upload_workers",
        "manifest_checkpoint",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.train_workers < 0:
        parser.error("--train-workers cannot be negative")
    return args


def job_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bucket": f"gs://{args.bucket}",
        "run_id": args.run_id,
        "gcs_prefix": f"runs/{args.run_id}",
        "target_images": args.max_images,
        "candidate_pool": math.ceil(args.max_images * args.candidate_multiplier),
        "source": OPENIMAGES_PAGE,
        "allowed_image_licenses": sorted(set(CANONICAL_LICENSES.values())),
        "weak_label_model": SIGLIP_MODEL,
        "weak_label_revision": SIGLIP_REVISION,
        "weak_label_license": SIGLIP_LICENSE,
        "styles": list(STYLES),
        "human_review_required": True,
    }


def run(args: argparse.Namespace) -> int:
    plan = job_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Vertex job has no NVIDIA GPU visible to PyTorch")
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)

    work_dir = args.work_dir.resolve()
    metadata_dir = work_dir / "metadata"
    data_root = work_dir / "data" / "styles"
    model_path = work_dir / "models" / "style_classifier.pth"
    manifest_path = work_dir / "manifest.jsonl"
    store = GCSStore(args.bucket, f"runs/{args.run_id}")
    status = {**plan, "status": "running", "started_at": utc_now()}
    store.upload_bytes("status.json", json.dumps(status, indent=2).encode(), "application/json")

    try:
        class_path = download_file(
            CLASS_DESCRIPTIONS_URL,
            metadata_dir / "class-descriptions.csv",
            max_bytes=20 * 1024 * 1024,
            timeout=args.timeout,
        )
        boxes_path = download_file(
            BOX_ANNOTATIONS_URL,
            metadata_dir / "train-boxes.csv",
            max_bytes=8 * 1024 * 1024 * 1024,
            timeout=args.timeout,
        )
        metadata_path = download_file(
            TRAIN_METADATA_URL,
            metadata_dir / "train-images.csv",
            max_bytes=3 * 1024 * 1024 * 1024,
            timeout=args.timeout,
        )
        class_ids = load_furniture_class_ids(class_path)
        candidate_ids = candidate_ids_from_rows(
            csv_rows(boxes_path),
            class_ids,
            min_distinct_classes=args.min_distinct_classes,
            min_boxes=args.min_boxes,
        )
        pool_size = math.ceil(args.max_images * args.candidate_multiplier)
        selected = reservoir_select_metadata(
            csv_rows(metadata_path),
            candidate_ids,
            limit=pool_size,
            seed=args.seed,
        )
        print(
            f"candidate_ids={len(candidate_ids)} licensed_candidate_pool={len(selected)}",
            flush=True,
        )
        if len(selected) < args.max_images:
            raise RuntimeError(
                f"Only {len(selected)} licensed candidates are available, below "
                f"--max-images={args.max_images}"
            )
        manifest = build_dataset(args, store, selected, data_root, manifest_path)
        train_classifier(args, data_root, model_path)
        checkpoint_uri = store.upload_file(
            "models/style_classifier.pth", model_path, "application/octet-stream"
        )
        manifest_uri = store.upload_file("manifest.jsonl", manifest_path, "application/x-ndjson")
        status.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "accepted_images": len(manifest),
                "style_counts": dict(
                    sorted(Counter(str(item["style"]) for item in manifest).items())
                ),
                "review_required_images": sum(
                    bool(item.get("review_required")) for item in manifest
                ),
                "checkpoint_uri": checkpoint_uri,
                "manifest_uri": manifest_uri,
                "metadata_sha256": {
                    "class_descriptions": sha256_file(class_path),
                    "box_annotations": sha256_file(boxes_path),
                    "train_metadata": sha256_file(metadata_path),
                },
            }
        )
        store.upload_bytes(
            "status.json", json.dumps(status, indent=2, sort_keys=True).encode(), "application/json"
        )
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": exc.__class__.__name__,
                "error": clean_text(str(exc), 2_000),
            }
        )
        try:
            store.upload_bytes(
                "status.json",
                json.dumps(status, indent=2, sort_keys=True).encode(),
                "application/json",
            )
            if manifest_path.exists():
                store.upload_file("manifest.jsonl", manifest_path, "application/x-ndjson")
        except Exception as status_error:  # pragma: no cover - last-resort reporting only.
            print(f"could not upload failure status: {status_error}", file=sys.stderr)
        raise


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Stopped by operator.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
