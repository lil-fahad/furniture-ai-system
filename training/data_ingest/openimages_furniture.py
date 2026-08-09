"""Download a furniture image subset from Open Images (public CSV endpoints).

Only the Python standard library is used (``urllib`` + ``concurrent.futures``),
so this module works in a bare Cloud Shell without extra packages.

Data sources (all under https://storage.googleapis.com/openimages/):
- Class descriptions: ``v6/oidv6-class-descriptions.csv`` (fallback: v5 file).
- Image ids + labels + URLs: the Open Images v6 train file
  ``v6/oidv6-train-images-with-labels-with-rotation.csv`` which carries an
  ``OriginalURL`` column per labelled image (``Thumbnail300KURL`` is kept as a
  documented fallback when ``OriginalURL`` is empty).

Rows without a usable URL are skipped. Image counts are capped per class with
``max_per_class``. Network calls use retries with exponential backoff and
explicit timeouts.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OPENIMAGES_BASE = "https://storage.googleapis.com/openimages"

CLASS_INDEX_URLS = (
    f"{OPENIMAGES_BASE}/v6/oidv6-class-descriptions.csv",
    f"{OPENIMAGES_BASE}/v5/class-descriptions.csv",
)

TRAIN_LABELS_URL = (
    f"{OPENIMAGES_BASE}/v6/oidv6-train-images-with-labels-with-rotation.csv"
)

#: Furniture class display name -> Open Images machine id (MID).
FURNITURE_CLASSES: dict[str, str] = {
    "Chair": "/m/01mzpv",
    "Table": "/m/04bcr3",
    "Sofa": "/m/02crq1",  # Open Images label name is "Couch"
    "Bed": "/m/03ssj5",
    "Cabinetry": "/m/01s105",
    "Desk": "/m/01y9k5",
    "Shelf": "/m/0dt3t",
}

#: Sidecar CSV written by :func:`select_image_ids` and consumed by
#: :func:`download_subset` (columns: class_name, image_id, url).
SELECTION_FILENAME = "selection.csv"

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 2.0
_USER_AGENT = "furniture-ai-data-ingest/1.3 (+https://github.com/lil-fahad/furniture-ai-system)"


def _open_url(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, attempts: int = DEFAULT_ATTEMPTS):
    """Open ``url`` with retries + exponential backoff; returns a response object."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(_BACKOFF_BASE_SECONDS**attempt)
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts: {last_error}")


def _download_to_file(url: str, dest: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    with _open_url(url, timeout=timeout) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(dest)


def fetch_class_index(dest_dir: Path) -> Path:
    """Download the Open Images class-descriptions CSV into ``dest_dir``.

    Returns the path of the saved CSV. Raises ``RuntimeError`` if none of the
    known endpoints work.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "class-descriptions.csv"
    last_error: Exception | None = None
    for url in CLASS_INDEX_URLS:
        try:
            _download_to_file(url, dest)
            break
        except RuntimeError as exc:  # try the next known endpoint
            last_error = exc
    else:
        raise RuntimeError(f"Could not fetch class descriptions: {last_error}")
    # Sanity check: every configured MID should be present in the index.
    content = dest.read_text(encoding="utf-8", errors="replace")
    missing = [name for name, mid in FURNITURE_CLASSES.items() if mid not in content]
    if missing:
        print(f"warning: MIDs not found in class index for: {', '.join(missing)}", file=sys.stderr)
    return dest


def select_image_ids(
    class_names: list[str], max_per_class: int, dest_dir: Path
) -> dict[str, list[str]]:
    """Stream the Open Images v6 train CSV and pick image ids per class.

    The CSV is parsed as a stream (it is far too large to load in memory).
    Rows lacking an ``OriginalURL`` (and a ``Thumbnail300KURL`` fallback) are
    skipped. Selection stops as soon as every requested class reached
    ``max_per_class`` usable rows.

    Writes a sidecar ``dest_dir/selection.csv`` (class_name,image_id,url) that
    :func:`download_subset` uses to resolve URLs without another network pass.

    Returns ``{class_name: [image_id, ...]}``.
    """
    unknown = sorted(set(class_names) - set(FURNITURE_CLASSES))
    if unknown:
        raise ValueError(
            f"Unknown furniture classes: {unknown}; known: {sorted(FURNITURE_CLASSES)}"
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    wanted = {FURNITURE_CLASSES[name]: name for name in class_names}
    selected: dict[str, list[str]] = {name: [] for name in class_names}
    urls: dict[str, str] = {}

    def full() -> bool:
        return all(len(ids) >= max_per_class for ids in selected.values())

    with _open_url(TRAIN_LABELS_URL, timeout=120) as response:
        text_stream = io.TextIOWrapper(response, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text_stream)
        header = [column.strip().lower() for column in next(reader)]
        try:
            id_col = header.index("imageid")
            label_col = header.index("labelname")
        except ValueError as exc:
            raise RuntimeError(f"Unexpected CSV header from {TRAIN_LABELS_URL}: {header}") from exc
        url_col = header.index("originalurl") if "originalurl" in header else None
        thumb_col = header.index("thumbnail300kurl") if "thumbnail300kurl" in header else None
        if url_col is None and thumb_col is None:
            raise RuntimeError(
                f"CSV at {TRAIN_LABELS_URL} has neither OriginalURL nor Thumbnail300KURL"
            )

        for row in reader:
            if full():
                break
            if len(row) <= max(id_col, label_col, url_col or 0, thumb_col or 0):
                continue
            class_name = wanted.get(row[label_col].strip())
            if class_name is None or len(selected[class_name]) >= max_per_class:
                continue
            url = ""
            if url_col is not None:
                url = row[url_col].strip()
            if not url and thumb_col is not None:
                url = row[thumb_col].strip()  # documented fallback thumbnail
            if not url:
                continue  # skip rows without any usable URL
            image_id = row[id_col].strip()
            if not image_id:
                continue
            selected[class_name].append(image_id)
            urls[image_id] = url

    sidecar = dest_dir / SELECTION_FILENAME
    with sidecar.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "image_id", "url"])
        for class_name in class_names:
            for image_id in selected[class_name]:
                writer.writerow([class_name, image_id, urls[image_id]])
    return selected


def _load_selection(dest_dir: Path) -> dict[str, dict[str, str]]:
    """Load the sidecar selection file: image_id -> {class_name, url}."""
    sidecar = Path(dest_dir) / SELECTION_FILENAME
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"{sidecar} not found; run select_image_ids() first so URLs can be resolved"
        )
    table: dict[str, dict[str, str]] = {}
    with sidecar.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["image_id"]] = {"class_name": row["class_name"], "url": row["url"]}
    return table


def _download_one(url: str, dest: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> bool:
    try:
        _download_to_file(url, dest, timeout=timeout)
    except RuntimeError:
        return False
    return True


def download_subset(
    mapping: dict[str, list[str]], dest_dir: Path, workers: int = 8
) -> dict[str, int]:
    """Download images for the selected ids into ``dest_dir/<class>/<id>.jpg``.

    URLs are resolved from the ``selection.csv`` sidecar written by
    :func:`select_image_ids`; ids without a recorded URL are skipped. Returns
    the number of successfully downloaded files per class.
    """
    dest_dir = Path(dest_dir)
    selection = _load_selection(dest_dir)
    counts: dict[str, int] = {class_name: 0 for class_name in mapping}

    jobs: list[tuple[str, str, Path]] = []
    for class_name, image_ids in mapping.items():
        for image_id in image_ids:
            record = selection.get(image_id)
            if record is None or not record["url"]:
                continue  # no URL recorded -> skip
            jobs.append((class_name, record["url"], dest_dir / class_name / f"{image_id}.jpg"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_download_one, url, dest): class_name for class_name, url, dest in jobs
        }
        for future in as_completed(futures):
            class_name = futures[future]
            try:
                if future.result():
                    counts[class_name] += 1
            except Exception as exc:  # network hiccup on one file must not kill the batch
                print(f"warning: download failed for class {class_name}: {exc}", file=sys.stderr)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.data_ingest.openimages_furniture",
        description="Download an Open Images furniture subset (ImageFolder layout).",
    )
    parser.add_argument("--dest", type=Path, required=True, help="Output directory for images")
    parser.add_argument(
        "--max-per-class", type=int, default=300, help="Max images per furniture class"
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args(argv)

    class_names = sorted(FURNITURE_CLASSES)
    print(f"==> fetching Open Images class index into {args.dest}")
    fetch_class_index(args.dest)
    print(f"==> selecting up to {args.max_per_class} image ids per class")
    mapping = select_image_ids(class_names, args.max_per_class, args.dest)
    for class_name in class_names:
        print(f"    {class_name}: {len(mapping[class_name])} ids selected")
    print(f"==> downloading images with {args.workers} workers")
    counts = download_subset(mapping, args.dest, workers=args.workers)
    for class_name in class_names:
        print(f"    {class_name}: {counts[class_name]} images downloaded")
    print(f"==> done: {sum(counts.values())} images under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
