"""Decode the supplier catalog committed as base64+gzip into plain artifacts.

The repo keeps ``data/suppliers_master.csv.gz.b64`` (base64 of a gzipped CSV)
to satisfy review/size constraints. For training we need the plain
``suppliers_master.csv.gz`` (GCS layout section 2.2 expects
``datasets/catalog/suppliers_master.csv.gz``); the decoded CSV is kept beside
it for convenience and validation.
"""

from __future__ import annotations

import base64
import binascii
import csv
import gzip
import io
from pathlib import Path

SOURCE_RELATIVE_PATH = Path("data") / "suppliers_master.csv.gz.b64"
GZ_FILENAME = "suppliers_master.csv.gz"
CSV_FILENAME = "suppliers_master.csv"


def prepare_catalog(dest_dir: Path, repo_root: Path) -> Path:
    """Decode ``data/suppliers_master.csv.gz.b64`` into ``dest_dir``.

    Writes both ``suppliers_master.csv.gz`` (returned) and the decompressed
    ``suppliers_master.csv``. Validates that the decoded payload is gzip and
    that the CSV header parses; raises ``ValueError`` otherwise.
    """
    dest_dir = Path(dest_dir)
    repo_root = Path(repo_root)
    source = repo_root / SOURCE_RELATIVE_PATH
    if not source.is_file():
        raise FileNotFoundError(f"supplier catalog source not found: {source}")

    try:
        compressed = base64.b64decode(source.read_bytes(), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{source} is not valid base64: {exc}") from exc
    if not compressed.startswith(b"\x1f\x8b"):
        raise ValueError(f"decoded {source} is not a gzip payload (bad magic bytes)")

    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / GZ_FILENAME
    gz_path.write_bytes(compressed)

    try:
        csv_bytes = gzip.decompress(compressed)
    except OSError as exc:
        raise ValueError(f"could not gunzip decoded catalog: {exc}") from exc
    csv_path = dest_dir / CSV_FILENAME
    csv_path.write_bytes(csv_bytes)

    text = csv_bytes.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or not rows[0] or not any(cell.strip() for cell in rows[0]):
        raise ValueError(f"decoded catalog at {csv_path} has no parseable CSV header")
    header = rows[0]
    if "Supplier Name" not in header[0] or len(header) < 5:
        raise ValueError(f"unexpected catalog header (first cell {header[0]!r})")
    return gz_path
