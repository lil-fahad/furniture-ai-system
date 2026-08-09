"""Thin, lazy wrapper over ``google-cloud-storage``.

The heavy dependency is imported *inside* functions only, so this module can be
imported (and unit-tested) in environments where ``google-cloud-storage`` is
not installed, e.g. local dev and CI.
"""

from __future__ import annotations

from pathlib import Path


def gcs_available() -> bool:
    """Return True when ``google-cloud-storage`` can be imported."""
    try:
        import google.cloud.storage  # noqa: F401
    except ImportError:
        return False
    return True


def upload_dir(local_dir: Path, bucket: str, prefix: str) -> int:
    """Upload every file under ``local_dir`` to ``gs://bucket/prefix/...``.

    Returns the number of uploaded files. Raises ``RuntimeError`` when the
    storage client library is unavailable.
    """
    if not gcs_available():
        raise RuntimeError(
            "google-cloud-storage is not installed; install it or run with upload disabled"
        )
    from google.cloud import storage

    local_dir = Path(local_dir)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    clean_prefix = prefix.strip("/")
    count = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        blob = bucket_obj.blob(f"{clean_prefix}/{relative}" if clean_prefix else relative)
        blob.upload_from_filename(str(path))
        count += 1
    return count


def download_dir(bucket: str, prefix: str, local_dir: Path) -> int:
    """Download all blobs under ``gs://bucket/prefix/`` into ``local_dir``.

    Returns the number of downloaded files. Raises ``RuntimeError`` when the
    storage client library is unavailable.
    """
    if not gcs_available():
        raise RuntimeError(
            "google-cloud-storage is not installed; install it or run in local mode"
        )
    from google.cloud import storage

    local_dir = Path(local_dir)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    clean_prefix = prefix.strip("/")
    count = 0
    for blob in client.list_blobs(bucket_obj, prefix=clean_prefix):
        relative = blob.name[len(clean_prefix):].lstrip("/") if clean_prefix else blob.name
        if not relative or blob.name.endswith("/"):
            continue  # skip placeholder "directory" blobs
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        count += 1
    return count
