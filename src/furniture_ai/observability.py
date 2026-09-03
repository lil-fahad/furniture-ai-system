from __future__ import annotations

import re
import time
import uuid

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def normalize_request_id(value: str | None) -> str:
    if value and _REQUEST_ID_RE.fullmatch(value):
        return value
    return uuid.uuid4().hex


def elapsed_ms(started: float) -> float:
    return max((time.perf_counter() - started) * 1000.0, 0.0)
