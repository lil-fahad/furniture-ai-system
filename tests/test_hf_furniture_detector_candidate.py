from __future__ import annotations

import json
from pathlib import Path

EXPECTED_ARTIFACT_SHA256 = "292c5f45918c4275d9c0dc6777a2f6fc656e50f87684e2aecd045929accc76b1"
EXPECTED_ARTIFACT_SIZE_BYTES = 166497908


def test_candidate_metadata_is_blocked_from_production() -> None:
    path = Path("models/candidates/chosungbeen_furniture_detr.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "candidate"
    assert payload["production_ready"] is False
    assert payload["license"] == "Apache-2.0"
    assert payload["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert payload["artifact_size_bytes"] == EXPECTED_ARTIFACT_SIZE_BYTES
    assert payload["evaluation"]["map"] is None
    assert payload["revision"] is None
    assert payload["admission"]["production_gate"] == "blocked_until_pinned_and_benchmarked"


def test_candidate_source_and_base_model_are_explicit() -> None:
    payload = json.loads(
        Path("models/candidates/chosungbeen_furniture_detr.json").read_text(encoding="utf-8")
    )
    assert payload["source"] == (
        "https://huggingface.co/chosungbeen/furniture_use_data_partial_finetuning"
    )
    assert payload["base_model"] == "facebook/detr-resnet-50"
