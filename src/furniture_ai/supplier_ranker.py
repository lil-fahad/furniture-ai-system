from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_SUPPLIER_DATA = Path("data/suppliers_master.csv.gz.b64")
DEFAULT_SUPPLIER_MODEL = Path("models/supplier_ranker/model.json")


@dataclass(frozen=True)
class SupplierPreference:
    category: str | None = None
    requires_dropshipping: bool = False
    requires_3d_models: bool = False
    requires_direct_fulfillment: bool = False
    max_lead_days: float | None = None
    max_moq: float | None = None
    max_price: float | None = None


def _first_number(value: str | None) -> float:
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(match.group()) if match else 0.0


def _range_numbers(value: str | None) -> tuple[float, float, float]:
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value or "")]
    if not numbers:
        return 0.0, 0.0, 0.0
    if len(numbers) == 1:
        return numbers[0], numbers[0], numbers[0]
    return min(numbers), max(numbers), sum(numbers) / len(numbers)


def _money_value(value: str | None) -> float:
    text = (value or "").lower().replace(",", "")
    match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*([kmb]?)", text)
    if not match:
        return 0.0
    multiplier = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}
    return float(match.group(1)) * multiplier[match.group(2)]


def _is_available(value: str | None) -> float:
    text = (value or "").strip().lower()
    negative_tokens = ("not confirmed", "unavailable", "no", "none")
    if not text or any(token in text for token in negative_tokens):
        return 0.0
    positive_tokens = ("available", "confirmed", "claimed", "excel", "csv", "xml")
    return float(any(token in text for token in positive_tokens))


def _country_city(value: str | None) -> tuple[str, str]:
    text = (value or "").strip()
    country = text.split("(", 1)[0].strip() or "unknown"
    match = re.search(r"\(([^)]+)\)", text)
    city = match.group(1).strip() if match else "unknown"
    return country, city


def structured_features(row: dict[str, str]) -> dict[str, Any]:
    price_min, price_max, price_average = _range_numbers(
        (row.get("Price Range") or "").replace("$", "")
    )
    lead_min, lead_max, lead_average = _range_numbers(row.get("Lead Time"))
    transaction_text = (row.get("Rating & Transactions") or "").split("|")[-1]
    country, city = _country_city(row.get("Country & City"))
    return {
        "years": _first_number(row.get("Years")),
        "rating": _first_number(row.get("Rating & Transactions")),
        "transaction_usd_log": math.log1p(_money_value(transaction_text)),
        "moq": _first_number(row.get("MOQ")),
        "price_min": price_min,
        "price_max": price_max,
        "price_average": price_average,
        "lead_min": lead_min,
        "lead_max": lead_max,
        "lead_average": lead_average,
        "saudi_shipping": _is_available(row.get("Saudi Shipping")),
        "private_label": _is_available(row.get("Private Label")),
        "dropshipping": _is_available(row.get("Dropshipping")),
        "direct_fulfillment": _is_available(row.get("Direct Fulfillment")),
        "catalog_data": _is_available(row.get("Excel/CSV/XML")),
        "api": _is_available(row.get("API")),
        "product_feed": _is_available(row.get("Product Feed")),
        "models_3d": _is_available(row.get("3D Models")),
        "high_res_images": _is_available(row.get("High-Res Images")),
        "image_rights": _is_available(row.get("Image Rights")),
        "category": row.get("Main Category") or "unknown",
        "catalog_format": row.get("Catalog Format") or "unknown",
        "three_d_formats": row.get("3D Formats") or "unknown",
        "customization": row.get("Customization") or "unknown",
        "country": country,
        "city": city,
    }


class StructuredSupplierTransformer(BaseEstimator, TransformerMixin):
    def fit(
        self, rows: Iterable[dict[str, str]], y: object = None
    ) -> StructuredSupplierTransformer:
        return self

    def transform(self, rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
        return [structured_features(row) for row in rows]


def load_supplier_rows(path: Path = DEFAULT_SUPPLIER_DATA) -> list[dict[str, str]]:
    if path.name.endswith(".gz.b64"):
        compressed = base64.b64decode(path.read_text(encoding="ascii"))
        text = gzip.decompress(compressed).decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class LinearSupplierModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, rows: Iterable[dict[str, str]]) -> np.ndarray:
        index = {name: position for position, name in enumerate(self.feature_names)}
        predictions: list[float] = []
        for row in rows:
            values = np.zeros(len(self.feature_names), dtype=float)
            for key, value in structured_features(row).items():
                if isinstance(value, str):
                    position = index.get(f"{key}={value}")
                    if position is not None:
                        values[position] = 1.0
                else:
                    position = index.get(key)
                    if position is not None:
                        values[position] = float(value)
            predictions.append(self.intercept + float(np.dot(values, self.coefficients)))
        return np.asarray(predictions, dtype=float)


def load_supplier_model(path: Path = DEFAULT_SUPPLIER_MODEL) -> LinearSupplierModel:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "linear-supplier-model-v1":
        raise ValueError("Unsupported supplier model format")
    names = tuple(str(item) for item in payload["feature_names"])
    coefficients = tuple(float(item) for item in payload["coefficients"])
    if len(names) != len(coefficients):
        raise ValueError("Supplier model feature and coefficient counts differ")
    return LinearSupplierModel(
        feature_names=names,
        coefficients=coefficients,
        intercept=float(payload["intercept"]),
    )


def _preference_adjustment(row: dict[str, str], preference: SupplierPreference) -> float:
    adjustment = 0.0
    category = (preference.category or "").strip().lower()
    row_category = (row.get("Main Category") or "").lower()
    if category:
        adjustment += 6.0 if category in row_category or row_category in category else -2.0

    requirements = (
        (preference.requires_dropshipping, row.get("Dropshipping")),
        (preference.requires_3d_models, row.get("3D Models")),
        (preference.requires_direct_fulfillment, row.get("Direct Fulfillment")),
    )
    for required, value in requirements:
        if required:
            adjustment += 4.0 if _is_available(value) else -18.0

    _, _, lead_average = _range_numbers(row.get("Lead Time"))
    if preference.max_lead_days is not None and lead_average:
        delta = lead_average - preference.max_lead_days
        adjustment += 3.0 if delta <= 0 else -min(18.0, delta * 0.6)

    moq = _first_number(row.get("MOQ"))
    if preference.max_moq is not None and moq:
        delta = moq - preference.max_moq
        adjustment += 3.0 if delta <= 0 else -min(15.0, delta * 0.5)

    _, _, price_average = _range_numbers((row.get("Price Range") or "").replace("$", ""))
    if preference.max_price is not None and price_average:
        delta = price_average - preference.max_price
        adjustment += 3.0 if delta <= 0 else -min(20.0, delta / 100.0)

    return adjustment


def rank_suppliers(
    rows: list[dict[str, str]],
    model: Any,
    preference: SupplierPreference | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    preference = preference or SupplierPreference()
    model_scores = np.asarray(model.predict(rows), dtype=float)
    ranked: list[dict[str, Any]] = []
    for row, model_score in zip(rows, model_scores, strict=True):
        adjustment = _preference_adjustment(row, preference)
        ranked.append(
            {
                "supplier_name": row.get("Supplier Name", ""),
                "category": row.get("Main Category", ""),
                "country_city": row.get("Country & City", ""),
                "official_website": row.get("Official Website", ""),
                "profile_link": row.get("Profile Link", ""),
                "saudi_shipping": row.get("Saudi Shipping", ""),
                "dropshipping": row.get("Dropshipping", ""),
                "models_3d": row.get("3D Models", ""),
                "lead_time": row.get("Lead Time", ""),
                "moq": row.get("MOQ", ""),
                "price_range": row.get("Price Range", ""),
                "model_score": round(float(model_score), 2),
                "preference_adjustment": round(adjustment, 2),
                "final_score": round(float(model_score) + adjustment, 2),
                "reference_score": _first_number(row.get("Suitability Score")),
                "key_strengths": row.get("Key Strengths", ""),
                "main_risks": row.get("Main Risks", ""),
            }
        )
    ranked.sort(key=lambda item: (item["final_score"], item["model_score"]), reverse=True)
    return ranked[: max(1, min(top_k, len(ranked)))]
