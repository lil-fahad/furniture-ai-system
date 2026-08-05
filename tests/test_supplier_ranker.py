from __future__ import annotations

from furniture_ai.supplier_ranker import (
    SupplierPreference,
    load_supplier_model,
    load_supplier_rows,
    rank_suppliers,
    structured_features,
)


def test_structured_features_parse_supplier_fields() -> None:
    row = {
        "Years": "5",
        "Rating & Transactions": "4.9/5 (29) | $140k+",
        "MOQ": "1-2 pcs",
        "Price Range": "$300-$1200",
        "Lead Time": "20-35 days",
        "Saudi Shipping": "Confirmed",
        "Private Label": "Available",
        "Dropshipping": "Not confirmed",
        "Direct Fulfillment": "Available",
        "Excel/CSV/XML": "Excel/CSV",
        "API": "Not confirmed",
        "Product Feed": "Not confirmed",
        "3D Models": "Available",
        "High-Res Images": "Available",
        "Image Rights": "Claimed",
        "Main Category": "Living/Bedroom",
        "Catalog Format": "Excel/PDF",
        "3D Formats": "3D Rendering",
        "Customization": "High",
        "Country & City": "China (Foshan)",
    }
    features = structured_features(row)
    assert features["years"] == 5
    assert features["price_average"] == 750
    assert features["lead_average"] == 27.5
    assert features["dropshipping"] == 0
    assert features["models_3d"] == 1
    assert features["city"] == "Foshan"


def test_rank_suppliers_uses_preferences() -> None:
    rows = load_supplier_rows()
    model = load_supplier_model()
    results = rank_suppliers(
        rows,
        model,
        SupplierPreference(requires_dropshipping=True, requires_3d_models=True),
        top_k=5,
    )
    assert len(results) == 5
    assert all(item["models_3d"] == "Available" for item in results)
    assert results[0]["final_score"] >= results[-1]["final_score"]
