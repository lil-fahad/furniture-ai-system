from furniture_ai.supplier_catalog import (
    deduplicate_supplier_rows,
    normalize_supplier_row,
    normalize_text,
    normalize_url,
    supplier_row_key,
)


def test_normalize_text_is_deterministic() -> None:
    assert normalize_text("  Sofa\t  Premium  ") == "sofa premium"


def test_normalize_url_removes_fragment_and_trailing_slash() -> None:
    assert normalize_url("HTTPS://Example.COM/catalog/#item") == "https://example.com/catalog/"


def test_supplier_row_key_is_stable_for_formatting_changes() -> None:
    first = {
        "Supplier ID": "SUP-1",
        "Product ID": "SKU-7",
        "Official Website": "https://Example.com/",
        "Supplier Name": " Acme  ",
        "Main Category": "Sofas",
    }
    second = {
        "supplier_id": "sup-1",
        "product_id": "sku-7",
        "source_url": "https://example.com/",
        "supplier_name": "acme",
        "category": "sofas",
    }
    assert supplier_row_key(first) == supplier_row_key(second)


def test_normalize_supplier_row_preserves_identity_and_normalizes_values() -> None:
    row = {"Supplier Name": "  Acme  ", "Official Website": "HTTPS://EXAMPLE.COM/"}
    normalized = normalize_supplier_row(row)
    assert normalized.identity_key == supplier_row_key(row)
    assert normalized.values["Supplier Name"] == "acme"
    assert normalized.values["Official Website"] == "https://example.com/"


def test_deduplication_preserves_first_seen_record_without_merging() -> None:
    rows = [
        {
            "Supplier ID": "S1",
            "Product ID": "P1",
            "Supplier Name": "Acme",
            "Main Category": "Sofa",
        },
        {
            "Supplier ID": "S1",
            "Product ID": "P1",
            "Supplier Name": " acme ",
            "Main Category": "sofa",
        },
        {
            "Supplier ID": "S1",
            "Product ID": "P2",
            "Supplier Name": "Acme",
            "Main Category": "Sofa",
        },
    ]
    result = deduplicate_supplier_rows(rows)
    assert result == [rows[0], rows[2]]


def test_conflicting_product_records_are_not_silently_merged() -> None:
    rows = [
        {
            "Supplier ID": "S1",
            "Product ID": "P1",
            "Supplier Name": "Acme",
            "Main Category": "Sofa",
            "MOQ": "1",
        },
        {
            "Supplier ID": "S1",
            "Product ID": "P1",
            "Supplier Name": "Acme",
            "Main Category": "Chair",
            "MOQ": "10",
        },
    ]
    assert len(deduplicate_supplier_rows(rows)) == 2
