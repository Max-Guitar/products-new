from __future__ import annotations

from typing import Any, Dict, Tuple
from urllib.parse import unquote

import pytest

pytest.importorskip("pandas")

from services.inventory import get_backorders_parallel, load_default_items


class DummySession:
    def get(self, *args, **kwargs):  # pragma: no cover - should not be used directly
        raise AssertionError("Network access should be mocked in tests")


@pytest.fixture
def mock_magento_get(monkeypatch):
    responses: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, Any]] = {}

    def handler(session, base_url, path, params=None, timeout=60):
        key = (path, tuple(sorted((params or {}).items())))
        if path.startswith("/stockItems/"):
            sku = unquote(path.split("/stockItems/")[1])
            return {"backorders": responses[("/stockItems/*", (("sku", sku),))]["backorders"]}
        if key not in responses:
            raise AssertionError(f"Unexpected request: {key}")
        return responses[key]

    def register(path: str, params: Dict[str, Any], data: Dict[str, Any]):
        responses[(path, tuple(sorted((params or {}).items())))] = data

    def register_backorder(sku: str, value: int):
        responses[("/stockItems/*", (("sku", sku),))] = {"backorders": value}

    monkeypatch.setattr("services.inventory.magento_get", handler)

    return register, register_backorder


def test_load_default_items_filters_and_maps(monkeypatch, mock_magento_get):
    register, register_backorder = mock_magento_get
    session = DummySession()
    base_url = "https://example.com"

    register(
        "/products/attribute-sets/sets/list",
        {
            "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_name",
            "searchCriteria[filter_groups][0][filters][0][value]": "Default",
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
        },
        {"items": [{"attribute_set_id": 4}]},
    )

    products_page = {
        "items": [
            {
                "sku": "SKU1",
                "name": "Item 1",
                "created_at": "2024-01-02 10:00:00",
                "type_id": "simple",
                "custom_attributes": [
                    {"attribute_code": "status", "value": 1},
                    {"attribute_code": "visibility", "value": 4},
                ],
            },
            {
                "sku": "SKU2",
                "name": "Item 2",
                "created_at": "2024-01-03 10:00:00",
                "type_id": "simple",
                "custom_attributes": [
                    {"attribute_code": "status", "value": 1},
                    {"attribute_code": "visibility", "value": 4},
                ],
            },
            {
                "sku": "SKU3",
                "name": "Item 3",
                "created_at": "2024-01-01 10:00:00",
                "type_id": "simple",
                "custom_attributes": [
                    {"attribute_code": "status", "value": 1},
                    {"attribute_code": "visibility", "value": 4},
                ],
            },
        ],
        "total_count": 3,
    }

    register(
        "/products",
        {
            "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_id",
            "searchCriteria[filter_groups][0][filters][0][value]": 4,
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": 200,
            "searchCriteria[currentPage]": 1,
        },
        products_page,
    )

    source_items_page = {
        "items": [
            {"sku": "SKU1", "quantity": 5},
            {"sku": "SKU2", "quantity": 0},
            {"sku": "SKU3", "quantity": 0},
        ],
        "total_count": 3,
    }

    register(
        "/inventory/source-items",
        {
            "searchCriteria[filter_groups][0][filters][0][field]": "source_code",
            "searchCriteria[filter_groups][0][filters][0][value]": "default",
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": 500,
            "searchCriteria[currentPage]": 1,
        },
        source_items_page,
    )

    register_backorder("SKU2", 2)
    register_backorder("SKU3", 0)

    df = load_default_items(session, base_url)
    assert list(df.columns) == ["sku", "name", "attribute set", "date created"]
    assert df.shape[0] == 2
    assert set(df["sku"]) == {"SKU1", "SKU2"}

    sku2_row = df[df["sku"] == "SKU2"].iloc[0]
    assert sku2_row["attribute set"] == "Default"
    assert sku2_row["date created"] == "2024-01-03 10:00:00"


def test_get_backorders_parallel(monkeypatch):
    session = DummySession()
    base_url = "https://example.com"

    def fake_get(session, base_url, path, params=None, timeout=60):
        sku = unquote(path.split("/stockItems/")[1])
        return {"backorders": {"SKU0": 0, "SKU1": 1, "SKU2": 2}.get(sku, 0)}

    monkeypatch.setattr("services.inventory.magento_get", fake_get)

    result = get_backorders_parallel(session, base_url, ["SKU0", "SKU1", "SKU2"], max_workers=2)
    assert result == {"SKU0": 0, "SKU1": 1, "SKU2": 2}
