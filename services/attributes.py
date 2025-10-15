"""Attribute-related helpers for Magento products."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import requests

from connectors.magento import magento_get


def _get_custom_attr_map(product: Dict[str, Any]) -> Dict[str, Any]:
    attrs = {}
    for item in product.get("custom_attributes", []) or []:
        code = item.get("attribute_code")
        if code:
            attrs[code] = item.get("value")
    return attrs


def get_product_by_sku(session: requests.Session, base_url: str, sku: str) -> Dict[str, Any]:
    return magento_get(session, base_url, f"/products/{sku}")


def get_attrset_attributes(
    session: requests.Session, base_url: str, attribute_set_id: int
) -> List[Dict[str, Any]]:
    data = magento_get(
        session,
        base_url,
        f"/products/attribute-sets/{attribute_set_id}/attributes",
    )
    if isinstance(data, dict):
        return data.get("items", [])
    return data


def get_attribute_meta(session: requests.Session, base_url: str, code: str) -> Dict[str, Any]:
    return magento_get(session, base_url, f"/products/attributes/{code}")


def build_attributes_table_for_sku(
    session: requests.Session,
    base_url: str,
    sku: str,
    include_empty: bool = False,
) -> pd.DataFrame:
    product = get_product_by_sku(session, base_url, sku)
    attr_map = _get_custom_attr_map(product)
    attribute_set_id = int(product.get("attribute_set_id", 0))

    attrs = get_attrset_attributes(session, base_url, attribute_set_id)
    rows = []
    for attr in attrs:
        code = attr.get("attribute_code")
        if not code:
            continue
        label = attr.get("frontend_label") or code
        value = attr_map.get(code, product.get(code))
        if isinstance(value, (list, dict)):
            value_display = json.dumps(value)
        else:
            value_display = value
        if not include_empty and (value_display is None or value_display == ""):
            continue
        meta = get_attribute_meta(session, base_url, code)
        rows.append(
            {
                "attribute_code": code,
                "label": label,
                "value": value_display,
                "frontend_input": meta.get("frontend_input"),
                "is_required": bool(meta.get("is_required")),
            }
        )
    return pd.DataFrame(rows)
