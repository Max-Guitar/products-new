from __future__ import annotations
"""Utilities for retrieving and caching Magento categories."""

from typing import Dict, List, Optional

import requests

from connectors.magento.client import magento_get


def _flatten_magento_categories(node, acc: List[Dict[str, object]]) -> None:
    """Recursively flatten Magento category tree structures."""

    if not node or not isinstance(node, dict):
        return
    cid = node.get("id")
    name = (node.get("name") or "").strip()
    if cid is not None and name:
        try:
            acc.append({"id": int(cid), "name": name})
        except (TypeError, ValueError):
            pass
    children = node.get("children_data") or node.get("children") or []
    for child in children:
        _flatten_magento_categories(child, acc)


def fetch_all_categories(
    session: requests.Session,
    base_url: str,
    store_id: int = 0,
) -> List[Dict[str, object]]:
    """Return a flattened list of all Magento categories for ``store_id``."""

    root = magento_get(session, base_url, f"/categories?storeId={store_id}")
    acc: List[Dict[str, object]] = []
    _flatten_magento_categories(root, acc)
    uniq: Dict[int, str] = {}
    for item in acc:
        cid = item.get("id")
        name = (item.get("name") or "").strip()
        if not isinstance(cid, int) or not name:
            continue
        uniq[cid] = name
    items = [{"id": cid, "name": name} for cid, name in uniq.items()]
    items.sort(key=lambda x: x["name"].casefold())
    return items


def ensure_categories_meta(
    meta_cache,
    session: requests.Session,
    base_url: str,
    store_id: int = 0,
) -> Optional[Dict[str, object]]:
    """Populate ``meta_cache`` with synthetic metadata for categories."""

    cats = fetch_all_categories(session, base_url, store_id=store_id)
    options = [
        {"label": cat["name"], "value": cat["id"]}
        for cat in cats
        if isinstance(cat.get("id"), int) and isinstance(cat.get("name"), str)
    ]
    options_map: Dict[str, int] = {}
    values_to_labels: Dict[str, str] = {}
    for opt in options:
        label = opt.get("label")
        value = opt.get("value")
        if not isinstance(label, str):
            continue
        if not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        options_map[label] = value
        options_map[label.casefold()] = value
        options_map[str(value)] = value
        values_to_labels[str(value)] = label
    meta = {
        "attribute_code": "categories",
        "frontend_input": "multiselect",
        "options": options,
        "options_map": options_map,
        "values_to_labels": values_to_labels,
        "valid_examples": [opt["label"] for opt in options[:10]],
    }
    if hasattr(meta_cache, "set_static"):
        meta_cache.set_static("categories", meta)
    return meta
