"""Inventory-related services for Magento products."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd
import requests

from connectors.magento import magento_get


_DEF_ATTR_SET_NAME = "Default"
_ALLOWED_TYPES = {"simple", "configurable"}


def _get_custom_attr(item: dict, code: str, default=None):
    for attr in item.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == code:
            return attr.get("value", default)
    return default


def get_attr_set_id(
    session: requests.Session, base_url: str, name: str = _DEF_ATTR_SET_NAME
) -> int:
    params = {
        "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_name",
        "searchCriteria[filter_groups][0][filters][0][value]": name,
        "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
    }
    data = magento_get(session, base_url, "/products/attribute-sets/sets/list", params=params)
    items = data.get("items", [])
    if not items:
        raise ValueError(f'Attribute set "{name}" not found')
    return int(items[0]["attribute_set_id"])


def iter_products_by_attr_set(
    session: requests.Session,
    base_url: str,
    attr_id: int,
    page_size: int = 200,
) -> Iterator[Tuple[dict, int]]:
    page = 1
    total_count: Optional[int] = None
    while True:
        params = {
            "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_id",
            "searchCriteria[filter_groups][0][filters][0][value]": attr_id,
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": page_size,
            "searchCriteria[currentPage]": page,
        }
        data = magento_get(session, base_url, "/products", params=params)
        if total_count is None:
            total_count = int(data.get("total_count", 0) or 0)
        items = data.get("items", []) or []
        if not items:
            break
        for item in items:
            yield item, total_count or 0
        if len(items) < page_size:
            break
        page += 1


def get_source_items(
    session: requests.Session,
    base_url: str,
    source_code: str = "default",
    page_size: int = 500,
) -> List[dict]:
    page = 1
    collected: List[dict] = []
    while True:
        params = {
            "searchCriteria[filter_groups][0][filters][0][field]": "source_code",
            "searchCriteria[filter_groups][0][filters][0][value]": source_code,
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": page_size,
            "searchCriteria[currentPage]": page,
        }
        data = magento_get(session, base_url, "/inventory/source-items", params=params)
        items = data.get("items", []) or []
        if not items:
            break
        collected.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return collected


def _fetch_backorder(session: requests.Session, base_url: str, sku: str) -> int:
    encoded = quote(sku, safe="")
    try:
        data = magento_get(session, base_url, f"/stockItems/{encoded}")
    except requests.HTTPError:
        return 0
    return int(data.get("backorders", 0))


def get_backorders_parallel(
    session: requests.Session,
    base_url: str,
    skus: Sequence[str],
    max_workers: int = 16,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, int]:
    if not skus:
        return {}
    max_workers = max(1, min(max_workers, len(skus)))
    results: Dict[str, int] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_fetch_backorder, session, base_url, sku): sku for sku in skus}
        for future in as_completed(future_map):
            sku = future_map[future]
            try:
                results[sku] = int(future.result())
            except Exception:
                results[sku] = 0
            completed += 1
            if progress_cb is not None:
                try:
                    progress_cb(completed)
                except Exception:
                    pass
    return results


def load_default_items(session: requests.Session, base_url: str) -> pd.DataFrame:
    attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)
    products = [item for item, _ in iter_products_by_attr_set(session, base_url, attr_set_id)]

    rows = []
    for product in products:
        status = int(_get_custom_attr(product, "status", product.get("status", 1)) or 1)
        visibility = int(
            _get_custom_attr(product, "visibility", product.get("visibility", 4)) or 4
        )
        type_id = product.get("type_id", "simple")
        if status != 1 or visibility == 1 or type_id not in _ALLOWED_TYPES:
            continue
        rows.append(
            {
                "sku": product.get("sku", ""),
                "name": product.get("name", ""),
                "created_at": product.get("created_at", ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    source_items = get_source_items(session, base_url, source_code="default")
    qty_map = {item.get("sku"): float(item.get("quantity", 0)) for item in source_items}
    df["qty"] = df["sku"].map(qty_map).fillna(0.0)

    zero_qty_skus = df.loc[df["qty"] <= 0, "sku"].tolist()
    backorders_map = get_backorders_parallel(session, base_url, zero_qty_skus)
    df["backorders"] = df["sku"].map(backorders_map).fillna(0).astype(int)

    df = df[(df["qty"] > 0) | (df["backorders"] == 2)].copy()
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    df["attribute set"] = _DEF_ATTR_SET_NAME
    df["date created"] = df["created_at"]
    return df[["sku", "name", "attribute set", "date created"]]
