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


def get_config_children(
    session: requests.Session, base_url: str, parent_sku: str
) -> List[str]:
    data = magento_get(
        session,
        base_url,
        f"/configurable-products/{quote(parent_sku, safe='')}/children",
    )
    return [child.get("sku", "") for child in data or []]


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
        for item in items:
            yield item, total_count or 0
        if not items or len(items) < page_size:
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
        collected.extend(items)
        if not items or len(items) < page_size:
            break
        page += 1
    return collected


def detect_default_source_code(session: requests.Session, base_url: str) -> str:
    try:
        data = magento_get(
            session,
            base_url,
            "/inventory/sources",
            {
                "searchCriteria[pageSize]": 500,
                "searchCriteria[currentPage]": 1,
            },
        )
        for source in data.get("items", []) or []:
            code = (source.get("source_code") or "").strip()
            name = (source.get("name") or "").strip().lower()
            if code.lower() == "default" or name in {"default", "default source"}:
                return code or "default"
    except Exception:
        pass
    return "default"


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
        rows.append(
            {
                "sku": product.get("sku", ""),
                "name": product.get("name", ""),
                "created_at": product.get("created_at", ""),
                "status": status,
                "visibility": visibility,
                "type_id": type_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    df = df[(df["status"] == 1) & (df["type_id"].isin(_ALLOWED_TYPES))].copy()
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    simple_skus = set(df.loc[df["type_id"] == "simple", "sku"])
    config_skus = df.loc[df["type_id"] == "configurable", "sku"].tolist()

    child_skus: set[str] = set()
    for parent_sku in config_skus:
        child_skus.update(get_config_children(session, base_url, parent_sku))

    def _norm(s: str) -> str:
        return str(s).strip().lower()

    source_code = detect_default_source_code(session, base_url)
    src_items = get_source_items(session, base_url, source_code=source_code)
    qty_map = {_norm(item.get("sku")): float(item.get("quantity", 0)) for item in src_items}

    df["qty"] = df["sku"].apply(_norm).map(qty_map).fillna(0.0)

    if child_skus:
        df_children = pd.DataFrame({"sku": list(child_skus)})
        df_children["qty"] = df_children["sku"].apply(_norm).map(qty_map).fillna(0.0)
    else:
        df_children = pd.DataFrame(columns=["sku", "qty"])

    matched = int((df["qty"] > 0).sum())
    matched_children = int((df_children["qty"] > 0).sum()) if not df_children.empty else 0
    print(
        f"[DEBUG] MSI qty>0 matches: simple/config rows={matched}, children rows={matched_children}, MSI total={len(src_items)}"
    )

    zero_main = df.loc[df["qty"] <= 0, "sku"].tolist()
    zero_child = df_children.loc[df_children["qty"] <= 0, "sku"].tolist()
    zero_skus = list(set(zero_main + zero_child))
    back_map = get_backorders_parallel(session, base_url, zero_skus)

    df_main_pos = df[df["qty"] > 0].copy()
    df_child_pos = df_children[df_children["qty"] > 0].copy()

    df_main_zero = df[df["qty"] <= 0].copy()
    df_main_zero["backorders"] = df_main_zero["sku"].map(back_map).fillna(0).astype(int)
    df_child_zero = df_children[df_children["qty"] <= 0].copy()
    df_child_zero["backorders"] = (
        df_child_zero["sku"].map(back_map).fillna(0).astype(int)
    )

    df_bo2 = pd.concat(
        [
            df_main_zero[df_main_zero["backorders"] == 2][["sku"]],
            df_child_zero[df_child_zero["backorders"] == 2][["sku"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    result_skus = (
        set(df_main_pos["sku"])
        | set(df_child_pos["sku"])
        | set(df_bo2["sku"])
    )

    if not result_skus:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    out = df[df["sku"].isin(result_skus)].copy()
    missing_children = (result_skus - set(out["sku"])) & child_skus
    if missing_children:
        missing_list = list(missing_children)
        child_rows = pd.DataFrame(
            {
                "sku": missing_list,
                "name": ["" for _ in missing_list],
                "created_at": ["" for _ in missing_list],
            }
        )
        out = pd.concat([out, child_rows], ignore_index=True, sort=False)

    out = out.drop_duplicates(subset=["sku"])
    out["attribute set"] = _DEF_ATTR_SET_NAME
    out["date created"] = out.get("created_at", "")
    return out[["sku", "name", "attribute set", "date created"]]
