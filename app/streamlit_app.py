from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from services.attributes import build_attributes_table_for_sku
from services.inventory import (
    get_attr_set_id,
    get_backorders_parallel,
    get_source_items,
    iter_products_by_attr_set,
)
from utils.http import get_session


_DEF_ATTR_SET_NAME = "Default"
_ALLOWED_TYPES = {"simple", "configurable"}


def _get_custom_attr_value(item: dict, code: str, default=None):
    for attr in item.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == code:
            return attr.get("value", default)
    return default


def load_items(session, base_url):
    attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)

    rows = []
    prog = st.progress(0.0, text="Loading products…")
    total_hint = 0
    for product, total in iter_products_by_attr_set(session, base_url, attr_set_id):
        total_hint = total or total_hint
        status = int(
            _get_custom_attr_value(product, "status", product.get("status", 1)) or 1
        )
        visibility = int(
            _get_custom_attr_value(product, "visibility", product.get("visibility", 4))
            or 4
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

        if total_hint:
            done = len(rows) / max(total_hint, 1)
            done = min(done, 1.0)
            prog.progress(done, text=f"Loading products… {int(done * 100)}%")

    prog.progress(1.0, text="Products loaded")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    prog2 = st.progress(0.0, text="Fetching MSI (default)…")
    source_items = get_source_items(session, base_url, source_code="default")
    prog2.progress(1.0, text=f"MSI fetched: {len(source_items)} rows")

    qty_map = {item.get("sku"): float(item.get("quantity", 0)) for item in source_items}
    df["qty"] = df["sku"].map(qty_map).fillna(0.0)

    zero_qty_skus = df.loc[df["qty"] <= 0, "sku"].tolist()
    total_backorder_tasks = len(zero_qty_skus)
    prog3 = st.progress(0.0, text=f"Checking backorders… 0/{total_backorder_tasks}")

    if total_backorder_tasks:
        def _progress_cb(completed: int):
            if completed % 50 == 0 or completed == total_backorder_tasks:
                ratio = completed / max(total_backorder_tasks, 1)
                prog3.progress(
                    ratio,
                    text=f"Checking backorders… {completed}/{total_backorder_tasks}",
                )

        backorders_map = get_backorders_parallel(
            session,
            base_url,
            zero_qty_skus,
            progress_cb=_progress_cb,
        )
    else:
        backorders_map = {}

    prog3.progress(1.0, text="Backorders complete")

    df["backorders"] = df["sku"].map(backorders_map).fillna(0).astype(int)

    df = df[(df["qty"] > 0) | (df["backorders"] == 2)].copy()
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    df["attribute set"] = _DEF_ATTR_SET_NAME
    df["date created"] = df["created_at"]
    return df[["sku", "name", "attribute set", "date created"]]


st.set_page_config(page_title="Default Set In-Stock Browser", layout="wide")

base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
auth_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
session = get_session(auth_token)

st.title("Default Set — In-Stock & Backorder Browser")
st.caption(
    "Filters: attribute set = Default; qty > 0 OR backorders = 2 (Allow Qty Below 0 & Notify Customer)."
)

if st.button("Load items", type="primary"):
    try:
        df_items = load_items(session, base_url)
        if df_items.empty:
            st.warning("No items match the Default set filter criteria.")
        else:
            df_ui = df_items.copy()
            df_ui["date created"] = pd.to_datetime(
                df_ui["date created"], errors="coerce"
            )
            df_ui = df_ui.sort_values("date created", ascending=False).reset_index(drop=True)
            st.success(
                f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)"
            )
            st.dataframe(df_ui, use_container_width=True)
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"Error: {exc}")
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")

st.divider()
st.header("Attributes lookup", anchor="attributes")
sku = st.text_input("SKU", placeholder="Enter product SKU")
include_empty = st.checkbox("Include empty attributes", value=False)

if st.button("Show attributes"):
    if not sku:
        st.warning("Please enter a SKU value first.")
    else:
        try:
            df_attrs = build_attributes_table_for_sku(
                session, base_url, sku.strip(), include_empty=include_empty
            )
            if df_attrs.empty:
                st.info("No attributes to display for the provided SKU.")
            else:
                st.dataframe(df_attrs, use_container_width=True)
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Error: {exc}")
