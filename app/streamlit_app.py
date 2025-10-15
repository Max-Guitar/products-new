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
    detect_default_source_code,
    get_source_items,
    iter_products_by_attr_set,
    list_attribute_sets,
    update_product_attribute_set,
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
        if visibility == 1 or type_id not in _ALLOWED_TYPES:
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

    st.info(
        f"STEP A — products in 'Default' after status/visibility/type: {len(df)}"
    )

    source_code = detect_default_source_code(session, base_url)
    prog2 = st.progress(0.0, text=f"Fetching MSI ({source_code})…")
    src_items = get_source_items(session, base_url, source_code=source_code)
    prog2.progress(1.0, text=f"MSI fetched ({source_code}): {len(src_items)} rows")
    st.info(f"STEP B — MSI source_items rows: {len(src_items)}")
    pos_qty = sum(1 for s in src_items if float(s.get("quantity", 0)) > 0)
    st.info(f"STEP B+ — MSI items with qty>0: {pos_qty}")

    qty_map = {item.get("sku"): float(item.get("quantity", 0)) for item in src_items}
    df["qty"] = df["sku"].map(qty_map).fillna(0.0)
    have_qty = int((df["qty"] > 0).sum())
    st.info(f"STEP C — products in set having qty>0: {have_qty}")

    df_pos = df[df["qty"] > 0].copy()
    df_pos["backorders"] = 0

    zero_qty_skus = df.loc[df["qty"] <= 0, "sku"].tolist()
    st.info(f"STEP D — zero-qty skus to check backorders: {len(zero_qty_skus)}")
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

    df_zero = df[df["qty"] <= 0].copy()
    df_zero["backorders"] = df_zero["sku"].map(backorders_map).fillna(0).astype(int)
    df_bo2 = df_zero[df_zero["backorders"] == 2]
    st.info(f"STEP E — backorders==2 count: {len(df_bo2)}")

    out = pd.concat([df_pos, df_bo2], ignore_index=True)
    st.success(f"STEP F — final rows (qty>0 OR bo=2): {len(out)}")

    if out.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "date created"])

    out["attribute set"] = _DEF_ATTR_SET_NAME
    out["date created"] = out["created_at"]
    return out[["sku", "name", "attribute set", "date created"]]


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
            st.session_state.pop("df_original", None)
            st.session_state.pop("df_edited", None)
            st.session_state.pop("attribute_sets", None)
        else:
            df_ui = df_items.copy()
            df_ui["date created"] = pd.to_datetime(
                df_ui["date created"], errors="coerce"
            )
            df_ui = df_ui.sort_values("date created", ascending=False).reset_index(drop=True)
            st.success(
                f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)"
            )
            st.session_state["df_original"] = df_ui.copy()
            st.session_state["df_edited"] = df_ui.copy()
            try:
                attribute_sets = list_attribute_sets(session, base_url)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Failed to fetch attribute sets: {exc}")
                attribute_sets = {}
            st.session_state["attribute_sets"] = attribute_sets
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"Error: {exc}")

if "df_original" in st.session_state:
    df_ui = st.session_state["df_original"]
    attribute_sets = st.session_state.get("attribute_sets", {})

    if df_ui.empty:
        st.warning("No items match the Default set filter criteria.")
    elif not attribute_sets:
        st.warning("Unable to load attribute sets for editing.")
        st.dataframe(df_ui, use_container_width=True)
    else:
        if "df_edited" not in st.session_state:
            st.session_state["df_edited"] = df_ui.copy()
        df_current = st.session_state["df_edited"]

        if "selected" not in df_current.columns:
            df_current = df_current.copy()
            df_current.insert(0, "selected", False)
            st.session_state["df_edited"] = df_current
        elif df_current.columns[0] != "selected":
            df_current = df_current.copy()
            selected_col = df_current.pop("selected")
            df_current.insert(0, "selected", selected_col)
            st.session_state["df_edited"] = df_current

        df_for_editor = st.session_state["df_edited"]
        options = list(attribute_sets.keys())
        options.extend(
            name
            for name in st.session_state["df_edited"]["attribute set"].dropna().unique()
            if name not in attribute_sets
        )
        # Preserve insertion order while removing duplicates.
        options = list(dict.fromkeys(options))

        edited_df = st.data_editor(
            df_for_editor,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "✓",
                    default=False,
                    help="Mark items for bulk actions",
                ),
                "attribute set": st.column_config.SelectboxColumn(
                    "Attribute Set",
                    help="Change attribute set",
                    options=options,
                    required=True,
                )
            },
            num_rows="fixed",
            use_container_width=True,
            key="editable_attribute_sets",
        )
        if not edited_df.equals(st.session_state["df_edited"]):
            st.session_state["df_edited"] = edited_df.copy()

        if st.button("Apply changes"):
            mask = edited_df["attribute set"] != df_ui["attribute set"]
            changed_rows = edited_df.loc[mask]
            if changed_rows.empty:
                st.info("No attribute set changes detected.")
            else:
                for _, row in changed_rows.iterrows():
                    sku = row.get("sku", "")
                    new_attr_set_name = str(row.get("attribute set") or "")
                    if not sku:
                        continue
                    if not new_attr_set_name:
                        st.error("Attribute set name is empty.")
                        continue
                    attr_set_id = attribute_sets.get(new_attr_set_name)
                    if attr_set_id is None:
                        st.error(
                            f"Attribute set ID not found for {new_attr_set_name!r}."
                        )
                        continue
                    try:
                        update_product_attribute_set(
                            session, base_url, sku, attr_set_id
                        )
                    except Exception as exc:  # pragma: no cover - UI error handling
                        st.error(
                            f"Failed to update {sku} to set {new_attr_set_name}: {exc}"
                        )
                    else:
                        st.success(
                            f"Updated {sku} to set {new_attr_set_name}"
                        )
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
