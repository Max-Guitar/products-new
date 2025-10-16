from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    build_attributes_display_table,
    collect_attributes_table,
    compute_allowed_attrs,
    get_attribute_sets_map,
    get_product_by_sku,
    probe_api_base,
)
from services.attributes import build_attributes_table_for_sku
from services.inventory import (
    get_attr_set_id,
    get_backorders_parallel,
    detect_default_source_code,
    get_source_items,
    iter_products_by_attr_set,
    list_attribute_sets,
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
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

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
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

    out["attribute set"] = _DEF_ATTR_SET_NAME
    return out[["sku", "name", "attribute set", "created_at"]]


st.set_page_config(page_title="Default Set In-Stock Browser", layout="wide")

base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
auth_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
session = get_session(auth_token)

st.markdown("### 🤖 AI Content Manager")
st.info("Let’s find items that need attribute set assignment.")

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
            df_ui["created_at"] = pd.to_datetime(
                df_ui["created_at"], errors="coerce"
            )
            df_ui = df_ui.sort_values("created_at", ascending=False).reset_index(drop=True)
            st.success(
                f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)"
            )
            st.session_state["df_original"] = df_ui.copy()
            try:
                attribute_sets = list_attribute_sets(session, base_url)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Failed to fetch attribute sets: {exc}")
                attribute_sets = {}
            st.session_state["attribute_sets"] = attribute_sets
            df_for_edit = df_ui.copy()
            if "hint" not in df_for_edit.columns:
                df_for_edit["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            df_for_edit = df_for_edit[[col for col in cols_order if col in df_for_edit.columns]]
            st.session_state["df_edited"] = df_for_edit.copy()
            st.session_state["show_attributes_trigger"] = False
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
            df_init = df_ui.copy()
            if "hint" not in df_init.columns:
                df_init["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            df_init = df_init[[col for col in cols_order if col in df_init.columns]]
            st.session_state["df_edited"] = df_init.copy()

        df_base = st.session_state["df_edited"].copy()

        if "hint" not in df_base.columns:
            df_base["hint"] = ""

        cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
        df_base = df_base[[col for col in cols_order if col in df_base.columns]]
        options = list(attribute_sets.keys())
        options.extend(
            name
            for name in df_base["attribute set"].dropna().unique()
            if name not in attribute_sets
        )
        options = list(dict.fromkeys(options))

        st.markdown("**Step 1. Assign the attribute sets**")

        edited_df = st.data_editor(
            df_base,
            column_config={
                "sku": st.column_config.TextColumn("SKU", disabled=True),
                "name": st.column_config.TextColumn("Name", disabled=True),
                "attribute set": st.column_config.SelectboxColumn(
                    label="🎯 Attribute Set",
                    help="Change attribute set",
                    options=options,
                    required=True,
                ),
                "hint": st.column_config.TextColumn("Hint"),
                "created_at": st.column_config.DatetimeColumn("Created At", disabled=True),
            },
            column_order=["sku", "name", "attribute set", "hint", "created_at"],
            use_container_width=True,
            num_rows="fixed",
            key="editor_key_main",
        )

        st.markdown("### Step 2. Items with updated attribute sets")

        show_attributes_clicked = False
        if isinstance(edited_df, pd.DataFrame) and st.button("➡️ Show Attributes"):
            st.session_state["df_edited"] = edited_df.copy()
            st.session_state["show_attributes_trigger"] = True
            show_attributes_clicked = True

        trigger = st.session_state.get("show_attributes_trigger", False)
        if trigger or show_attributes_clicked:
            if "df_edited" not in st.session_state:
                st.info("Нет изменённых товаров.")
            elif "df_original" not in st.session_state:
                st.info("Нет исходных данных для сравнения.")
            else:
                df_new = st.session_state["df_edited"].copy()
                df_old = st.session_state["df_original"].copy()

                required_cols = {"sku", "attribute set"}
                if not (
                    required_cols.issubset(df_new.columns)
                    and required_cols.issubset(df_old.columns)
                ):
                    st.info("Нет изменённых товаров.")
                else:
                    df_new_idx = df_new.set_index("sku")
                    df_old_idx = df_old.set_index("sku")
                    common_skus = df_new_idx.index.intersection(df_old_idx.index)
                    if common_skus.empty:
                        st.info("Нет изменённых товаров.")
                    else:
                        df_new_common = df_new_idx.loc[common_skus]
                        new_sets = df_new_common["attribute set"].fillna("")
                        old_sets = df_old_idx.loc[common_skus, "attribute set"].fillna("")
                        diff_mask = new_sets != old_sets
                        df_changed = df_new_common.loc[diff_mask].reset_index()

                        if df_changed.empty:
                            st.info("Нет изменённых товаров.")
                        else:
                            api_base = st.session_state.get("ai_api_base")
                            attr_sets_map = st.session_state.get("ai_attr_sets_map")
                            setup_failed = False

                            try:
                                if not api_base:
                                    with st.spinner("🔌 Подключение к Magento…"):
                                        api_base = probe_api_base(session, base_url)
                                    st.session_state["ai_api_base"] = api_base
                                if not attr_sets_map and api_base:
                                    with st.spinner("📚 Загрузка attribute sets…"):
                                        attr_sets_map = get_attribute_sets_map(
                                            session, api_base
                                        )
                                    st.session_state["ai_attr_sets_map"] = attr_sets_map
                            except Exception as exc:  # pragma: no cover - UI interaction
                                st.warning(
                                    f"Не удалось подготовить подключение к Magento: {exc}"
                                )
                                setup_failed = True

                            if setup_failed or not api_base or not attr_sets_map:
                                st.session_state["show_attributes_trigger"] = False
                            else:
                                grouped = df_changed.groupby(
                                    "attribute set", dropna=False
                                )

                                for attr_set_value, group in grouped:
                                    attr_title = (
                                        attr_set_value if pd.notna(attr_set_value) else "—"
                                    )
                                    st.markdown(f"#### 🎯 Attribute Set: {attr_title}")

                                    for _, row in group.iterrows():
                                        sku_value = str(row.get("sku", "")).strip()
                                        if not sku_value:
                                            continue

                                        name_value = row.get("name")
                                        header = f"**{sku_value}**"
                                        if isinstance(name_value, str) and name_value.strip():
                                            header = (
                                                f"**{sku_value} — {name_value.strip()}**"
                                            )
                                        st.markdown(header)

                                        try:
                                            with st.spinner("Получение данных из Magento…"):
                                                product = get_product_by_sku(
                                                    session, api_base, sku_value
                                                )
                                                attr_set_id = None
                                                if pd.notna(attr_set_value):
                                                    attr_set_id = attribute_sets.get(
                                                        attr_set_value
                                                    )
                                                if attr_set_id is None:
                                                    attr_set_id = product.get(
                                                        "attribute_set_id"
                                                    )
                                                allowed = compute_allowed_attrs(
                                                    attr_set_id,
                                                    SET_ATTRS,
                                                    attr_sets_map or {},
                                                    ALWAYS_ATTRS,
                                                )
                                                if not allowed:
                                                    st.info("Нет атрибутов для отображения.")
                                                    continue
                                                allowed_sorted = sorted(allowed)
                                                df_full = collect_attributes_table(
                                                    product,
                                                    allowed_sorted,
                                                    session,
                                                    api_base,
                                                )
                                                if df_full.empty:
                                                    st.info(
                                                        "Нет данных по атрибутам, пропуск."
                                                    )
                                                    continue
                                        except Exception as exc:  # pragma: no cover - UI interaction
                                            st.warning(
                                                f"{sku_value}: не удалось получить атрибуты ({exc})"
                                            )
                                            continue

                                        df_display = build_attributes_display_table(df_full)
                                        st.dataframe(
                                            df_display,
                                            use_container_width=True,
                                        )
        else:
            st.info("Нет изменённых товаров.")
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
