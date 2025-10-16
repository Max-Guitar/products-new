from __future__ import annotations
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    collect_attributes_table,
    compute_allowed_attrs,
    get_attribute_meta,
    get_attribute_sets_map,
    get_product_by_sku,
    list_categories,
    probe_api_base,
)
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


_ATTRIBUTE_SET_ICONS = {
    "Accessories": "🧩",
    "Acoustic guitar": "🎻",
    "Amps": "🎚️",
    "Bass Guitar": "🎸",
    "Default": "🧾",
    "Effects": "🎛️",
    "Electric guitar": "🎸",
}
_DEFAULT_ATTRIBUTE_ICON = "🧩"


def _get_custom_attr_value(item: dict, code: str, default=None):
    for attr in item.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == code:
            return attr.get("value", default)
    return default


def _attr_label(meta: dict, code: str) -> str:
    for key in ("default_frontend_label", "frontend_label", "store_label"):
        label = meta.get(key)
        if isinstance(label, str) and label.strip():
            return label.strip()
    if code == "sku":
        return "SKU"
    if code == "name":
        return "Name"
    return code.replace("_", " ").title()


def _meta_options(meta: dict) -> list:
    options = []
    for opt in meta.get("options", []) or []:
        label = opt.get("label")
        if isinstance(label, str) and label.strip():
            options.append(label.strip())
    # preserve order but deduplicate while keeping first occurrence
    seen = set()
    unique = []
    for item in options:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _attr_set_icon(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return _DEFAULT_ATTRIBUTE_ICON
    return _ATTRIBUTE_SET_ICONS.get(name.strip(), _DEFAULT_ATTRIBUTE_ICON)


def _format_attr_set_title(name: str) -> str:
    clean_name = name.strip() if isinstance(name, str) and name.strip() else "—"
    icon = _attr_set_icon(clean_name)
    return f"{icon} {clean_name}"


def _format_category_label(name: str, cat_id: str) -> str:
    base = name.strip() if isinstance(name, str) and name.strip() else ""
    if not base:
        return f"#{cat_id}"
    return f"{base} (#{cat_id})"


def _prepare_category_options(raw_categories: list[dict]) -> list[dict]:
    prepared = []
    seen_labels = set()
    for item in raw_categories:
        cat_id = str(item.get("id", "")).strip()
        if not cat_id:
            continue
        name = str(item.get("name", "")).strip()
        label = _format_category_label(name, cat_id)
        if label in seen_labels:
            label = f"{label}#{cat_id}"
        seen_labels.add(label)
        prepared.append({"id": cat_id, "name": name or label, "label": label})
    prepared.sort(key=lambda x: (x["name"].lower(), x["id"]))
    return prepared


def _categories_ids_to_labels(values, id_to_label: dict[str, str]) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    elif values in (None, ""):
        raw_values = []
    else:
        raw_values = [values]

    labels = []
    for value in raw_values:
        cat_id = str(value).strip()
        if not cat_id:
            continue
        label = id_to_label.get(cat_id, _format_category_label("", cat_id))
        if label not in labels:
            labels.append(label)
    return labels


def _split_multiselect_input(values) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        iterable = [str(item).strip() for item in values]
    elif isinstance(values, str):
        iterable = [item.strip() for item in values.split(",")]
    elif values in (None, ""):
        iterable = []
    elif isinstance(values, float) and pd.isna(values):
        iterable = []
    else:
        iterable = [str(values).strip()]

    return [item for item in iterable if item]


def _format_multiselect_display_value(values) -> str:
    parts = _split_multiselect_input(values)
    if not parts:
        return ""
    return ", ".join(parts)


def _categories_labels_to_ids(values, label_to_id: dict[str, str]) -> list[str]:
    raw_values = _split_multiselect_input(values)

    ids = []
    for value in raw_values:
        label = str(value).strip()
        if not label:
            continue
        if label in label_to_id:
            cat_id = label_to_id[label]
        elif label.startswith("#"):
            cat_id = label.lstrip("#")
        else:
            cat_id = label
        if cat_id not in ids:
            ids.append(cat_id)
    return ids


def _convert_df_for_storage(
    df: pd.DataFrame,
    label_to_id: dict[str, str],
    multiselect_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    storage = df.copy(deep=True)

    columns_to_process = set(multiselect_columns or [])
    for column in columns_to_process:
        if column in storage.columns and column != "categories":
            storage[column] = storage[column].apply(_split_multiselect_input)

    if "categories" in storage.columns:
        storage["categories"] = storage["categories"].apply(
            lambda values: _categories_labels_to_ids(values, label_to_id)
        )
    return storage


def _value_for_editor(attr_row: dict, meta: dict):
    frontend_input = (meta.get("frontend_input") or "").lower()
    raw = attr_row.get("raw_value") if attr_row else None
    label = attr_row.get("label") if attr_row else None

    if frontend_input == "boolean":
        if isinstance(raw, bool):
            return raw
        if raw in (None, ""):
            return False
        raw_str = str(raw).strip().lower()
        return raw_str in {"1", "true", "yes", "y", "on"}

    if frontend_input == "multiselect":
        selected = []
        values = []
        if isinstance(raw, (list, tuple, set)):
            values = [str(item) for item in raw]
        elif raw not in (None, ""):
            values = [item.strip() for item in str(raw).split(",")]
        elif label:
            values = [item.strip() for item in str(label).split(",")]

        id_to_label = {}
        for opt in meta.get("options", []) or []:
            opt_value = opt.get("value")
            opt_label = opt.get("label")
            if opt_value is not None and opt_label:
                id_to_label[str(opt_value)] = str(opt_label)

        for value in values:
            if not value:
                continue
            selected.append(id_to_label.get(str(value), str(value)))
        return selected

    if frontend_input == "select":
        id_to_label = {}
        for opt in meta.get("options", []) or []:
            opt_value = opt.get("value")
            opt_label = opt.get("label")
            if opt_value is not None and opt_label:
                id_to_label[str(opt_value)] = str(opt_label)

        if raw not in (None, "") and str(raw) in id_to_label:
            return id_to_label[str(raw)]
        if label:
            return str(label)
        if raw not in (None, ""):
            return str(raw)
        return ""

    value = label if label not in (None, "") else raw
    if value is None:
        return ""
    return str(value)


def _normalize_editor_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _sanitize_for_storage(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set)):
        sanitized = [str(item).strip() for item in value if str(item).strip()]
        return sanitized
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _build_column_config(column_order, column_meta):
    config = {}
    for column in column_order:
        if column == "sku":
            config[column] = st.column_config.TextColumn("SKU", disabled=True)
            continue
        if column == "name":
            config[column] = st.column_config.TextColumn("Name", disabled=True)
            continue
        meta = column_meta.get(column, {})
        label = _attr_label(meta, column)
        frontend_input = (meta.get("frontend_input") or "").lower()
        options = _meta_options(meta)
        if frontend_input == "boolean":
            config[column] = st.column_config.CheckboxColumn(label)
        elif frontend_input == "select" and options:
            config[column] = st.column_config.SelectboxColumn(
                label=label,
                options=options,
                required=False,
            )
        elif frontend_input == "multiselect":
            help_parts = ["Comma-separated values"]
            if options:
                preview = ", ".join(options[:5])
                help_parts.append(f"Options include: {preview}")
            config[column] = st.column_config.Column(
                label=label,
                help=". ".join(help_parts),
            )
        else:
            config[column] = st.column_config.TextColumn(label)
    return config


def _update_step2_edits(step2_edits, original_df, edited_df, editable_columns):
    if original_df.empty or edited_df.empty:
        return

    original_idx = original_df.set_index("sku")
    edited_idx = edited_df.set_index("sku")

    for sku, orig_row in original_idx.iterrows():
        if sku not in edited_idx.index:
            continue
        edited_row = edited_idx.loc[sku]
        changes = {}
        for column in editable_columns:
            if column not in original_idx.columns or column not in edited_idx.columns:
                continue
            orig_value = orig_row[column]
            edited_value = edited_row[column]
            if _normalize_editor_value(orig_value) != _normalize_editor_value(
                edited_value
            ):
                changes[column] = _sanitize_for_storage(edited_value)
        if changes:
            step2_edits[sku] = changes
        elif sku in step2_edits:
            step2_edits.pop(sku, None)


def _prepare_step2_tables(
    df_changed,
    session,
    api_base,
    attribute_sets,
    attr_sets_map,
    category_options,
):
    tables = []
    core_codes = ["sku", "name", "brand", "condition"]
    meta_cache = {}

    id_to_label = {item["id"]: item["label"] for item in category_options}
    label_to_id = {item["label"]: item["id"] for item in category_options}
    category_labels = [item["label"] for item in category_options]

    grouped = df_changed.groupby("attribute set", dropna=False)
    for attr_set_value, group in grouped:
        rows = []
        column_meta = {}
        attr_codes_seen = set()
        attr_title = attr_set_value if pd.notna(attr_set_value) else "—"

        for _, row in group.iterrows():
            sku_value = str(row.get("sku", "")).strip()
            if not sku_value:
                continue

            name_value = row.get("name")
            try:
                with st.spinner("Получение данных из Magento…"):
                    product = get_product_by_sku(session, api_base, sku_value)
            except Exception as exc:  # pragma: no cover - UI interaction
                st.warning(f"{sku_value}: не удалось получить атрибуты ({exc})")
                continue

            attr_set_id = None
            if pd.notna(attr_set_value):
                attr_set_id = attribute_sets.get(attr_set_value)
            if attr_set_id is None:
                attr_set_id = product.get("attribute_set_id")

            allowed = compute_allowed_attrs(
                attr_set_id,
                SET_ATTRS,
                attr_sets_map or {},
                ALWAYS_ATTRS,
            )

            editor_codes = []
            for code in core_codes:
                if code not in editor_codes:
                    editor_codes.append(code)
            for code in sorted(allowed):
                if code not in editor_codes:
                    editor_codes.append(code)

            df_full = collect_attributes_table(
                product,
                editor_codes,
                session,
                api_base,
            )
            attr_rows = df_full.to_dict(orient="index") if not df_full.empty else {}

            row_data = {
                "sku": sku_value,
                "name": str(name_value).strip()
                if isinstance(name_value, str) and name_value.strip()
                else str(product.get("name", "")),
            }

            for code in editor_codes:
                if code in {"sku", "name"}:
                    continue
                meta = meta_cache.get(code)
                if meta is None:
                    meta = get_attribute_meta(session, api_base, code) or {}
                    meta_cache[code] = meta
                column_meta.setdefault(code, meta)
                value = _value_for_editor(attr_rows.get(code, {}), meta)
                row_data[code] = value
                attr_codes_seen.add(code)

            category_links = (
                (product.get("extension_attributes") or {}).get("category_links") or []
            )
            categories = [
                str(link.get("category_id", "")).strip()
                for link in category_links
                if str(link.get("category_id", "")).strip()
            ]
            row_data["categories"] = _categories_ids_to_labels(categories, id_to_label)

            rows.append(row_data)

        if not rows:
            continue

        df_table = pd.DataFrame(rows)
        if "sku" not in df_table.columns:
            df_table["sku"] = ""
        if "name" not in df_table.columns:
            df_table["name"] = ""

        df_table["sku"] = df_table["sku"].astype(str)

        for code in core_codes:
            if code not in df_table.columns:
                df_table[code] = ""

        if "categories" not in df_table.columns:
            df_table["categories"] = [[] for _ in range(len(df_table))]

        other_columns = sorted(
            code
            for code in set(column_meta.keys()) | attr_codes_seen
            if code not in {"brand", "condition", "sku", "name", "categories"}
        )

        column_order = [col for col in core_codes if col in df_table.columns]
        column_order.extend([col for col in other_columns if col in df_table.columns])
        if "categories" in df_table.columns:
            column_order.append("categories")

        df_table = df_table[column_order]

        column_meta_with_base = {code: column_meta.get(code, {}) for code in column_order}
        column_config = _build_column_config(column_order, column_meta_with_base)
        multiselect_columns = [
            column
            for column, meta in column_meta_with_base.items()
            if (meta.get("frontend_input") or "").lower() == "multiselect"
        ]
        category_help = "Comma-separated category labels"
        if category_labels:
            preview = ", ".join(category_labels[:5])
            category_help += f". Examples: {preview}"
        if "categories" in column_order:
            column_config["categories"] = st.column_config.Column(
                "Categories",
                help=category_help,
            )
        display_df = df_table.copy(deep=True)
        for column in multiselect_columns:
            if column in display_df.columns:
                display_df[column] = display_df[column].apply(
                    _format_multiselect_display_value
                )
        if "categories" in display_df.columns:
            display_df["categories"] = display_df["categories"].apply(
                _format_multiselect_display_value
            )
        editable_columns = [col for col in column_order if col not in {"sku", "name"}]

        storage_df = _convert_df_for_storage(
            df_table, label_to_id, multiselect_columns
        )

        tables.append(
            {
                "id": f"table::{attr_title}",
                "title": attr_title,
                "display_title": _format_attr_set_title(attr_title),
                "data": display_df,
                "storage_df": storage_df,
                "storage_df_original": storage_df.copy(deep=True),
                "column_config": column_config,
                "column_order": column_order,
                "editable_columns": editable_columns,
                "category_label_to_id": label_to_id,
                "multiselect_columns": multiselect_columns,
            }
        )

    return tables


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
        if isinstance(edited_df, pd.DataFrame) and st.button("Show Attributes"):
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
                                need_rebuild = show_attributes_clicked or not st.session_state.get(
                                    "step2_tables"
                                )
                                if need_rebuild:
                                    categories_options = st.session_state.get(
                                        "step2_category_options"
                                    )
                                    if not categories_options:
                                        try:
                                            with st.spinner("📂 Загрузка категорий…"):
                                                raw_categories = list_categories(
                                                    session, api_base
                                                )
                                            categories_options = _prepare_category_options(
                                                raw_categories
                                            )
                                        except Exception as exc:  # pragma: no cover - UI interaction
                                            st.warning(
                                                f"Не удалось загрузить категории: {exc}"
                                            )
                                            categories_options = []
                                        st.session_state[
                                            "step2_category_options"
                                        ] = categories_options
                                    tables = _prepare_step2_tables(
                                        df_changed,
                                        session,
                                        api_base,
                                        attribute_sets,
                                        attr_sets_map,
                                        categories_options,
                                    )
                                    st.session_state["step2_tables"] = tables
                                    st.session_state["step2_edits"] = {}
                                    editor_keys_to_clear = [
                                        key
                                        for key in list(st.session_state.keys())
                                        if key.startswith("step2_editor_")
                                    ]
                                    for editor_key in editor_keys_to_clear:
                                        st.session_state.pop(editor_key, None)
                                    st.session_state.pop("editor_state", None)

                                tables = st.session_state.get("step2_tables", [])
                                if not tables:
                                    st.info("Нет атрибутов для отображения.")
                                else:
                                    step2_edits = st.session_state.setdefault(
                                        "step2_edits", {}
                                    )
                                    for idx, entry in enumerate(tables):
                                        attr_title = entry.get("display_title")
                                        if not attr_title:
                                            attr_title = _format_attr_set_title(
                                                entry.get("title", "—")
                                            )
                                        st.subheader(attr_title)

                                        entry_id = entry.get("id")
                                        base_df = entry.get("data")
                                        if not isinstance(base_df, pd.DataFrame):
                                            base_df = pd.DataFrame()

                                        column_config = entry.get("column_config", {})
                                        column_order = entry.get("column_order")
                                        unique_id = entry_id or f"table::{idx}"
                                        editor_key = f"step2_editor_{unique_id}"

                                        if editor_key not in st.session_state:
                                            st.session_state[editor_key] = (
                                                base_df.copy(deep=True)
                                            )

                                        edited_df = st.data_editor(
                                            data=st.session_state[editor_key],
                                            column_config=column_config,
                                            column_order=column_order,
                                            use_container_width=True,
                                            num_rows="fixed",
                                            hide_index=True,
                                            key=editor_key,
                                        )

                                        if isinstance(edited_df, pd.DataFrame):
                                            st.session_state[editor_key] = edited_df.copy(
                                                deep=True
                                            )

                                        current_df = st.session_state[editor_key]
                                        edited_copy = current_df.copy(deep=True)

                                        editable_columns = entry.get(
                                            "editable_columns", []
                                        )
                                        storage_df_original = entry.get(
                                            "storage_df_original",
                                            entry.get("storage_df", pd.DataFrame()),
                                        )
                                        if isinstance(
                                            storage_df_original, pd.DataFrame
                                        ):
                                            original_storage = storage_df_original.copy(
                                                deep=True
                                            )
                                        else:
                                            original_storage = pd.DataFrame()
                                        label_to_id = entry.get(
                                            "category_label_to_id", {}
                                        )
                                        multiselect_columns = entry.get(
                                            "multiselect_columns", []
                                        )

                                        edited_storage = _convert_df_for_storage(
                                            edited_copy,
                                            label_to_id,
                                            multiselect_columns,
                                        )
                                        entry["storage_df"] = edited_storage.copy(
                                            deep=True
                                        )

                                        changes_store = step2_edits.setdefault(
                                            "_changes", {}
                                        )
                                        _update_step2_edits(
                                            changes_store,
                                            original_storage,
                                            edited_storage,
                                            editable_columns,
                                        )

                                        if "_changes" in step2_edits:
                                            step2_edits["_changes"] = {
                                                sku: values
                                                for sku, values in step2_edits[
                                                    "_changes"
                                                ].items()
                                                if values
                                            }
        else:
            st.info("Нет изменённых товаров.")
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")
