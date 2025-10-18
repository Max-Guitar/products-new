from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import quote
import re

import pandas as pd
import streamlit as st

from connectors.magento.attributes import AttributeMetaCache
from connectors.magento.categories import ensure_categories_meta
from connectors.magento.client import get_default_products
from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    HTMLResponseError,
    collect_attributes_table,
    compute_allowed_attrs,
    get_attribute_sets_map,
    get_product_by_sku,
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
from services.normalizers import normalize_for_magento
from utils.http import build_magento_headers, get_session


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


ID_RX = re.compile(r"#(\d+)\)?$")


DEBUG = False


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
        if not isinstance(opt, dict):
            continue
        label = opt.get("label")
        if not isinstance(label, str) or not label.strip():
            label = str(opt.get("value", "")).strip()
        else:
            label = label.strip()
        if label:
            options.append(label)
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


def _parse_category_token(token: object) -> int | None:
    if token is None:
        return None
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, float):
        if math.isnan(token):
            return None
        try:
            return int(token)
        except (TypeError, ValueError):
            return None
    text = str(token).strip()
    if not text:
        return None
    match = ID_RX.search(text)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _cat_to_ids(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[int] = []
        for item in value:
            cid = _parse_category_token(item)
            if cid is not None and cid not in result:
                result.append(cid)
        return result
    cid = _parse_category_token(value)
    return [cid] if cid is not None else []


def _ids_to_labels(ids: list[int], values_to_labels: dict[str, str]) -> list[str]:
    labels: list[str] = []
    for cid in ids or []:
        key = str(cid)
        label = values_to_labels.get(key)
        if label and label not in labels:
            labels.append(label)
    return labels


def _labels_to_ids(labels: list[str], options_map: dict[str, int]) -> list[int]:
    if not labels:
        return []
    ids: list[int] = []
    for label in labels:
        if label is None:
            continue
        text = str(label).strip()
        if not text:
            continue
        if text in options_map:
            cid = options_map[text]
        else:
            lower = text.casefold()
            cid = options_map.get(lower)
            if cid is None:
                try:
                    cid = int(text)
                except (TypeError, ValueError):
                    continue
        if isinstance(cid, float) and math.isnan(cid):
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        if cid_int not in ids:
            ids.append(cid_int)
    return ids


def _categories_value_to_ids(value: object, options_map: dict[str, int]) -> list[int]:
    ids = _cat_to_ids(value)
    if ids:
        return ids
    labels: list[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                labels.append(text)
    else:
        labels = _split_multiselect_input(value)
    labels = [label for label in labels if label]
    if not labels:
        return []
    return _labels_to_ids(labels, options_map)


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
        options_map: dict[str, int] = {}
        for key, value in (label_to_id or {}).items():
            try:
                options_map[key] = int(value)
                options_map[key.casefold()] = int(value)
            except (AttributeError, TypeError, ValueError):
                continue

        storage["categories"] = storage["categories"].apply(
            lambda labels: _labels_to_ids(
                list(labels) if isinstance(labels, (list, tuple, set)) else _split_multiselect_input(labels),
                options_map,
            )
        )
    return storage


def _value_for_editor(attr_row: dict, meta: dict):
    frontend_input = (meta.get("frontend_input") or "").lower()
    raw = attr_row.get("raw_value") if attr_row else None
    label = attr_row.get("label") if attr_row else None
    value_to_label = meta.get("values_to_labels") or {}

    def _lookup(value):
        if value is None:
            return None
        candidates = []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            candidates.append(stripped)
            candidates.append(stripped.lower())
            try:
                candidates.append(int(stripped))
            except (TypeError, ValueError):
                pass
        else:
            candidates.append(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidates.append(int(value))
                candidates.append(str(int(value)))
        for candidate in candidates:
            if candidate in value_to_label:
                mapped = value_to_label[candidate]
                if isinstance(mapped, str):
                    return mapped
                return str(mapped)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    if frontend_input == "boolean":
        if isinstance(raw, bool):
            return raw
        if raw in (None, ""):
            return False
        raw_str = str(raw).strip().lower()
        return raw_str in {"1", "true", "yes", "y", "on"}

    if frontend_input == "multiselect":
        selected = []
        values: list[str] = []
        if isinstance(raw, (list, tuple, set)):
            values = [str(item) for item in raw]
        elif raw not in (None, ""):
            values = [item.strip() for item in str(raw).split(",")]
        elif label:
            values = [item.strip() for item in str(label).split(",")]

        for value in values:
            if not value:
                continue
            mapped = _lookup(value)
            if mapped is None:
                mapped = str(value)
            selected.append(mapped)
        return selected

    if frontend_input == "select":
        mapped = _lookup(raw)
        if mapped:
            return mapped
        if label:
            return str(label)
        if raw not in (None, ""):
            return str(raw)
        return ""

    value = label if label not in (None, "") else raw
    if value is None:
        return ""
    if isinstance(value, str):
        return value
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
        if column == "attribute set":
            config[column] = st.column_config.TextColumn(
                "Attribute Set", disabled=True
            )
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


def _build_column_config_for_step1_like(step: str) -> tuple[dict, list[str]]:
    if step == "step1":
        config = st.session_state.get("step1_column_config_cache", {})
        disabled = st.session_state.get("step1_disabled_cols_cache", [])
        return config, disabled
    if step == "step2":
        return {}, []
    return {}, []


def _reset_step2_state():
    st.session_state.pop("step2", None)
    st.session_state.pop("show_attrs", None)


def _ensure_step2_state() -> dict:
    state = st.session_state.setdefault("step2", {})
    state.setdefault("dfs", {})
    state.setdefault("original", {})
    state.setdefault("column_config", {})
    state.setdefault("disabled", {})
    state.setdefault("row_meta", {})
    state.setdefault("set_names", {})
    state.setdefault("wide", {})
    state.setdefault("wide_orig", {})
    state.setdefault("wide_meta", {})
    state.setdefault("wide_colcfg", {})
    state.setdefault("wide_synced", {})
    state.setdefault("staged", {})
    return state


def _build_value_column_config(row_meta_map: dict[str, dict]):
    if not row_meta_map:
        return st.column_config.Column("Value")

    frontend_inputs = {
        (meta.get("frontend_input") or meta.get("backend_type") or "text").lower()
        for meta in row_meta_map.values()
    }

    frontend_inputs.discard("")

    def _options_union() -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()
        for meta in row_meta_map.values():
            for option in meta.get("options", []) or []:
                if isinstance(option, str):
                    label = option.strip()
                else:
                    label = ""
                if not label or label in seen:
                    continue
                seen.add(label)
                collected.append(label)
        return collected

    if len(frontend_inputs) == 1:
        input_type = frontend_inputs.pop()
        options = _options_union()
        if input_type == "select":
            return st.column_config.SelectboxColumn("Value", options=options)
        if input_type == "multiselect":
            return st.column_config.MultiselectColumn("Value", options=options)
        if input_type == "boolean":
            return st.column_config.CheckboxColumn("Value")
        if input_type in {"int", "integer", "decimal", "price", "float"}:
            return st.column_config.NumberColumn("Value")
        return st.column_config.TextColumn("Value")

    return st.column_config.Column("Value")


def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert attribute rows to a wide table with one row per SKU."""

    if df_long.empty:
        return pd.DataFrame(columns=["sku", "name"]).astype(object)

    base = df_long[["sku", "name"]].drop_duplicates().set_index("sku")
    pv = df_long.pivot_table(
        index="sku",
        columns="attribute_code",
        values="value",
        aggfunc=lambda x: x.iloc[-1],
    )
    out = base.join(pv, how="left").reset_index()
    for column in out.columns:
        out[column] = out[column].astype(object)
    return out


def apply_wide_edits_to_long(
    df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> pd.DataFrame:
    """Propagate edits from a wide table back to the long representation."""

    if not isinstance(df_long, pd.DataFrame) or df_long.empty:
        return df_long.copy(deep=True) if isinstance(df_long, pd.DataFrame) else df_long

    if not isinstance(df_wide, pd.DataFrame) or df_wide.empty:
        return df_long.copy(deep=True)

    if "sku" not in df_wide.columns:
        return df_long.copy(deep=True)

    wide_indexed = df_wide.set_index("sku")
    updated = df_long.copy(deep=True)

    for row_idx, row in updated.iterrows():
        sku = row.get("sku")
        code = row.get("attribute_code")
        if not sku or not code:
            continue
        if sku not in wide_indexed.index:
            continue
        if code not in wide_indexed.columns:
            continue
        updated.at[row_idx, "value"] = wide_indexed.at[sku, code]

    return updated


def _ensure_wide_meta_options(
    meta_map: dict[str, dict], sample_df: pd.DataFrame | None
) -> None:
    if not isinstance(meta_map, dict):
        return
    for meta in meta_map.values():
        if not isinstance(meta, dict):
            continue
        options = meta.get("options")
        if isinstance(options, list):
            continue
        meta["options"] = []


def _refresh_wide_meta_from_cache(
    meta_cache: AttributeMetaCache | None,
    meta_map: dict[str, dict],
    attr_codes: list[str],
    wide_df: pd.DataFrame | None,
) -> None:
    if not isinstance(meta_map, dict) or not attr_codes:
        return

    if not isinstance(meta_cache, AttributeMetaCache):
        return

    unique_codes: list[str] = []
    for raw in attr_codes:
        if not isinstance(raw, str):
            continue
        code = raw.strip()
        if not code or code in unique_codes:
            continue
        unique_codes.append(code)

    target_codes = [code for code in unique_codes if code != "categories"]
    if target_codes:
        meta_cache.build_and_set_static_for(target_codes, store_id=0)

    for code in unique_codes:
        cached_meta = meta_cache.get(code)
        if isinstance(cached_meta, dict):
            meta_map[code] = cached_meta


def _coerce_for_ui(df: pd.DataFrame, meta_map: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for code, meta in (meta_map or {}).items():
        t = (meta.get("frontend_input") or meta.get("backend_type") or "text").lower()
        if code not in out.columns:
            continue
        if t == "boolean":
            out[code] = out[code].apply(
                lambda v: str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
                if v not in (None, "")
                else False
            )
        elif t == "multiselect":
            if code == "categories":
                values_to_labels = meta.get("values_to_labels") or {}
                raw_map = meta.get("options_map") or {}
                options_map: dict[str, int] = {}
                if isinstance(raw_map, dict):
                    for key, value in raw_map.items():
                        if not isinstance(key, str):
                            continue
                        options_map[key] = value
                        options_map[key.casefold()] = value
                        try:
                            options_map[str(int(value))] = int(value)
                        except (TypeError, ValueError):
                            continue
                out[code] = out[code].apply(
                    lambda v: _ids_to_labels(
                        _categories_value_to_ids(v, options_map),
                        values_to_labels,
                    )
                )
            else:
                out[code] = out[code].apply(
                    lambda v: v
                    if isinstance(v, (list, tuple, set))
                    else ([] if v in (None, "") else [str(v).strip()])
                )
        else:
            out[code] = out[code].astype(object)
    return out


def _to_list_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            if isinstance(item, float) and math.isnan(item):
                continue
            normalized.append(str(item))
        return normalized
    return [str(value)]


def _ensure_list_of_strings(value: object) -> list[str]:
    return _to_list_str(value)


def _ensure_config_options(
    column: str, cfg_obj, options: list[str]
):  # pragma: no cover - UI helper
    if not options:
        return cfg_obj
    if hasattr(cfg_obj, "_config") and isinstance(cfg_obj._config, dict):
        cfg_obj._config["options"] = options
        return cfg_obj
    try:
        cfg_obj.options = options
        return cfg_obj
    except Exception:
        label = (
            getattr(cfg_obj, "label", None)
            or getattr(cfg_obj, "_label", None)
            or column
        )
        if isinstance(cfg_obj, st.column_config.SelectboxColumn):
            return st.column_config.SelectboxColumn(label, options=options)
        if isinstance(cfg_obj, st.column_config.MultiselectColumn):
            return st.column_config.MultiselectColumn(label, options=options)
    return cfg_obj


def _coerce_column_for_config(
    df: pd.DataFrame, column: str, cfg_obj
) -> tuple[pd.DataFrame, object]:  # pragma: no cover - UI helper
    if isinstance(cfg_obj, st.column_config.CheckboxColumn):
        df[column] = df[column].fillna(False).astype(bool)
        return df, cfg_obj
    if isinstance(cfg_obj, st.column_config.SelectboxColumn):
        df[column] = df[column].astype("string")
        options = getattr(cfg_obj, "options", None) or []
        str_options = [str(opt) for opt in options]
        return df, _ensure_config_options(column, cfg_obj, str_options)
    if isinstance(cfg_obj, st.column_config.MultiselectColumn):
        df[column] = df[column].apply(_ensure_list_of_strings)
        options = getattr(cfg_obj, "options", None) or []
        str_options = [str(opt) for opt in options]
        return df, _ensure_config_options(column, cfg_obj, str_options)
    return df, cfg_obj


def _probe_editor_groups(
    df: pd.DataFrame,
    columns: list[str],
    column_config: dict[str, object],
    disabled: list[str],
    key_prefix: str,
):  # pragma: no cover - UI helper
    def _try(columns_chunk: list[str], suffix: str) -> bool:
        sub_df = df[columns_chunk]
        sub_cfg = {
            col: column_config[col]
            for col in columns_chunk
            if col in column_config
        }
        sub_disabled = [col for col in disabled if col in columns_chunk]
        try:
            st.data_editor(
                sub_df,
                column_config=sub_cfg,
                disabled=sub_disabled,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key=f"{key_prefix}::{suffix}",
            )
        except Exception as exc:
            st.warning(
                f"Editor failed for columns {columns_chunk}: {exc}")
            return False
        return True

    def _split(columns_chunk: list[str], suffix: str) -> set[str]:
        if not columns_chunk:
            return set()
        if _try(columns_chunk, suffix):
            return set()
        if len(columns_chunk) == 1:
            return {columns_chunk[0]}
        mid = len(columns_chunk) // 2
        left = _split(columns_chunk[:mid], suffix + "L")
        right = _split(columns_chunk[mid:], suffix + "R")
        return left | right

    return _split(columns, "root")


def _apply_categories_fallback(meta_map: dict[str, dict]) -> None:
    if not isinstance(meta_map, dict):
        return
    if not st.session_state.get("step2_categories_failed"):
        return
    meta = meta_map.get("categories")
    if isinstance(meta, dict):
        meta["frontend_input"] = "text"
        meta.pop("options", None)


def colcfg_from_meta(
    code: str, meta: dict | None, sample_df: pd.DataFrame | None = None
):
    meta = meta or {}
    column_label = _attr_label(meta, code)
    ftype_raw = meta.get("frontend_input") or meta.get("backend_type") or "text"
    ftype = str(ftype_raw).lower()
    labels = [
        opt.get("label")
        for opt in (meta.get("options") or [])
        if isinstance(opt, dict) and opt.get("label")
    ]

    if ftype == "multiselect" and labels:
        return st.column_config.MultiselectColumn(column_label, options=labels)
    if ftype == "boolean":
        return st.column_config.CheckboxColumn(column_label)
    if ftype in {"select", "dropdown"} and labels:
        return st.column_config.SelectboxColumn(column_label, options=labels)
    if ftype in {"int", "integer", "decimal", "price", "float"}:
        return st.column_config.NumberColumn(column_label)
    return st.column_config.TextColumn(column_label)


def build_wide_colcfg(
    wide_meta: dict[str, dict], sample_df: pd.DataFrame | None = None
):
    cfg = {
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "name": st.column_config.TextColumn("Name", disabled=True),
    }

    for code, original_meta in list(wide_meta.items()):
        meta = original_meta or {}
        cfg[code] = colcfg_from_meta(code, meta)

    return cfg


def build_attributes_df(
    df_changed: pd.DataFrame,
    session,
    api_base: str,
    attribute_sets: dict,
    attr_sets_map: dict,
    meta_cache: AttributeMetaCache,
):
    core_codes = ["brand", "condition"]
    rows_by_set: dict[int, list[dict]] = defaultdict(list)
    row_meta_by_set: dict[int, dict[str, dict]] = defaultdict(dict)
    set_names: dict[int, str] = {}

    cats_meta = meta_cache.get("categories") if isinstance(meta_cache, AttributeMetaCache) else {}
    if not isinstance(cats_meta, dict):
        cats_meta = {}
    cat_values_to_labels: dict[str, str] = cats_meta.get("values_to_labels") or {}
    raw_options_map: dict[str, int] = cats_meta.get("options_map") or {}
    cat_options_map: dict[str, int] = {}
    for key, value in raw_options_map.items():
        if not isinstance(key, str):
            continue
        cat_options_map[key] = value
        cat_options_map[key.casefold()] = value
        try:
            cat_options_map[str(int(value))] = int(value)
        except (TypeError, ValueError):
            continue
    cat_labels = [
        opt.get("label")
        for opt in (cats_meta.get("options") or [])
        if isinstance(opt, dict) and opt.get("label")
    ]

    for _, row in df_changed.iterrows():
        sku_value = str(row.get("sku", "")).strip()
        if not sku_value:
            continue

        name_value = row.get("name")
        attr_set_value = row.get("attribute set")
        try:
            with st.spinner(f"🔄 {sku_value}: загрузка атрибутов…"):
                product = get_product_by_sku(session, api_base, sku_value)
        except Exception as exc:  # pragma: no cover - UI interaction
            st.warning(f"{sku_value}: не удалось получить атрибуты ({exc})")
            continue

        attr_set_id = None
        if pd.notna(attr_set_value):
            attr_set_id = attribute_sets.get(attr_set_value)
        if attr_set_id is None:
            attr_set_id = product.get("attribute_set_id")

        if attr_set_id is None:
            continue

        try:
            attr_set_id_int = int(attr_set_id)
        except (TypeError, ValueError):
            continue

        set_names[attr_set_id_int] = attr_sets_map.get(attr_set_id_int, str(attr_set_id_int))

        allowed = compute_allowed_attrs(
            attr_set_id_int,
            SET_ATTRS,
            attr_sets_map or {},
            ALWAYS_ATTRS,
        )

        editor_codes: list[str] = []
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

        loadable_codes = [
            code
            for code in editor_codes
            if code not in {"sku", "name"}
        ]
        meta_cache.load(loadable_codes)

        base_name = (
            str(name_value).strip()
            if isinstance(name_value, str) and name_value.strip()
            else str(product.get("name", ""))
        )

        for code in editor_codes:
            if code in {"sku", "name", "attribute set"}:
                continue
            meta = meta_cache.get(code) or {}
            value = _value_for_editor(attr_rows.get(code, {}), meta)
            frontend_input = (
                meta.get("frontend_input")
                or meta.get("backend_type")
                or "text"
            ).lower()

            if frontend_input in {"int", "integer"}:
                try:
                    value = int(str(value).strip()) if str(value).strip() else None
                except (TypeError, ValueError):
                    value = str(value).strip() if isinstance(value, str) else value
            elif frontend_input in {"decimal", "price", "float"}:
                try:
                    value = float(str(value).strip()) if str(value).strip() else None
                except (TypeError, ValueError):
                    value = str(value).strip() if isinstance(value, str) else value
            elif frontend_input == "boolean":
                value = bool(value)
            elif frontend_input == "multiselect":
                if not isinstance(value, list):
                    value = _split_multiselect_input(value)
            elif value is None:
                value = ""

            row_id = f"{sku_value}|{code}"
            row_meta_by_set[attr_set_id_int][row_id] = {
                "attribute_code": code,
                "frontend_input": frontend_input,
                "backend_type": (meta.get("backend_type") or "").lower(),
                "options": _meta_options(meta),
                "valid_examples": meta.get("valid_examples") or [],
            }
            rows_by_set[attr_set_id_int].append(
                {
                    "__row_id": row_id,
                    "sku": sku_value,
                    "name": base_name,
                    "attribute_code": code,
                    "value": value,
                }
            )

        category_links = (
            (product.get("extension_attributes") or {}).get("category_links") or []
        )
        categories = [
            str(link.get("category_id", "")).strip()
            for link in category_links
            if str(link.get("category_id", "")).strip()
        ]
        categories_ids = _cat_to_ids(categories)
        if not categories_ids and categories:
            categories_ids = _labels_to_ids(categories, cat_options_map)
        category_labels = _ids_to_labels(categories_ids, cat_values_to_labels)
        row_id = f"{sku_value}|categories"
        row_meta_by_set[attr_set_id_int][row_id] = {
            "attribute_code": "categories",
            "frontend_input": "multiselect",
            "backend_type": "text",
            "options": cat_labels,
            "valid_examples": category_labels[:5],
        }
        rows_by_set[attr_set_id_int].append(
            {
                "__row_id": row_id,
                "sku": sku_value,
                "name": base_name,
                "attribute_code": "categories",
                "value": category_labels,
            }
        )

    if not rows_by_set:
        return {}, {}, {}, meta_cache, {}, {}

    dfs: dict[int, pd.DataFrame] = {}
    column_configs: dict[int, dict] = {}
    disabled: dict[int, list[str]] = {}

    for set_id, rows in rows_by_set.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df.sort_values(["sku", "attribute_code"]).reset_index(drop=True)
        df = df.set_index("__row_id", drop=True)
        df = df[["sku", "name", "attribute_code", "value"]]
        dfs[set_id] = df
        column_configs[set_id] = {
            "sku": st.column_config.TextColumn("SKU", disabled=True),
            "name": st.column_config.TextColumn("Name", disabled=True),
            "attribute_code": st.column_config.TextColumn(
                "Attribute Code", disabled=True
            ),
            "value": _build_value_column_config(row_meta_by_set.get(set_id, {})),
        }
        disabled[set_id] = ["sku", "name", "attribute_code"]

    return dfs, column_configs, disabled, meta_cache, row_meta_by_set, set_names

def _is_blank_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return all(_is_blank_value(item) for item in value)
    return False


def _normalize_seq_for_compare(value: object) -> tuple:
    if not isinstance(value, (list, tuple, set)):
        return ()
    normalized: list[object] = []
    for item in value:
        if _is_blank_value(item):
            continue
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue
            normalized.append(stripped)
        else:
            normalized.append(item)
    if not normalized:
        return ()
    try:
        return tuple(sorted(normalized, key=lambda x: str(x)))
    except Exception:  # pragma: no cover - safety fallback
        return tuple(normalized)


def _values_equal(left: object, right: object) -> bool:
    if _is_blank_value(left) and _is_blank_value(right):
        return True
    if isinstance(left, float) and pd.isna(left):
        left = None
    if isinstance(right, float) and pd.isna(right):
        right = None
    if isinstance(left, str):
        left = left.strip()
    if isinstance(right, str):
        right = right.strip()
    if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
        return _normalize_seq_for_compare(left) == _normalize_seq_for_compare(right)
    return left == right


def _df_differs(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        return isinstance(a, pd.DataFrame) != isinstance(b, pd.DataFrame)

    a2 = a.copy(deep=True)
    b2 = b.copy(deep=True)
    cols = list(dict.fromkeys(list(a2.columns) + list(b2.columns)))
    a2 = a2.reindex(columns=cols).astype(object)
    b2 = b2.reindex(columns=cols).astype(object)

    if len(a2) != len(b2):
        return True

    for col in cols:
        la = list(a2[col]) if col in a2 else [None] * len(a2)
        lb = list(b2[col]) if col in b2 else [None] * len(b2)
        for left, right in zip(la, lb):
            if not _values_equal(left, right):
                return True

    return False


def apply_product_update(session, api_base: str, sku: str, attributes: dict):
    if not attributes:
        return

    payload: dict[str, object] = {"sku": sku}
    custom_attributes = []
    extension_attributes: dict[str, object] = {}

    for code, value in attributes.items():
        if code == "categories":
            category_links = [
                {"position": idx, "category_id": str(cat_id)}
                for idx, cat_id in enumerate(value or [])
            ]
            extension_attributes["category_links"] = category_links
        else:
            custom_attributes.append({"attribute_code": code, "value": value})

    if custom_attributes:
        payload["custom_attributes"] = custom_attributes
    if extension_attributes:
        payload["extension_attributes"] = extension_attributes

    url = f"{api_base.rstrip('/')}/products/{quote(sku, safe='')}"
    resp = session.put(
        url,
        json={"product": payload},
        headers=build_magento_headers(session=session),
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Magento update failed for {sku}: {resp.status_code} {resp.text[:200]}"
        )


def save_step2_to_magento():
    step2_state = st.session_state.get("step2")
    if not isinstance(step2_state, dict):
        st.info("Нет изменений для сохранения.")
        return

    session = st.session_state.get("mg_session")
    base_url = st.session_state.get("mg_base_url")
    if not session or not base_url:
        st.error("Magento session не инициализирован")
        return

    staged_map = (step2_state.get("staged") or {}).copy()
    if staged_map:
        for set_id, draft in staged_map.items():
            if not isinstance(draft, pd.DataFrame):
                continue
            draft_wide = draft.copy(deep=True)
            step2_state["wide"][set_id] = draft_wide
            step2_state["wide_orig"][set_id] = draft_wide.copy(deep=True)
            existing_long = step2_state["dfs"].get(set_id)
            updated_long = apply_wide_edits_to_long(existing_long, draft_wide)
            if isinstance(updated_long, pd.DataFrame):
                if "value" in updated_long.columns:
                    updated_long["value"] = updated_long["value"].astype(object)
                step2_state["dfs"][set_id] = updated_long
                step2_state["original"][set_id] = updated_long.copy(deep=True)
            else:
                step2_state["dfs"][set_id] = draft_wide.copy(deep=True)
                step2_state["original"][set_id] = draft_wide.copy(deep=True)
        step2_state["staged"].clear()

    wide_map: dict[int, pd.DataFrame] = step2_state.get("wide", {})
    wide_meta_map: dict[int, dict[str, dict]] = step2_state.get("wide_meta", {})
    baseline_map: dict[int, pd.DataFrame] = step2_state.get("wide_synced", {})
    meta_cache = step2_state.get("meta_cache")
    if meta_cache is None:
        meta_cache = step2_state.get("meta")

    if not wide_map:
        st.info("Нет изменений для сохранения.")
        return

    payload_by_sku: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []

    for set_id, wide_df in wide_map.items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue

        attr_cols = [col for col in wide_df.columns if col not in ("sku", "name")]
        if not attr_cols:
            continue

        meta_for_set = wide_meta_map.get(set_id, {})
        baseline_df = baseline_map.get(set_id)
        baseline_idx = (
            baseline_df.set_index("sku")
            if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty
            else None
        )
        current_idx = wide_df.set_index("sku")

        for sku, row in current_idx.iterrows():
            sku_value = str(sku).strip() if isinstance(sku, str) else str(sku)
            if not sku_value:
                continue

            baseline_row: pd.Series | None
            if baseline_idx is not None and sku in baseline_idx.index:
                baseline_row = baseline_idx.loc[sku]
                if isinstance(baseline_row, pd.DataFrame):
                    baseline_row = baseline_row.iloc[-1]
            else:
                baseline_row = None

            for code in attr_cols:
                new_raw = row.get(code)
                old_raw = (
                    baseline_row.get(code)
                    if isinstance(baseline_row, pd.Series) and code in baseline_row.index
                    else None
                )

                if _values_equal(new_raw, old_raw):
                    continue

                meta = meta_for_set.get(code) or {}
                attr_code = meta.get("attribute_code") or code

                try:
                    normalized = normalize_for_magento(attr_code, new_raw, meta_cache)
                except Exception as exc:  # pragma: no cover - safety guard
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": repr(new_raw),
                            "hint_examples": f"normalize error: {exc}",
                        }
                    )
                    continue

                if normalized is None:
                    if _is_blank_value(new_raw):
                        continue
                    meta_info: dict[str, object] = {}
                    if isinstance(meta_cache, AttributeMetaCache):
                        meta_info = meta_cache.get(attr_code) or meta
                    elif isinstance(meta_for_set, dict):
                        meta_info = meta or {}
                    examples = list(meta_info.get("valid_examples") or [])
                    if not examples:
                        options = _meta_options(meta_info)
                        for label in options:
                            if label and label not in examples:
                                examples.append(label)
                            if len(examples) >= 5:
                                break
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": str(new_raw),
                            "hint_examples": ", ".join(examples[:5]),
                        }
                    )
                    continue

                payload_by_sku.setdefault(sku_value, {})[attr_code] = normalized

    if not payload_by_sku and not errors:
        st.info("Нет изменений для сохранения.")
        return

    api_base = st.session_state.get("ai_api_base")
    if not api_base:
        st.warning("Magento API не инициализирован.")
        return

    ok_skus: set[str] = set()
    for sku, attrs in payload_by_sku.items():
        try:
            apply_product_update(session, api_base, sku, attrs)
            ok_skus.add(sku)
        except Exception as exc:  # pragma: no cover - network interaction
            errors.append(
                {
                    "sku": sku,
                    "attribute": "*batch*",
                    "raw": repr(attrs),
                    "hint_examples": str(exc),
                }
            )

    if ok_skus:
        st.success("Updated SKUs:")
        st.markdown("\n".join(f"- `{sku}`" for sku in sorted(ok_skus)))

    if errors:
        import pandas as _pd

        st.warning("Some rows failed; review and fix:")
        st.dataframe(
            _pd.DataFrame(errors, columns=["sku", "attribute", "raw", "hint_examples"]),
            use_container_width=True,
        )

    if ok_skus:
        for set_id, wide_df in wide_map.items():
            if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
                continue
            current_idx = wide_df.set_index("sku")
            baseline_df = baseline_map.get(set_id)
            if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
                baseline_idx = baseline_df.set_index("sku")
            else:
                baseline_idx = pd.DataFrame(columns=current_idx.columns)
                baseline_idx.index.name = "sku"
            updated_idx = baseline_idx.copy()
            changed = False
            for sku in ok_skus:
                if sku not in current_idx.index:
                    continue
                row = current_idx.loc[sku]
                updated_idx.loc[sku, :] = row
                changed = True
            if changed:
                result_df = updated_idx.reset_index()
                for column in result_df.columns:
                    result_df[column] = result_df[column].astype(object)
                step2_state["wide_synced"][set_id] = result_df


def load_items(
    session,
    base_url,
    *,
    attr_set_id: int | None = None,
    products: list[dict] | None = None,
):
    if attr_set_id is None:
        attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)

    rows = []
    prog = st.progress(0.0, text="Loading products…")
    total_hint = 0

    if products is None:
        iterator = iter_products_by_attr_set(session, base_url, attr_set_id)
        for product, total in iterator:
            total_hint = total or total_hint
            status = int(
                _get_custom_attr_value(product, "status", product.get("status", 1))
                or 1
            )
            visibility = int(
                _get_custom_attr_value(
                    product, "visibility", product.get("visibility", 4)
                )
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
                    "status": status,
                }
            )

            if total_hint:
                done = len(rows) / max(total_hint, 1)
                done = min(done, 1.0)
                prog.progress(done, text=f"Loading products… {int(done * 100)}%")
    else:
        total_hint = len(products)
        for idx, product in enumerate(products, start=1):
            product_attr_id = product.get("attribute_set_id")
            if product_attr_id is not None and attr_set_id is not None:
                try:
                    attr_value = int(product_attr_id)
                except (TypeError, ValueError):
                    attr_value = None
                if attr_value is not None and attr_value != int(attr_set_id):
                    continue

            status = int(
                _get_custom_attr_value(product, "status", product.get("status", 1))
                or 1
            )
            visibility = int(
                _get_custom_attr_value(
                    product, "visibility", product.get("visibility", 4)
                )
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
                    "status": status,
                }
            )

            if total_hint:
                done = idx / max(total_hint, 1)
                done = min(done, 1.0)
                prog.progress(done, text=f"Loading products… {int(done * 100)}%")

    prog.progress(1.0, text="Products loaded")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

    if DEBUG and "status" in df.columns:
        st.write("STATUS COUNTS:", df["status"].value_counts(dropna=False))

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

magento_base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
magento_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
magento_session = get_session(auth_token=magento_token)
st.session_state["mg_session"] = magento_session
st.session_state["mg_base_url"] = magento_base_url

if not st.session_state.get("_mg_auth_logged"):
    st.write(
        "Auth header:",
        bool(st.session_state["mg_session"].headers.get("Authorization")),
    )
    st.session_state["_mg_auth_logged"] = True

session = st.session_state.get("mg_session")
base_url = st.session_state.get("mg_base_url")
if not session or not base_url:
    st.error("Magento session не инициализирован")
    st.stop()

st.markdown("### 🤖 AI Content Manager")
st.info("Let’s find items that need attribute set assignment.")

if "step1_editor_mode" not in st.session_state:
    st.session_state["step1_editor_mode"] = "all"

c1, c2 = st.columns(2)
btn_all = c1.button("Load items", type="primary", key="btn_load_all")
btn_50 = c2.button("Load 50 (fast)", key="btn_load_50_fast")

if btn_all:
    requested_run_mode: str | None = "all"
elif btn_50:
    requested_run_mode = "fast50"
else:
    requested_run_mode = None

if requested_run_mode:
    run_mode = requested_run_mode
    other_mode = "fast50" if run_mode == "all" else "all"
    limit = 50 if run_mode == "fast50" else None
    enabled_only = True if run_mode == "fast50" else None
    minimal_fields = run_mode == "fast50"
    st.session_state["step1_editor_mode"] = run_mode
    st.session_state["show_attributes_trigger"] = False

    for prefix in ("default_products", "df_original", "df_edited", "attribute_sets"):
        st.session_state.pop(f"{prefix}_{other_mode}", None)

    cache_key = f"default_products_{run_mode}"
    df_original_key = f"df_original_{run_mode}"
    df_edited_key = f"df_edited_{run_mode}"
    attribute_sets_key = f"attribute_sets_{run_mode}"

    st.info(
        f"Loading default products{f' (limit={limit})' if limit else ''}…"
    )
    try:
        attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)
        data = get_default_products(
            session,
            base_url,
            attr_set_id=attr_set_id,
            qty_min=0,
            limit=limit,
            minimal_fields=minimal_fields,
            enabled_only=enabled_only,
        )
        st.session_state[cache_key] = data
        st.success(
            f"Loaded {len(data or [])} products{f' (limit={limit})' if limit else ''}"
        )
        df_items = load_items(
            session,
            base_url,
            attr_set_id=attr_set_id,
            products=data,
        )
        if df_items.empty:
            st.warning("No items match the Default set filter criteria.")
            st.session_state.pop(df_original_key, None)
            st.session_state.pop(df_edited_key, None)
            st.session_state.pop(attribute_sets_key, None)
            _reset_step2_state()
        else:
            df_ui = df_items.copy()
            df_ui["created_at"] = pd.to_datetime(
                df_ui["created_at"], errors="coerce"
            )
            df_ui = df_ui.sort_values("created_at", ascending=False).reset_index(drop=True)
            st.success(
                f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)"
            )
            st.session_state[df_original_key] = df_ui.copy()
            try:
                attribute_sets = list_attribute_sets(session, base_url)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Failed to fetch attribute sets: {exc}")
                attribute_sets = {}
            st.session_state[attribute_sets_key] = attribute_sets
            df_for_edit = df_ui.copy()
            if "hint" not in df_for_edit.columns:
                df_for_edit["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            df_for_edit = df_for_edit[[col for col in cols_order if col in df_for_edit.columns]]
            st.session_state[df_edited_key] = df_for_edit.copy()
            _reset_step2_state()
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"Error: {exc}")

current_run_mode = st.session_state.get("step1_editor_mode", "all")
df_original_key = f"df_original_{current_run_mode}"
df_edited_key = f"df_edited_{current_run_mode}"
attribute_sets_key = f"attribute_sets_{current_run_mode}"
editor_key = f"editor_step1_{current_run_mode}"

if df_original_key in st.session_state:
    df_ui = st.session_state[df_original_key]
    attribute_sets = st.session_state.get(attribute_sets_key, {})

    if df_ui.empty:
        st.warning("No items match the Default set filter criteria.")
    elif not attribute_sets:
        st.warning("Unable to load attribute sets for editing.")
        st.dataframe(df_ui, use_container_width=True)
    else:
        if df_edited_key not in st.session_state:
            df_init = df_ui.copy()
            if "hint" not in df_init.columns:
                df_init["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            df_init = df_init[[col for col in cols_order if col in df_init.columns]]
            st.session_state[df_edited_key] = df_init.copy()

        df_base = st.session_state[df_edited_key].copy()

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

        step2_active = st.session_state.get("show_attributes_trigger", False)

        if not step2_active:
            step1_column_config = {
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
            }
            st.session_state["step1_column_config_cache"] = step1_column_config
            st.session_state["step1_disabled_cols_cache"] = [
                "sku",
                "name",
                "created_at",
            ]
            col_cfg, disabled_cols = _build_column_config_for_step1_like(step="step1")
            edited_df = st.data_editor(
                df_base,
                column_config=col_cfg,
                disabled=disabled_cols,
                column_order=["sku", "name", "attribute set", "hint", "created_at"],
                use_container_width=True,
                num_rows="fixed",
                key=editor_key,
            )

            if isinstance(edited_df, pd.DataFrame) and st.button("Show Attributes"):
                st.session_state[df_edited_key] = edited_df.copy()
                st.session_state["show_attributes_trigger"] = True
                _reset_step2_state()
        else:
            st.markdown("### Step 2. Items with updated attribute sets")

            if df_edited_key not in st.session_state or df_original_key not in st.session_state:
                st.warning("Нет изменённых товаров.")
                st.session_state["show_attributes_trigger"] = False
            else:
                df_new = st.session_state[df_edited_key].copy()
                df_old = st.session_state[df_original_key].copy()

                required_cols = {"sku", "attribute set"}
                if not (
                    required_cols.issubset(df_new.columns)
                    and required_cols.issubset(df_old.columns)
                ):
                    st.warning("Нет изменённых товаров.")
                    st.session_state["show_attributes_trigger"] = False
                else:
                    df_new_idx = df_new.set_index("sku")
                    df_old_idx = df_old.set_index("sku")
                    common_skus = df_new_idx.index.intersection(df_old_idx.index)
                    if common_skus.empty:
                        st.warning("Нет изменённых товаров.")
                        st.session_state["show_attributes_trigger"] = False
                    else:
                        df_new_common = df_new_idx.loc[common_skus]
                        new_sets = df_new_common["attribute set"].fillna("")
                        old_sets = df_old_idx.loc[common_skus, "attribute set"].fillna("")
                        diff_mask = new_sets != old_sets
                        df_changed = df_new_common.loc[diff_mask].reset_index()

                        if df_changed.empty:
                            st.warning("Нет изменённых товаров.")
                            st.session_state["show_attributes_trigger"] = False
                        else:
                            api_base = st.session_state.get("ai_api_base")
                            attr_sets_map = st.session_state.get("ai_attr_sets_map")
                            setup_failed = False

                            html_error = False
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
                            except HTMLResponseError as exc:
                                st.error(str(exc))
                                html_error = True
                            except Exception as exc:  # pragma: no cover - UI interaction
                                st.warning(
                                    f"Не удалось подготовить подключение к Magento: {exc}"
                                )
                                setup_failed = True

                            if html_error:
                                st.session_state["show_attributes_trigger"] = False
                                st.stop()

                            if setup_failed or not api_base or not attr_sets_map:
                                st.session_state["show_attributes_trigger"] = False
                            else:
                                step2_state = _ensure_step2_state()
                                step2_state.setdefault("staged", {})

                                # Унифицированный кэш метаданных
                                meta_cache: AttributeMetaCache | None = step2_state.get(
                                    "meta_cache"
                                )
                                if not isinstance(meta_cache, AttributeMetaCache):
                                    # Первый запуск — создаём и сохраняем
                                    meta_cache = AttributeMetaCache(session, api_base)
                                    step2_state["meta_cache"] = meta_cache

                                categories_meta = step2_state.get("categories_meta")
                                if not isinstance(categories_meta, dict) or not categories_meta.get(
                                    "options"
                                ):
                                    try:
                                        with st.spinner("📂 Загрузка категорий…"):
                                            categories_meta = ensure_categories_meta(
                                                meta_cache,
                                                session,
                                                api_base,
                                                store_id=0,
                                            )
                                        if not isinstance(categories_meta, dict):
                                            categories_meta = (
                                                meta_cache.get("categories")
                                                if isinstance(
                                                    meta_cache, AttributeMetaCache
                                                )
                                                else {}
                                            )
                                        st.session_state["step2_categories_failed"] = False
                                    except Exception as exc:  # pragma: no cover - UI interaction
                                        st.warning(
                                            f"Не удалось загрузить категории: {exc}"
                                        )
                                        categories_meta = (
                                            meta_cache.get("categories")
                                            if isinstance(meta_cache, AttributeMetaCache)
                                            else {}
                                        )
                                        st.session_state["step2_categories_failed"] = True
                                    step2_state["categories_meta"] = categories_meta
                                else:
                                    st.session_state["step2_categories_failed"] = False

                                if not step2_state["dfs"]:
                                    (
                                        dfs_by_set,
                                        column_configs,
                                        disabled_cols_map,
                                        meta_cache,
                                        row_meta_map,
                                        set_names,
                                    ) = build_attributes_df(
                                        df_changed,
                                        session,
                                        api_base,
                                        attribute_sets,
                                        attr_sets_map,
                                        meta_cache,
                                    )

                                    for set_id, df_set in dfs_by_set.items():
                                        stored_df = df_set.copy(deep=True)
                                        step2_state["dfs"][set_id] = stored_df
                                        step2_state["original"][set_id] = stored_df.copy(
                                            deep=True
                                        )
                                        step2_state["column_config"][set_id] = (
                                            column_configs.get(set_id, {})
                                        )
                                        step2_state["disabled"][set_id] = (
                                            disabled_cols_map.get(
                                                set_id,
                                                [
                                                    "sku",
                                                    "name",
                                                    "attribute_code",
                                                ],
                                            )
                                        )
                                        step2_state["row_meta"][set_id] = row_meta_map.get(
                                            set_id, {}
                                        )
                                        step2_state["set_names"][set_id] = set_names.get(
                                            set_id, str(set_id)
                                        )
                                        wide_df = make_wide(stored_df)
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        for column in wide_df.columns:
                                            wide_df[column] = wide_df[column].astype(object)
                                        step2_state["wide"][set_id] = wide_df
                                        step2_state["wide_orig"][set_id] = wide_df.copy(
                                            deep=True
                                        )
                                        step2_state["wide_synced"][set_id] = wide_df.copy(
                                            deep=True
                                        )
                                        attr_cols = [
                                            col
                                            for col in wide_df.columns
                                            if col not in ("sku", "name")
                                        ]
                                        cacheable = isinstance(
                                            meta_cache, AttributeMetaCache
                                        )
                                        if DEBUG:
                                            st.write(
                                                "DEBUG attr_cols:",
                                                sorted(attr_cols),
                                            )
                                        resolved_map: dict[str, str] = {}
                                        if cacheable:
                                            resolved_map = meta_cache.resolve_codes_in_set(
                                                set_id, ["brand", "condition"]
                                            )
                                            effective_codes: list[str] = []
                                            seen_effective: set[str] = set()
                                            for code in attr_cols:
                                                if not isinstance(code, str):
                                                    continue
                                                effective = resolved_map.get(code, code)
                                                if effective and effective not in seen_effective:
                                                    seen_effective.add(effective)
                                                    effective_codes.append(effective)
                                            if effective_codes:
                                                meta_cache.build_and_set_static_for(
                                                    effective_codes, store_id=0
                                                )
                                            dbg_source = getattr(
                                                meta_cache, "_debug_http", {}
                                            )
                                            dbg_snapshot = (
                                                dbg_source.copy()
                                                if isinstance(dbg_source, dict)
                                                else {}
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "SET ATTRS DEBUG:",
                                                    dbg_snapshot.get(f"set_{set_id}"),
                                                    dbg_snapshot.get(f"set_{set_id}_err"),
                                                )
                                                st.write("RESOLVED:", resolved_map)
                                        wide_meta = step2_state.setdefault("wide_meta", {})
                                        meta_map: dict[str, dict] = {}
                                        if cacheable:
                                            for code in attr_cols:
                                                actual_code = resolved_map.get(code, code)
                                                cached = meta_cache.get(actual_code) or {}
                                                if not isinstance(cached, dict):
                                                    cached = {}
                                                meta_map[code] = cached
                                        else:
                                            meta_map = {code: {} for code in attr_cols}
                                        wide_meta[set_id] = meta_map
                                        resolved_storage = step2_state.setdefault(
                                            "resolved_codes", {}
                                        )
                                        resolved_storage[set_id] = resolved_map
                                        _apply_categories_fallback(meta_map)
                                        _ensure_wide_meta_options(
                                            meta_map,
                                            step2_state["wide"].get(set_id),
                                        )
                                        meta_for_debug = wide_meta[set_id]
                                        if DEBUG:
                                            def _meta_shot(code: str) -> dict:
                                                m = meta_for_debug.get(code, {})
                                                if not isinstance(m, dict):
                                                    m = {}
                                                return {
                                                    "frontend_input": m.get("frontend_input"),
                                                    "options_count": len(m.get("options") or []),
                                                    "first_options": (m.get("options") or [])[:5],
                                                    "has_maps": bool(m.get("options_map"))
                                                    and bool(m.get("values_to_labels")),
                                                    "options_source": m.get("options_source"),
                                                }

                                            counts: dict[str, int] = {}
                                            for meta in meta_for_debug.values():
                                                if not isinstance(meta, dict):
                                                    continue
                                                t = str(meta.get("frontend_input") or "text").lower()
                                                counts[t] = counts.get(t, 0) + 1

                                            st.caption(f"DEBUG set_id={set_id}")
                                            st.write("brand:", _meta_shot("brand"))
                                            st.write("condition:", _meta_shot("condition"))
                                            st.write("DEBUG types distribution:", counts)

                                        step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                            meta_for_debug,
                                            sample_df=step2_state["wide"].get(set_id),
                                        )
                                    step2_state["meta_cache"] = meta_cache

                                    for set_id, df0 in step2_state["dfs"].items():
                                        if "value" in df0.columns:
                                            df0["value"] = df0["value"].astype(object)

                                    if step2_state["dfs"] and not step2_state["wide"]:
                                        for set_id, df_existing in step2_state["dfs"].items():
                                            if not isinstance(df_existing, pd.DataFrame):
                                                continue
                                            wide_df = make_wide(df_existing)
                                            if "attribute_set_id" in wide_df.columns:
                                                wide_df["attribute_set_id"] = set_id
                                            else:
                                                wide_df.insert(1, "attribute_set_id", set_id)
                                            for column in wide_df.columns:
                                                wide_df[column] = wide_df[column].astype(object)
                                            step2_state["wide"][set_id] = wide_df
                                            step2_state["wide_orig"][set_id] = wide_df.copy(
                                                deep=True
                                            )
                                            if set_id not in step2_state["wide_synced"]:
                                                step2_state["wide_synced"][
                                                    set_id
                                                ] = wide_df.copy(deep=True)
                                            attr_codes = [
                                                col
                                                for col in wide_df.columns
                                                if col not in ("sku", "name")
                                            ]
                                            cacheable = isinstance(
                                                meta_cache, AttributeMetaCache
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "DEBUG attr_cols:",
                                                    sorted(attr_codes),
                                                )
                                            resolved_map: dict[str, str] = {}
                                            if cacheable:
                                                resolved_map = meta_cache.resolve_codes_in_set(
                                                    set_id, ["brand", "condition"]
                                                )
                                                effective_codes: list[str] = []
                                                seen_effective: set[str] = set()
                                                for code in attr_codes:
                                                    if not isinstance(code, str):
                                                        continue
                                                    effective = resolved_map.get(code, code)
                                                    if effective and effective not in seen_effective:
                                                        seen_effective.add(effective)
                                                        effective_codes.append(effective)
                                                if effective_codes:
                                                    meta_cache.build_and_set_static_for(
                                                        effective_codes, store_id=0
                                                    )
                                                dbg_source = getattr(
                                                    meta_cache, "_debug_http", {}
                                                )
                                                dbg_snapshot = (
                                                    dbg_source.copy()
                                                    if isinstance(dbg_source, dict)
                                                    else {}
                                                )
                                                if DEBUG:
                                                    st.write(
                                                        "SET ATTRS DEBUG:",
                                                        dbg_snapshot.get(
                                                            f"set_{set_id}"
                                                        ),
                                                        dbg_snapshot.get(
                                                            f"set_{set_id}_err"
                                                        ),
                                                    )
                                                    st.write("RESOLVED:", resolved_map)
                                            wide_meta = step2_state.setdefault(
                                                "wide_meta", {}
                                            )
                                            meta_map: dict[str, dict] = {}
                                            if cacheable:
                                                for code in attr_codes:
                                                    actual_code = resolved_map.get(code, code)
                                                    cached = meta_cache.get(actual_code) or {}
                                                    if not isinstance(cached, dict):
                                                        cached = {}
                                                    meta_map[code] = cached
                                            else:
                                                meta_map = {
                                                    code: {} for code in attr_codes
                                                }
                                            wide_meta[set_id] = meta_map
                                            resolved_storage = step2_state.setdefault(
                                                "resolved_codes", {}
                                            )
                                            resolved_storage[set_id] = resolved_map
                                            _apply_categories_fallback(meta_map)
                                            _ensure_wide_meta_options(
                                                meta_map,
                                                step2_state["wide"].get(set_id),
                                            )
                                            meta_for_debug = wide_meta[set_id]
                                            if DEBUG:
                                                def _meta_shot(code: str) -> dict:
                                                    m = meta_for_debug.get(code, {})
                                                    if not isinstance(m, dict):
                                                        m = {}
                                                    return {
                                                        "frontend_input": m.get("frontend_input"),
                                                        "options_count": len(m.get("options") or []),
                                                        "first_options": (m.get("options") or [])[:5],
                                                        "has_maps": bool(m.get("options_map"))
                                                        and bool(m.get("values_to_labels")),
                                                        "options_source": m.get("options_source"),
                                                    }

                                                counts: dict[str, int] = {}
                                                for meta in meta_for_debug.values():
                                                    if not isinstance(meta, dict):
                                                        continue
                                                    t = str(meta.get("frontend_input") or "text").lower()
                                                    counts[t] = counts.get(t, 0) + 1

                                                st.caption(f"DEBUG set_id={set_id}")
                                                st.write("brand:", _meta_shot("brand"))
                                                st.write("condition:", _meta_shot("condition"))
                                                st.write("DEBUG types distribution:", counts)

                                            step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                                meta_for_debug,
                                                sample_df=step2_state["wide"].get(set_id),
                                            )

                                    st.session_state["show_attrs"] = bool(
                                        step2_state["wide"]
                                    )

                                if not step2_state["wide"]:
                                    st.warning("Нет атрибутов для отображения.")
                                elif st.session_state.get("show_attrs"):
                                    wide_meta = step2_state.setdefault("wide_meta", {})
                                    categories_meta = step2_state.get("categories_meta", {})
                                    if not isinstance(categories_meta, dict):
                                        categories_meta = {}
                                    labels_all = [
                                        opt.get("label")
                                        for opt in (categories_meta.get("options") or [])
                                        if isinstance(opt, dict) and opt.get("label")
                                    ]
                                    for set_id in sorted(step2_state["wide"].keys()):
                                        wide_df = step2_state["wide"].get(set_id)
                                        if not isinstance(wide_df, pd.DataFrame):
                                            continue

                                        wide_df = wide_df.copy(deep=True)
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        step2_state["wide"][set_id] = wide_df

                                        meta_map = wide_meta.get(set_id, {})
                                        if not isinstance(meta_map, dict):
                                            meta_map = {}
                                            wide_meta[set_id] = meta_map

                                        _apply_categories_fallback(meta_map)
                                        df_ref = _coerce_for_ui(wide_df, meta_map)
                                        _ensure_wide_meta_options(meta_map, df_ref)

                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "categories" in df_ref.columns
                                        ):
                                            df_ref["categories"] = df_ref["categories"].apply(
                                                lambda v: list(v)
                                                if isinstance(v, (list, tuple, set))
                                                else ([] if v in (None, "") else [str(v).strip()])
                                            )

                                        string_cols = [
                                            col
                                            for col in ("brand", "condition")
                                            if col in df_ref.columns
                                        ]
                                        if string_cols:
                                            df_ref[string_cols] = df_ref[
                                                string_cols
                                            ].astype("string")

                                        if DEBUG:
                                            st.caption(f"DEBUG set_id={set_id}")
                                            st.write(
                                                "META snapshot:",
                                                {
                                                    code: meta_map.get(code)
                                                    for code in (
                                                        "brand",
                                                        "condition",
                                                        "categories",
                                                    )
                                                },
                                            )

                                        column_config = build_wide_colcfg(
                                            meta_map, sample_df=df_ref
                                        )
                                        column_config["attribute_set_id"] = (
                                            st.column_config.NumberColumn(
                                                "Attribute Set ID", disabled=True
                                            )
                                        )
                                        if "categories" in df_ref.columns:
                                            existing_cfg = column_config.get("categories")
                                            label = (
                                                getattr(existing_cfg, "label", None)
                                                or getattr(existing_cfg, "_label", None)
                                                or "categories"
                                            )
                                            column_config["categories"] = (
                                                st.column_config.MultiselectColumn(
                                                    label, options=labels_all
                                                )
                                            )
                                        step2_state["wide_colcfg"][set_id] = column_config

                                        column_order = [
                                            "sku",
                                            "name",
                                            "attribute_set_id",
                                        ]
                                        column_order.extend(
                                            [
                                                col
                                                for col in df_ref.columns
                                                if col not in column_order
                                            ]
                                        )

                                        if DEBUG:
                                            st.write(
                                                "DEBUG column_order:",
                                                column_order,
                                            )

                                        for current_set_id, group in df_ref.groupby(
                                            "attribute_set_id"
                                        ):
                                            set_title = step2_state["set_names"].get(
                                                current_set_id, str(current_set_id)
                                            )
                                            st.subheader(
                                                f"{_format_attr_set_title(set_title)} (ID {current_set_id})"
                                            )

                                            ordered_columns = [
                                                col
                                                for col in column_order
                                                if col in group.columns
                                            ]
                                            disabled_cols = [
                                                col
                                                for col in [
                                                    "sku",
                                                    "name",
                                                    "attribute_set_id",
                                                ]
                                                if col in group.columns
                                            ]
                                            editor_df = st.data_editor(
                                                group.reset_index(drop=True),
                                                key=f"editor_set_{current_set_id}",
                                                column_config={
                                                    key: value
                                                    for key, value in column_config.items()
                                                    if key in group.columns
                                                },
                                                column_order=ordered_columns,
                                                disabled=disabled_cols,
                                                use_container_width=True,
                                                hide_index=True,
                                                num_rows="fixed",
                                            )

                                            if isinstance(editor_df, pd.DataFrame):
                                                for column in editor_df.columns:
                                                    editor_df[column] = editor_df[
                                                        column
                                                    ].astype(object)
                                                base_synced = (
                                                    step2_state.get("wide_synced", {})
                                                    .get(current_set_id)
                                                )
                                                if _df_differs(
                                                    editor_df, base_synced
                                                ):
                                                    step2_state["staged"][
                                                        current_set_id
                                                    ] = editor_df.copy(
                                                        deep=True
                                                    )
                                                else:
                                                    step2_state["staged"].pop(
                                                        current_set_id, None
                                                    )

                                    if st.button(
                                        "Save to Magento",
                                        type="primary",
                                        key="step2_save",
                                    ):
                                        save_step2_to_magento()

                                    if st.button("Reload", key="step2_reload"):
                                        _reset_step2_state()
                                        st.experimental_rerun()
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")
