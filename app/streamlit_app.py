from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import quote

import pandas as pd
import streamlit as st

from connectors.magento.attributes import AttributeMetaCache
from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    HTMLResponseError,
    collect_attributes_table,
    compute_allowed_attrs,
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
    df = sample_df if isinstance(sample_df, pd.DataFrame) else None
    for code, meta in meta_map.items():
        if not isinstance(meta, dict):
            continue
        frontend = str(meta.get("frontend_input") or "").lower()
        if frontend not in {"select", "boolean"}:
            continue
        options = meta.get("options") or []
        if options:
            continue
        v2l = meta.get("values_to_labels") or {}
        meta["options"] = [
            {"label": lbl, "value": key}
            for key, lbl in list(v2l.items())[:200]
            if str(lbl).strip()
        ]
        if meta["options"]:
            continue
        if df is None or code not in df.columns:
            continue
        seen: set[str] = set()
        opts: list[dict[str, str]] = []
        for value in df[code].dropna().astype(str):
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            opts.append({"label": cleaned, "value": cleaned})
            if len(opts) >= 200:
                break
        meta["options"] = opts


def _refresh_wide_meta_from_cache(
    meta_cache: AttributeMetaCache | None,
    meta_map: dict[str, dict],
    attr_codes: list[str],
    wide_df: pd.DataFrame | None,
) -> None:
    if not isinstance(meta_map, dict) or not attr_codes:
        return

    df = wide_df if isinstance(wide_df, pd.DataFrame) else None

    for code in attr_codes:
        cached_meta = (
            meta_cache.get(code)
            if isinstance(meta_cache, AttributeMetaCache)
            else None
        )
        existing = meta_map.get(code)
        if not isinstance(existing, dict):
            existing = {}
        else:
            existing = dict(existing)

        if isinstance(cached_meta, dict):
            merged = dict(existing)
            for key, value in cached_meta.items():
                if key == "options":
                    cleaned_opts: list[dict[str, object]] = []
                    for opt in value or []:
                        if not isinstance(opt, dict):
                            continue
                        label = opt.get("label")
                        if not isinstance(label, str) or not label.strip():
                            label = str(opt.get("value", "")).strip()
                        else:
                            label = label.strip()
                        if not label:
                            continue
                        cleaned_opts.append({"label": label, "value": opt.get("value")})
                    merged["options"] = cleaned_opts
                else:
                    merged[key] = value
            existing = merged

        options_list = existing.get("options")
        if not isinstance(options_list, list):
            existing["options"] = []
        if not existing.get("options") and df is not None and code in df.columns:
            seen: set[str] = set()
            fallback: list[dict[str, str]] = []
            series = df[code].dropna()
            for raw in series.astype(str):
                cleaned = raw.strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                fallback.append({"label": cleaned, "value": cleaned})
                if len(fallback) >= 200:
                    break
            existing["options"] = fallback

        meta_map[code] = existing


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
            out[code] = out[code].apply(
                lambda v: v
                if isinstance(v, (list, tuple, set))
                else ([] if v in (None, "") else [str(v).strip()])
            )
        else:
            out[code] = out[code].astype(object)
    return out


def _apply_categories_fallback(meta_map: dict[str, dict]) -> None:
    if not isinstance(meta_map, dict):
        return
    if not st.session_state.get("step2_categories_failed"):
        return
    meta = meta_map.get("categories")
    if isinstance(meta, dict):
        meta["frontend_input"] = "text"
        meta.pop("options", None)


def build_wide_colcfg(
    wide_meta: dict[str, dict], sample_df: pd.DataFrame | None = None
):
    cfg = {
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "name": st.column_config.TextColumn("Name", disabled=True),
    }

    df_sample = sample_df if isinstance(sample_df, pd.DataFrame) else None

    for code, original_meta in list(wide_meta.items()):
        meta = original_meta or {}
        series = df_sample[code] if (df_sample is not None and code in df_sample.columns) else None

        t = (meta.get("frontend_input") or meta.get("backend_type") or "text").lower()
        opts = [
            str(opt.get("label"))
            for opt in (meta.get("options") or [])
            if str(opt.get("label", "")).strip()
        ]
        if not opts:
            v2l = meta.get("values_to_labels") or {}
            opts = [
                lbl
                for lbl in {str(v).strip() for v in v2l.values()}
                if lbl
            ]
        if (
            not opts
            and isinstance(sample_df, pd.DataFrame)
            and (code in sample_df.columns)
        ):
            seen = set()
            for v in sample_df[code].dropna().astype(str):
                s = v.strip()
                if s and s not in seen:
                    seen.add(s)
                    opts.append(s)

        select_like = (t == "select") or (
            opts and t in {"", "text", "varchar", "static"}
        )

        if t == "multiselect":
            is_list_series = False
            if isinstance(series, pd.Series):
                is_list_series = series.apply(
                    lambda v: isinstance(v, (list, tuple, set)) or v in (None, "")
                ).all()
            if is_list_series:
                cfg[code] = st.column_config.MultiselectColumn(
                    _attr_label(meta, code), options=opts
                )
            else:
                cfg[code] = st.column_config.TextColumn(_attr_label(meta, code))
        elif t == "boolean":
            cfg[code] = st.column_config.CheckboxColumn(
                _attr_label(meta, code)
            )
        elif select_like and len(opts) >= 1:
            cfg[code] = st.column_config.SelectboxColumn(
                _attr_label(meta, code), options=opts
            )
        elif t in {"int", "integer", "decimal", "price", "float"}:
            cfg[code] = st.column_config.NumberColumn(_attr_label(meta, code))
        else:
            cfg[code] = st.column_config.TextColumn(_attr_label(meta, code))

    return cfg


def build_attributes_df(
    df_changed: pd.DataFrame,
    session,
    api_base: str,
    attribute_sets: dict,
    attr_sets_map: dict,
    category_options: list[dict],
    meta_cache: AttributeMetaCache,
):
    core_codes = ["brand", "condition"]
    rows_by_set: dict[int, list[dict]] = defaultdict(list)
    row_meta_by_set: dict[int, dict[str, dict]] = defaultdict(dict)
    set_names: dict[int, str] = {}

    id_to_label = {str(item.get("id")): item.get("label") for item in category_options}
    label_to_id = {item.get("label"): item.get("id") for item in category_options}

    if label_to_id:
        categories_aliases: dict[object, object] = {}
        categories_value_to_label: dict[object, str] = {}
        valid_examples: list[str] = []
        for label, value in label_to_id.items():
            if not label:
                continue
            clean_label = str(label).strip()
            if not clean_label:
                continue
            if clean_label not in valid_examples and len(valid_examples) < 5:
                valid_examples.append(clean_label)
            categories_aliases[clean_label] = value
            categories_aliases[clean_label.lower()] = value
            categories_value_to_label[clean_label] = clean_label
            categories_value_to_label[clean_label.lower()] = clean_label
            str_value = str(value).strip()
            if str_value:
                categories_aliases[str_value] = value
                categories_value_to_label[str_value] = clean_label
                try:
                    int_value = int(str_value)
                except (TypeError, ValueError):
                    int_value = None
                if int_value is not None:
                    categories_aliases[int_value] = value
                    categories_aliases[str(int_value)] = value
                    categories_value_to_label[int_value] = clean_label

        category_meta = {
            "frontend_input": "multiselect",
            "backend_type": "text",
            "options_map": categories_aliases,
            "values_to_labels": categories_value_to_label,
            "valid_examples": valid_examples,
            "options": [
                {
                    "label": item.get("label"),
                    "value": item.get("id"),
                }
                for item in category_options
            ],
        }
        meta_cache.set_static("categories", category_meta)

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
        category_labels = _categories_ids_to_labels(categories, id_to_label)
        row_id = f"{sku_value}|categories"
        row_meta_by_set[attr_set_id_int][row_id] = {
            "attribute_code": "categories",
            "frontend_input": "multiselect",
            "backend_type": "text",
            "options": [
                item.get("label")
                for item in category_options
                if isinstance(item.get("label"), str) and item.get("label").strip()
            ],
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

    if step2_state.get("staged"):
        st.warning("Сначала нажмите \"💾 Сохранить изменения\".")
        return

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

    session = st.session_state.get("session")
    if session is None:
        st.warning("Magento session не инициализирован.")
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
            st.session_state["show_attributes_trigger"] = False
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
            _reset_step2_state()
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
                key="editor_key_main",
            )

            if isinstance(edited_df, pd.DataFrame) and st.button("Show Attributes"):
                st.session_state["df_edited"] = edited_df.copy()
                st.session_state["show_attributes_trigger"] = True
                _reset_step2_state()
        else:
            st.markdown("### Step 2. Items with updated attribute sets")

            if "df_edited" not in st.session_state or "df_original" not in st.session_state:
                st.info("Нет изменённых товаров.")
                st.session_state["show_attributes_trigger"] = False
                _reset_step2_state()
            else:
                df_new = st.session_state["df_edited"].copy()
                df_old = st.session_state["df_original"].copy()

                required_cols = {"sku", "attribute set"}
                if not (
                    required_cols.issubset(df_new.columns)
                    and required_cols.issubset(df_old.columns)
                ):
                    st.info("Нет изменённых товаров.")
                    st.session_state["show_attributes_trigger"] = False
                    _reset_step2_state()
                else:
                    df_new_idx = df_new.set_index("sku")
                    df_old_idx = df_old.set_index("sku")
                    common_skus = df_new_idx.index.intersection(df_old_idx.index)
                    if common_skus.empty:
                        st.info("Нет изменённых товаров.")
                        st.session_state["show_attributes_trigger"] = False
                        _reset_step2_state()
                    else:
                        df_new_common = df_new_idx.loc[common_skus]
                        new_sets = df_new_common["attribute set"].fillna("")
                        old_sets = df_old_idx.loc[common_skus, "attribute set"].fillna("")
                        diff_mask = new_sets != old_sets
                        df_changed = df_new_common.loc[diff_mask].reset_index()

                        if df_changed.empty:
                            st.info("Нет изменённых товаров.")
                            st.session_state["show_attributes_trigger"] = False
                            _reset_step2_state()
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
                                _reset_step2_state()
                                st.stop()

                            if setup_failed or not api_base or not attr_sets_map:
                                st.session_state["show_attributes_trigger"] = False
                                _reset_step2_state()
                            else:
                                categories_options = st.session_state.get(
                                    "step2_category_options"
                                )
                                categories_failed = st.session_state.get(
                                    "step2_categories_failed", False
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
                                        categories_failed = False
                                    except Exception as exc:  # pragma: no cover - UI interaction
                                        st.warning(
                                            f"Не удалось загрузить категории: {exc}"
                                        )
                                        categories_options = []
                                        categories_failed = True
                                    st.session_state[
                                        "step2_category_options"
                                    ] = categories_options
                                    st.session_state[
                                        "step2_categories_failed"
                                    ] = categories_failed
                                else:
                                    st.session_state[
                                        "step2_categories_failed"
                                    ] = categories_failed

                                step2_state = _ensure_step2_state()
                                step2_state.setdefault("staged", {})

                                if not step2_state["dfs"]:
                                    meta_cache_obj = AttributeMetaCache(
                                        session, api_base
                                    )
                                    (
                                        dfs_by_set,
                                        column_configs,
                                        disabled_cols_map,
                                        meta_cache_obj,
                                        row_meta_map,
                                        set_names,
                                    ) = build_attributes_df(
                                        df_changed,
                                        session,
                                        api_base,
                                        attribute_sets,
                                        attr_sets_map,
                                        categories_options,
                                        meta_cache_obj,
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
                                            meta_cache_obj, AttributeMetaCache
                                        )
                                        if cacheable:
                                            try:
                                                meta_cache_obj.load(attr_cols)
                                            except RuntimeError:
                                                st.error(
                                                    "❌ Magento returned non-JSON for attributes/options. "
                                                    "Check REST base (/rest/V1) or ACL/WAF."
                                                )
                                                st.stop()
                                        wide_meta = step2_state.setdefault("wide_meta", {})
                                        if cacheable:
                                            wide_meta[set_id] = {
                                                c: meta_cache_obj.get(c) for c in attr_cols
                                            }
                                        meta_map = wide_meta.setdefault(set_id, {})
                                        if not isinstance(meta_map, dict):
                                            meta_map = {}
                                            wide_meta[set_id] = meta_map
                                        for code in attr_cols:
                                            if code not in meta_map:
                                                meta_map[code] = {}
                                        try:
                                            _refresh_wide_meta_from_cache(
                                                meta_cache_obj if cacheable else None,
                                                meta_map,
                                                attr_cols,
                                                wide_df,
                                            )
                                        except RuntimeError:
                                            st.error(
                                                "❌ Magento returned non-JSON for attributes/options. "
                                                "Check REST base (/rest/V1) or ACL/WAF."
                                            )
                                            st.stop()
                                        _apply_categories_fallback(meta_map)
                                        for code, meta in meta_map.items():
                                            if not isinstance(meta, dict):
                                                continue
                                            frontend = (
                                                meta.get("frontend_input") or "text"
                                            ).lower()
                                            if frontend not in {"select", "boolean"}:
                                                continue
                                            if meta.get("options"):
                                                continue
                                            v2l = meta.get("values_to_labels") or {}
                                            options = [
                                                {"label": lbl, "value": key}
                                                for key, lbl in v2l.items()
                                                if str(lbl).strip()
                                            ]
                                            if not options and code in wide_df.columns:
                                                seen: set[str] = set()
                                                inferred: list[dict[str, str]] = []
                                                for value in (
                                                    wide_df[code]
                                                    .dropna()
                                                    .astype(str)
                                                ):
                                                    cleaned = value.strip()
                                                    if not cleaned or cleaned in seen:
                                                        continue
                                                    seen.add(cleaned)
                                                    inferred.append(
                                                        {"label": cleaned, "value": cleaned}
                                                    )
                                                    if len(inferred) >= 200:
                                                        break
                                                options = inferred
                                            meta["options"] = options[:200]
                                        _ensure_wide_meta_options(
                                            meta_map,
                                            step2_state["wide"].get(set_id),
                                        )
                                        if set_id not in step2_state["wide_colcfg"]:
                                            meta_map = step2_state["wide_meta"].get(
                                                set_id, {}
                                            )
                                            _apply_categories_fallback(meta_map)
                                            step2_state["wide_colcfg"][set_id] = (
                                                build_wide_colcfg(
                                                    meta_map,
                                                    sample_df=step2_state["wide"].get(
                                                        set_id
                                                    ),
                                                )
                                            )
                                    step2_state["meta_cache"] = meta_cache_obj

                                    for set_id, df0 in step2_state["dfs"].items():
                                        if "value" in df0.columns:
                                            df0["value"] = df0["value"].astype(object)

                                    if step2_state["dfs"] and not step2_state["wide"]:
                                        meta_obj = step2_state.get("meta_cache")
                                        for set_id, df_existing in step2_state["dfs"].items():
                                            if not isinstance(df_existing, pd.DataFrame):
                                                continue
                                            wide_df = make_wide(df_existing)
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
                                                meta_obj, AttributeMetaCache
                                            )
                                            if cacheable:
                                                try:
                                                    meta_obj.load(attr_codes)
                                                except RuntimeError:
                                                    st.error(
                                                        "❌ Magento returned non-JSON for attributes/options. "
                                                        "Check REST base (/rest/V1) or ACL/WAF."
                                                    )
                                                    st.stop()
                                            wide_meta = step2_state.setdefault(
                                                "wide_meta", {}
                                            )
                                            if cacheable:
                                                wide_meta[set_id] = {
                                                    c: meta_obj.get(c) for c in attr_codes
                                                }
                                            meta_map = wide_meta.setdefault(set_id, {})
                                            if not isinstance(meta_map, dict):
                                                meta_map = {}
                                                wide_meta[set_id] = meta_map
                                            for code in attr_codes:
                                                if code not in meta_map:
                                                    meta_map[code] = {}
                                            try:
                                                _refresh_wide_meta_from_cache(
                                                    meta_obj if cacheable else None,
                                                    meta_map,
                                                    attr_codes,
                                                    wide_df,
                                                )
                                            except RuntimeError:
                                                st.error(
                                                    "❌ Magento returned non-JSON for attributes/options. "
                                                    "Check REST base (/rest/V1) or ACL/WAF."
                                                )
                                                st.stop()
                                            _apply_categories_fallback(meta_map)
                                            for code, meta in meta_map.items():
                                                if not isinstance(meta, dict):
                                                    continue
                                                frontend = (
                                                    meta.get("frontend_input") or "text"
                                                ).lower()
                                                if frontend not in {"select", "boolean"}:
                                                    continue
                                                if meta.get("options"):
                                                    continue
                                                v2l = meta.get("values_to_labels") or {}
                                                options = [
                                                    {"label": lbl, "value": key}
                                                    for key, lbl in v2l.items()
                                                    if str(lbl).strip()
                                                ]
                                                if not options and code in wide_df.columns:
                                                    seen: set[str] = set()
                                                    inferred: list[dict[str, str]] = []
                                                    for value in (
                                                        wide_df[code]
                                                        .dropna()
                                                        .astype(str)
                                                    ):
                                                        cleaned = value.strip()
                                                        if not cleaned or cleaned in seen:
                                                            continue
                                                        seen.add(cleaned)
                                                        inferred.append(
                                                            {"label": cleaned, "value": cleaned}
                                                        )
                                                        if len(inferred) >= 200:
                                                            break
                                                    options = inferred
                                                meta["options"] = options[:200]
                                            _ensure_wide_meta_options(
                                                meta_map,
                                                step2_state["wide"].get(set_id),
                                            )
                                            if set_id not in step2_state["wide_colcfg"]:
                                                step2_state["wide_colcfg"][set_id] = (
                                                    build_wide_colcfg(
                                                        meta_map,
                                                        sample_df=step2_state["wide"].get(
                                                            set_id
                                                        ),
                                                    )
                                                )

                                    st.session_state["show_attrs"] = bool(
                                        step2_state["wide"]
                                    )

                                if not step2_state["wide"]:
                                    st.info("Нет атрибутов для отображения.")
                                elif st.session_state.get("show_attrs"):
                                    for set_id in sorted(step2_state["wide"].keys()):
                                        wide_df = step2_state["wide"].get(set_id)
                                        if not isinstance(wide_df, pd.DataFrame):
                                            continue

                                        meta_map = step2_state["wide_meta"].get(
                                            set_id, {}
                                        )
                                        if not isinstance(meta_map, dict):
                                            meta_map = {}
                                        _apply_categories_fallback(meta_map)
                                        df_ref = _coerce_for_ui(wide_df, meta_map)
                                        _ensure_wide_meta_options(meta_map, df_ref)
                                        def _meta_shot(code: str) -> dict:
                                            m = meta_map.get(code, {}) if isinstance(meta_map, dict) else {}
                                            if not isinstance(m, dict):
                                                m = {}
                                            return {
                                                "frontend_input": m.get("frontend_input"),
                                                "options_count": len(m.get("options") or []),
                                                "first_options": (m.get("options") or [])[:3],
                                            }

                                        st.caption(f"DEBUG set_id={set_id}")
                                        st.write("DEBUG brand:", _meta_shot("brand"))
                                        st.write("DEBUG condition:", _meta_shot("condition"))

                                        missing = [
                                            c
                                            for c, m in (meta_map or {}).items()
                                            if isinstance(m, dict)
                                            and (
                                                str(m.get("frontend_input") or "").lower()
                                                in {"select", "boolean"}
                                            )
                                            and not (m.get("options") or [])
                                        ]
                                        if missing:
                                            st.write(
                                                "DEBUG missing options for:",
                                                missing[:10],
                                            )
                                        colcfg = build_wide_colcfg(
                                            meta_map, sample_df=df_ref
                                        )
                                        step2_state["wide_colcfg"][set_id] = colcfg

                                        set_title = step2_state["set_names"].get(
                                            set_id, str(set_id)
                                        )
                                        st.markdown(
                                            f"#### {_format_attr_set_title(set_title)} (ID {set_id})"
                                        )
                                        column_order = [
                                            "sku",
                                            "name",
                                        ] + [
                                            col
                                            for col in df_ref.columns
                                            if col not in ("sku", "name")
                                        ]
                                        meta_cache = step2_state.get("meta_cache")
                                        wide_meta_map = step2_state.get("wide_meta", {})
                                        st.caption(
                                            "brand alias: "
                                            f"{getattr(meta_cache, '_aliases', {}).get('brand')}, "
                                            "brand options: "
                                            f"{len((wide_meta_map.get(set_id, {}) or {}).get('brand', {}).get('options', []))}"
                                        )
                                        edited_df = st.data_editor(
                                            df_ref,
                                            key=f"step2_wide::{set_id}",
                                            column_config=colcfg,
                                            disabled=["sku", "name"],
                                            use_container_width=True,
                                            hide_index=True,
                                            num_rows="fixed",
                                            column_order=column_order,
                                        )

                                        if isinstance(edited_df, pd.DataFrame):
                                            for column in edited_df.columns:
                                                edited_df[column] = edited_df[
                                                    column
                                                ].astype(object)
                                            step2_state["staged"][set_id] = edited_df.copy(
                                                deep=True
                                            )

                                    if st.button(
                                        "💾 Сохранить изменения",
                                        key="step2_commit",
                                    ):
                                        for set_id, draft in (
                                            step2_state.get("staged") or {}
                                        ).items():
                                            if not isinstance(draft, pd.DataFrame):
                                                continue
                                            draft_wide = draft.copy(deep=True)
                                            step2_state["wide"][set_id] = draft_wide
                                            step2_state["wide_orig"][set_id] = draft_wide.copy(
                                                deep=True
                                            )
                                            existing_long = step2_state["dfs"].get(set_id)
                                            updated_long = apply_wide_edits_to_long(
                                                existing_long, draft_wide
                                            )
                                            if isinstance(updated_long, pd.DataFrame):
                                                if "value" in updated_long.columns:
                                                    updated_long["value"] = updated_long[
                                                        "value"
                                                    ].astype(object)
                                                step2_state["dfs"][set_id] = updated_long
                                                step2_state["original"][
                                                    set_id
                                                ] = updated_long.copy(deep=True)
                                            else:
                                                step2_state["dfs"][set_id] = (
                                                    draft_wide.copy(deep=True)
                                                )
                                                step2_state["original"][set_id] = (
                                                    draft_wide.copy(deep=True)
                                                )
                                        step2_state["staged"].clear()
                                        st.success("Изменения сохранены.")

                                    if st.button(
                                        "Save to Magento",
                                        type="primary",
                                        key="step2_save",
                                    ):
                                        save_step2_to_magento()
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")
