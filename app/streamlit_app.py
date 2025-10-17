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


def _normalize_diff_value(value):
    if isinstance(value, (list, tuple, set)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return tuple(normalized)
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return value


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


def _prepare_value_for_save(value):
    if isinstance(value, tuple):
        return [item for item in value]
    return value


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
    resp = session.put(url, json={"product": payload}, timeout=30)
    if not resp.ok:
        raise RuntimeError(
            f"Magento update failed for {sku}: {resp.status_code} {resp.text[:200]}"
        )


def save_step2_to_magento():
    step2_state = st.session_state.get("step2")
    if not isinstance(step2_state, dict):
        st.info("Нет изменений для сохранения.")
        return

    dfs: dict[int, pd.DataFrame] = step2_state.get("dfs", {})
    originals: dict[int, pd.DataFrame] = step2_state.get("original", {})
    row_meta: dict[int, dict[str, dict]] = step2_state.get("row_meta", {})
    meta_cache = step2_state.get("meta_cache")
    if meta_cache is None:
        meta_cache = step2_state.get("meta")

    if not dfs:
        st.info("Нет изменений для сохранения.")
        return

    payload_by_sku: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []

    for set_id, current_df in dfs.items():
        if not isinstance(current_df, pd.DataFrame) or current_df.empty:
            continue
        original_df = originals.get(set_id)
        if not isinstance(original_df, pd.DataFrame) or original_df.empty:
            continue

        common_idx = current_df.index.intersection(original_df.index)
        if common_idx.empty:
            continue

        meta_for_set = row_meta.get(set_id, {})

        for row_id in common_idx:
            current_value = current_df.at[row_id, "value"]
            original_value = original_df.at[row_id, "value"]
            if _normalize_diff_value(current_value) == _normalize_diff_value(original_value):
                continue

            sku = str(current_df.at[row_id, "sku"])
            code = str(current_df.at[row_id, "attribute_code"])
            info = meta_for_set.get(row_id, {})
            prepared_value = _prepare_value_for_save(current_value)
            try:
                normalized = normalize_for_magento(code, prepared_value, meta_cache)
            except Exception as exc:  # pragma: no cover - safety guard
                errors.append(
                    {
                        "sku": sku,
                        "attribute": code,
                        "raw": repr(current_value),
                        "hint_examples": f"normalize error: {exc}",
                    }
                )
                continue

            if normalized is None:
                if _is_blank_value(current_value):
                    continue
                meta_info = (
                    meta_cache.get(code)
                    if isinstance(meta_cache, AttributeMetaCache)
                    else {}
                )
                examples = info.get("valid_examples") or meta_info.get("valid_examples") or []
                if not examples:
                    options = info.get("options") or _meta_options(meta_info)
                    for label in options:
                        if label and label not in examples:
                            examples.append(label)
                        if len(examples) >= 5:
                            break
                errors.append(
                    {
                        "sku": sku,
                        "attribute": code,
                        "raw": str(current_value),
                        "hint_examples": ", ".join(examples[:5]),
                    }
                )
                continue

            payload_by_sku.setdefault(sku, {})[code] = normalized

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
        for set_id, df_current in dfs.items():
            if isinstance(df_current, pd.DataFrame):
                originals[set_id] = df_current.copy(deep=True)


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
                                _reset_step2_state()
                            else:
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

                                step2_state = _ensure_step2_state()

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
                                        step2_state["dfs"][set_id] = df_set.copy(deep=True)
                                        step2_state["original"][set_id] = df_set.copy(
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
                                    step2_state["meta_cache"] = meta_cache_obj

                                    st.session_state["show_attrs"] = bool(
                                        step2_state["dfs"]
                                    )

                                if not step2_state["dfs"]:
                                    st.info("Нет атрибутов для отображения.")
                                elif st.session_state.get("show_attrs"):
                                    for set_id, df_set in sorted(
                                        step2_state["dfs"].items()
                                    ):
                                        set_title = step2_state["set_names"].get(
                                            set_id, str(set_id)
                                        )
                                        st.markdown(
                                            f"#### {_format_attr_set_title(set_title)} (ID {set_id})"
                                        )
                                        edited_df = st.data_editor(
                                            df_set,
                                            key=f"step2_editor::{set_id}",
                                            column_config=step2_state["column_config"][
                                                set_id
                                            ],
                                            disabled=step2_state["disabled"][set_id],
                                            column_order=[
                                                "sku",
                                                "name",
                                                "attribute_code",
                                                "value",
                                            ],
                                            use_container_width=True,
                                            num_rows="fixed",
                                            hide_index=True,
                                        )

                                        if isinstance(edited_df, pd.DataFrame):
                                            step2_state["dfs"][set_id] = edited_df.copy(
                                                deep=True
                                            )

                                    if st.button(
                                        "Save to Magento",
                                        type="primary",
                                        key="step2_save",
                                    ):
                                        save_step2_to_magento()
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")
