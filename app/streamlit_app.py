from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
import os
import traceback
from string import Template

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import quote
import re

import requests
import openai

import pandas as pd
import streamlit as st
import random

from utils.json_parse import extract_json_object, short_preview_of

st.session_state.setdefault("allow_ai_overwrite_text", False)
st.session_state.setdefault("ai_force_text_override", False)
st.session_state.setdefault("ai_rerun_requested", False)  # оставляем на случай логики флага
st.session_state.setdefault("step3_active", False)
st.session_state.setdefault("descriptions", {})
st.session_state.setdefault("step3", {})
st.session_state.setdefault("step3_generation_pending", False)
st.session_state.setdefault("step3_output_rows", [])
st.session_state.setdefault("step2_output_rows", [])

header_ph = st.session_state.setdefault("_header_ph", st.empty())
if not st.session_state.get("_header_rendered", False):
    with header_ph:
        st.title("🤖 Peter v.1.3 (AI Content Manager)")
    st.session_state["_header_rendered"] = True

st.set_page_config(
    page_title="MaxGuitar AI - v3.9.5",
    page_icon="🤖",
    layout="wide",
)

# --- DEBUG UI PANEL ---
if "_trace_events" not in st.session_state:
    st.session_state["_trace_events"] = []


def trace(event: dict):
    # дублируем в session_state и в консоль
    st.session_state["_trace_events"].append(event)
    try:
        print("[TRACE]", event)
    except Exception:
        pass

# Рисуем Debug trace ТОЛЬКО когда мы не в шаге 2
if not st.session_state.get("show_attributes_trigger", False):
    with st.expander("🧪 Debug trace", expanded=False):
        st.json(st.session_state["_trace_events"][-500:])  # последние 500 событий

from connectors.magento.attributes import AttributeMetaCache
from connectors.magento.categories import ensure_categories_meta
from connectors.magento.client import get_default_products, get_api_base
from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    ELECTRIC_SET_IDS,
    DEFAULT_OPENAI_MODEL,
    HTMLResponseError,
    OPENAI_MODEL_ALIASES,
    collect_attributes_table,
    compute_allowed_attrs,
    derive_styles_from_texts,
    enrich_ai_suggestions,
    get_attribute_sets_map,
    get_ai_conversation,
    get_product_by_sku,
    infer_missing,
    normalize_category_label,
    probe_api_base,
    resolve_model_name,
    should_force_override,
)
from services.apply import build_magento_payload, map_value_for_magento
from services.llm_extract import brand_from_title, is_numeric_spec
from services.inventory import (
    get_attr_set_id,
    get_backorders_parallel,
    detect_default_source_code,
    get_source_items,
    iter_products_by_attr_set,
    list_attribute_sets,
)
from services.normalizers import (
    TEXT_INPUT_TYPES,
    _coerce_ai_value,
    normalize_for_magento,
    normalize_units,
)
from utils.http import build_magento_headers, get_session


logger = logging.getLogger(__name__)


def _force_mark(sku: str, code: str):
    fs = st.session_state.setdefault("force_send", {})
    fs.setdefault(str(sku), set()).add(str(code))


_DEF_ATTR_SET_NAME = "Default"
STEP3_AUTOSAVE_PATH = Path(".cache/step3_autosave.json")
_ALLOWED_TYPES = {"simple", "configurable"}


GENERATE_DESCRIPTION_COLUMN = "generate_description"


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


BASE_FIRST = [
    "sku",
    "name",
    "price",
    "condition",
    "country_of_manufacture",
    "brand",
]


ORDER_PRESETS = {
    # ELECTRIC GUITAR
    "electric guitar": [
        "series",
        "model",
        "guitarstylemultiplechoice",  # rendered as "Guitar style"
        "body_material",
        "top_material",
        "finish",
        "bridge_type",
        "bridge",
        "pickup_config",
        "bridge_pickup",
        "middle_pickup",
        "neck_pickup",
        "controls",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "semi_hollow_body",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],

    # ACOUSTIC GUITAR
    "acoustic guitar": [
        "series",
        "acoustic_guitar_style",
        "acoustic_body_shape",
        "body_material",
        "top_material",
        "finish",
        "bridge",
        "controls",
        "acoustic_pickup",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "acoustic_cutaway",
        "electro_acoustic",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],

   # BASS GUITAR
"bass guitar": [
    "series",
    "model",
    "guitarstylemultiplechoice",  # rendered as "Guitar style"
    "body_material",
    "top_material",
    "finish",
    "bridge_type",
    "bridge",
    "pickup_config",
    "bridge_pickup",
    "middle_pickup",
    "neck_pickup",
    "controls",
    "scale_mensur",
    "neck_profile",
    "neck_material",
    "neck_radius",
    "neck_nutwidth",
    "fretboard_material",
    "tuning_machines",
    "no_strings",
    "vintage",
    "cases_covers",
    "acoustic_bass",
    "amount_of_frets",
    "orientation",
    "kids_size",
    "categories",
    "short_description",
],

    # AMPS
    "amps": [
        "amp_style",
        "series",
        "type",
        "effect_type",
        "controls",
        "built_in_fx",
        "speaker_configuration",
        "battery",
        "power",
        "power_polarity",
        "vintage",
        "categories",
        "short_description",
    ],

    # EFFECTS
    "effects": [
        "effect_type",
        "controls",
        "battery",
        "power",
        "power_polarity",
        "vintage",
        "categories",
        "short_description",
    ],

    # ACCESSORIES
    "accessories": [
        "accessory_type",
        "strings",
        "cables",
        "parts",
        "merchandise",
        "cases_covers",
        "categories",
        "short_description",
    ],
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _norm_code(s: str) -> str:
    return re.sub(r"[\s_]+", "", _norm(s))


def build_column_order(set_name: str, df_cols: list[str]) -> list[str]:
    cols = list(df_cols)
    norm_to_actual: dict[str, str] = {}
    for col in cols:
        if not isinstance(col, str):
            continue
        key = _norm_code(col)
        if key and key not in norm_to_actual:
            norm_to_actual[key] = col

    def _resolve(codes: Iterable[str]) -> list[str]:
        resolved: list[str] = []
        for code in codes:
            key = _norm_code(code)
            col = norm_to_actual.get(key)
            if col and col not in resolved:
                resolved.append(col)
        return resolved

    first = _resolve(BASE_FIRST)

    preset_codes = ORDER_PRESETS.get(_norm(set_name), [])
    middle = [col for col in _resolve(preset_codes) if col not in first]

    used = set(first) | set(middle)
    tail = sorted((col for col in cols if col not in used), key=lambda c: _norm(c))

    return first + middle + tail


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


def _norm_label(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00A0", " ")
    s = s.strip().casefold()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("&", "and")
    s = s.replace("’", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s


def _format_multiselect_display_value(values) -> str:
    parts = _split_multiselect_input(values)
    if not parts:
        return ""
    return ", ".join(parts)


def _pin_sku_name(order: Iterable[str], df_cols: Iterable[str]) -> list[str]:
    if not isinstance(order, list):
        try:
            order = list(order or [])  # type: ignore[arg-type]
        except TypeError:
            order = []
    if not isinstance(df_cols, list):
        try:
            df_cols = list(df_cols or [])  # type: ignore[arg-type]
        except TypeError:
            df_cols = []
    lead = [c for c in ("sku", "name") if c in df_cols]
    rest = [c for c in order if c not in lead and c in df_cols]
    missing = [c for c in df_cols if c not in lead and c not in rest]
    return lead + rest + missing


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

    step2_state = _ensure_step2_state()

    columns_to_process = set(multiselect_columns or [])
    for column in columns_to_process:
        if column in storage.columns and column != "categories":
            storage[column] = storage[column].apply(_split_multiselect_input)

    if "categories" in storage.columns:
        from app.state.step2 import _labels_to_ids

        categories_meta = step2_state.get("categories_meta", {})
        label_to_id = {
            (opt.get("label") or "").strip().lower(): opt.get("value")
            for cat_id, meta in categories_meta.items()
            for opt in meta.get("options") or []
            if opt.get("label") and opt.get("value") is not None
        }

        def _convert_category_cell(cell):
            if isinstance(cell, str):
                parts = [p.strip() for p in cell.split(",") if p.strip()]
            elif isinstance(cell, (list, tuple)):
                parts = cell
            else:
                return []

            ids = _labels_to_ids(parts, label_to_id)
            return [int(x) for x in ids if str(x).isdigit()]

        storage["categories"] = storage["categories"].map(_convert_category_cell)
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
    step2_state = st.session_state.get("step2")
    if step2_state is None:
        return
    if not isinstance(step2_state, dict):
        st.session_state["step2"] = {}
        return

    for key in (
        "wide",
        "wide_orig",
        "wide_synced",
        "dfs",
        "original",
        "staged",
        "wide_meta",
        "current_set_id",
        "ai_suggestions",
        "ai_cells",
        "ai_errors",
        "ai_logs",
    ):
        step2_state.pop(key, None)


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
    state.setdefault("ai_suggestions", {})
    state.setdefault("ai_cells", {})
    state.setdefault("ai_errors", [])
    state.setdefault("ai_logs", {})
    return state


@dataclass
class Step3Product:
    sku: str
    name: str
    attribute_set_id: int | None
    attribute_set_name: str
    generate: bool
    short_description: str
    attributes: dict[str, object]


def _save_step3_autosave(
    descriptions: Mapping[str, Mapping[str, object]] | None,
    rows: Sequence[Mapping[str, object]] | None = None,
) -> None:
    if not isinstance(descriptions, Mapping) or not descriptions:
        return

    sanitized: dict[str, dict[str, str]] = {}
    for sku, payload in descriptions.items():
        if not isinstance(payload, Mapping):
            continue
        sanitized[str(sku)] = {
            lang: _clean_description_value(value)
            for lang, value in payload.items()
            if isinstance(lang, str)
        }

    if not sanitized:
        return

    data: dict[str, object] = {"descriptions": sanitized}

    if rows:
        serialized_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            serialized_rows.append(
                {str(key): _clean_description_value(value) for key, value in row.items()}
            )
        if serialized_rows:
            data["products"] = serialized_rows

    try:
        STEP3_AUTOSAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STEP3_AUTOSAVE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - file system failures
        logger.warning("Failed to persist Step 3 autosave: %s", exc)


def _try_restore_step3_autosave() -> None:
    descriptions = st.session_state.get("descriptions")
    if isinstance(descriptions, Mapping) and descriptions:
        return

    if not STEP3_AUTOSAVE_PATH.exists():
        return

    try:
        with STEP3_AUTOSAVE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # pragma: no cover - file system failures
        logger.warning("Failed to load Step 3 autosave: %s", exc)
        return

    if not isinstance(payload, Mapping):
        return

    restored_descriptions: dict[str, dict[str, str]] = {}
    raw_descriptions = payload.get("descriptions")
    if isinstance(raw_descriptions, Mapping):
        for sku, values in raw_descriptions.items():
            if not isinstance(values, Mapping):
                continue
            restored_descriptions[str(sku)] = {
                lang: _clean_description_value(value)
                for lang, value in values.items()
                if isinstance(lang, str)
            }

    rows_payload = payload.get("products")
    if isinstance(rows_payload, list):
        st.session_state["step3_output_rows"] = rows_payload

    if restored_descriptions:
        st.session_state["descriptions"] = restored_descriptions
        st.info("✅ Step 3 data restored from autosave.")
    elif isinstance(rows_payload, list) and rows_payload:
        st.info("✅ Step 3 data restored from autosave.")

ACOUSTIC_SET_NAMES = {"acoustic guitar"}
ELECTRIC_SET_NAMES = {"electric guitar"}
BASS_SET_NAMES = {"bass guitar"}
AMP_SET_NAMES = {"amps", "amp"}
EFFECT_SET_NAMES = {"effects", "effect"}
ACCESSORY_SET_NAMES = {"accessories", "default"}

BRAND_ACCENTS: dict[str, str] = {
    "martin": (
        "Highlight Martin's pre-war heritage, meticulous hand-scalloped bracing, "
        "and the warm projection that makes these acoustics stage-ready."
    ),
    "taylor": (
        "Reference Taylor's precision bolt-on NT neck, ultra-consistent build quality, "
        "and the articulate, modern voice that records beautifully."
    ),
    "gibson": (
        "Mention Gibson's round-shoulder legacy, scalloped X-bracing, and the rich "
        "midrange that supports vocalists in Americana and rock."
    ),
    "guild": (
        "Talk about Guild's arched-back projection, robust mahogany builds, and their "
        "folk scene credentials from the '60s onward."
    ),
    "fender": (
        "Note Fender's fast-playing necks, stage-ready electronics, and the balance "
        "between sparkle and warmth players expect from the brand."
    ),
}

ACOUSTIC_BODY_KEYWORDS = {
    "dreadnought": "Dreadnought",
    "om": "OM/000",
    "000": "OM/000",
    "ooo": "OM/000",
    "grand auditorium": "Grand Auditorium",
    "jumbo": "Jumbo",
    "parlor": "Parlor",
    "parlour": "Parlor",
    "concert": "Concert",
    "orchestra": "OM/000",
}

BASS_INTROS = [
    "A modern take on a classic electric bass guitar, the {name} is ready for the groove.",
    "Plug in the {name} and you immediately feel how effortlessly it fills the low end.",
    "The {name} delivers boutique-level bass tones with roadworthy reliability.",
]

BASS_OUTROS = [
    "Plug in, set your tone, and let the music happen—this bass is ready.",
    "Every {brand} bass is inspected and set up by Max Guitar’s luthiers before it ships.",
    "From the first rehearsal to the biggest stage, this bass leaves our workshop tour-ready.",
]

BASS_ANGLES = [
    "Angle: Focus on how the woods and construction influence the fundamental punch and sustain.",
    "Angle: Emphasize pickup configuration, onboard electronics, and how they sit in a dense mix.",
    "Angle: Highlight ergonomics—neck profile, balance on a strap, and long-session comfort.",
    "Angle: Discuss dynamic response for fingerstyle, pick, and slap techniques.",
]

AMP_INTROS = [
    "The {name} amplifier was engineered to translate every nuance of your playing to the crowd.",
    "Dial in the {name} and discover how responsive a great guitar amp can be.",
    "Designed for discerning players, the {name} gives you stage power with studio refinement.",
]

AMP_OUTROS = [
    "Our amp technicians bias, noise-test, and approve each unit—it's ready for the next gig.",
    "Every amplifier is inspected, tuned, and signed off by the Max Guitar workshop team.",
    "Bench-tested by our specialists, this amp leaves the shop fully dialed-in and reliable.",
]

AMP_ANGLES = [
    "Angle: Explain how the amp handles pedals, FX loops, and channel switching on stage.",
    "Angle: Focus on speaker configuration, cabinet design, and how it projects in different venues.",
    "Angle: Highlight tube vs. solid-state topology, feel under the fingers, and dynamic response.",
]

EFFECTS_INTROS = [
    "Stomp on the {name} and the character of your tone changes instantly.",
    "The {name} guitar effects pedal captures the signature sound players chase on countless records.",
    "Built for creative guitarists, the {name} turns any board into a tone laboratory.",
]

EFFECTS_OUTROS = [
    "Each pedal is tested for footswitch reliability and low-noise performance before shipping.",
    "Our electronics specialist checks power, switching, and audio path so your effect arrives gig-ready.",
    "We verify bypass integrity, control taper, and noise floor—this effect is ready for the stage.",
]

EFFECTS_ANGLES = [
    "Angle: Detail the circuit architecture, analog vs. digital design, and how controls reshape tone.",
    "Angle: Describe famous rigs or genres where this effect shines, from ambient washes to tight riffs.",
    "Angle: Focus on pedalboard practicality—power options, enclosure toughness, and switching behavior.",
]

ACCESSORY_INTROS = [
    "The {name} is a guitar accessory designed to make daily playing smoother.",
    "Elevate your rig with the {name}, a purpose-built guitar accessory from Max Guitar.",
]

ACCESSORY_OUTROS = [
    "Inspected by our team for authenticity and quality, it ships ready to support your next session.",
    "Every accessory leaving Max Guitar is checked for durability and real-world usability.",
]

TRANSLATION_MODEL = "gpt-4.1-mini"
TRANSLATION_MODEL_FALLBACK = OPENAI_MODEL_ALIASES.get("4-mini") or "gpt-4o-mini"
ACOUSTIC_EN_MODEL = OPENAI_MODEL_ALIASES.get("5") or "gpt-5"
ELECTRIC_EN_MODEL = OPENAI_MODEL_ALIASES.get("5") or "gpt-5"
BASS_MODEL = OPENAI_MODEL_ALIASES.get("5") or "gpt-5"
AMP_MODEL = OPENAI_MODEL_ALIASES.get("5") or "gpt-5"
EFFECT_MODEL = OPENAI_MODEL_ALIASES.get("5") or "gpt-5"
ACCESSORY_MODEL = OPENAI_MODEL_ALIASES.get("5-mini") or DEFAULT_OPENAI_MODEL


def _commit_step2_staged(step2_state: dict | None) -> None:
    if not isinstance(step2_state, dict):
        return

    staged_map = (step2_state.get("staged") or {}).copy()
    if not staged_map:
        return

    for set_id, draft in staged_map.items():
        if not isinstance(draft, pd.DataFrame):
            continue
        draft_wide = draft.copy(deep=True)
        step2_state.setdefault("wide", {})[set_id] = draft_wide
        step2_state.setdefault("wide_orig", {})[set_id] = draft_wide.copy(deep=True)
        existing_long = (step2_state.get("dfs") or {}).get(set_id)
        updated_long = apply_wide_edits_to_long(existing_long, draft_wide)
        if isinstance(updated_long, pd.DataFrame):
            if "value" in updated_long.columns:
                updated_long["value"] = updated_long["value"].astype(object)
            step2_state.setdefault("dfs", {})[set_id] = updated_long
            step2_state.setdefault("original", {})[set_id] = updated_long.copy(deep=True)
        else:
            step2_state.setdefault("dfs", {})[set_id] = draft_wide.copy(deep=True)
            step2_state.setdefault("original", {})[set_id] = draft_wide.copy(deep=True)
    step2_state.setdefault("staged", {}).clear()


def _normalize_text(value: object) -> str:
    if isinstance(value, float) and pd.isna(value):
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _clean_description_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return _normalize_text(value)


def _strip_truncation_warning(value: object) -> str:
    text = _clean_description_value(value)
    prefix = "⚠️ "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _is_translation_truncated(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if "<p" in stripped and not stripped.endswith("</p>"):
        return True
    return False


def _decorate_truncated_translation(value: object) -> str:
    text = _strip_truncation_warning(value)
    if _is_translation_truncated(text):
        return f"⚠️ {text}"
    return text


def _ensure_html_description(value: object) -> str:
    text = _clean_description_value(value)
    if not text:
        return ""

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, Mapping):
        for key in ("description", "html", "body", "en"):
            candidate = parsed.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return text


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return bool(value)
    return bool(value)


def _extract_video_embed(text: object) -> tuple[str, str]:
    raw = _normalize_text(text)
    if not raw:
        return "", ""

    iframe_with_wrapper = re.compile(
        r"(<(?:div|figure)[^>]*>.*?<iframe.*?</iframe>.*?</(?:div|figure)>)",
        re.IGNORECASE | re.DOTALL,
    )
    iframe_only = re.compile(r"(<iframe.*?</iframe>)", re.IGNORECASE | re.DOTALL)

    match = iframe_with_wrapper.search(raw)
    if not match:
        match = iframe_only.search(raw)

    if not match:
        return "", raw

    embed = match.group(1).strip()
    remainder = (raw[: match.start()] + raw[match.end() :]).strip()
    return embed, remainder


def _restore_embed(embed: str, body: str) -> str:
    body_clean = body.strip()
    if not embed:
        return body_clean
    if not body_clean:
        return embed
    return f"{embed}\n{body_clean}"


def _prepend_embed_if_valid(embed: str, text: str) -> str:
    embed_clean = (embed or "").strip()
    if not embed_clean or "<iframe" not in embed_clean.lower():
        return _clean_description_value(text)

    text_clean = _clean_description_value(text)
    if not text_clean:
        return embed_clean
    if embed_clean in text_clean:
        return text_clean
    return f"{embed_clean}\n{text_clean}"


def _detect_body_shape(name: str, attributes: Mapping[str, object]) -> str | None:
    combined = " ".join(
        part
        for part in [name, _normalize_text(attributes.get("acoustic_body_shape"))]
        if part
    ).lower()
    for keyword, label in ACOUSTIC_BODY_KEYWORDS.items():
        if keyword in combined:
            return label
    return None


def _resolve_brand_accent(brand_value: object, name: str) -> str | None:
    brand = _normalize_text(brand_value) or _normalize_text(name.split()[0] if name else "")
    lowered = brand.lower()
    if lowered in BRAND_ACCENTS:
        return BRAND_ACCENTS[lowered]
    for key, accent in BRAND_ACCENTS.items():
        if key in lowered:
            return accent
    return None


def _format_attribute_summary(attributes: Mapping[str, object]) -> str:
    if not isinstance(attributes, Mapping):
        return ""
    lines: list[str] = []
    for key, value in attributes.items():
        if key in {"sku", "name", "short_description", GENERATE_DESCRIPTION_COLUMN}:
            continue
        normalized = _normalize_text(value)
        if not normalized:
            continue
        lines.append(f"- {key}: {normalized}")
    return "\n".join(lines)


def _openai_client_kwargs() -> dict[str, object]:
    api_key = st.session_state.get("ai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured.")
    base_override = (
        st.session_state.get("openai_api_base")
        or os.getenv("OPENAI_API_BASE", "").strip()
    )
    kwargs: dict[str, object] = {"api_key": api_key}
    if base_override:
        kwargs["base_url"] = base_override.rstrip("/")
    return kwargs


def _call_openai_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1100,
    response_format: dict[str, object] | None = None,
    timeout: int = 120,
) -> str:
    client_kwargs = _openai_client_kwargs()
    try:
        client = openai.OpenAI(**client_kwargs)
    except Exception as exc:  # pragma: no cover - network client init
        raise RuntimeError(f"OpenAI client init failed: {exc}") from exc

    resolved_model = resolve_model_name(model, default=model)
    json_suffix = "Respond ONLY with a valid minified JSON. No markdown. No prose. No formatting instructions."
    if response_format:
        if system_prompt:
            if json_suffix not in system_prompt:
                system_prompt = f"{system_prompt.rstrip()}\n{json_suffix}"
        else:
            system_prompt = json_suffix

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs: dict[str, object] = {
        "model": resolved_model,
        "messages": messages,
        "timeout": timeout,
    }
    if response_format:
        kwargs["response_format"] = response_format
    kwargs["temperature"] = 0
    temperature = 0
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens

    trace(
        {
            "where": "step3:openai_call",
            "model": resolved_model,
            "temperature": temperature,
            "keys": sorted(kwargs.keys()),
        }
    )

    try:
        chat = client.chat.completions.create(**kwargs)
    except Exception as exc:  # pragma: no cover - API failure handling
        stripped_kwargs = dict(kwargs)
        for key in ("temperature", "response_format", "max_completion_tokens"):
            stripped_kwargs.pop(key, None)
        if stripped_kwargs != kwargs:
            trace(
                {
                    "where": "step3:openai_retry",
                    "model": resolved_model,
                    "reason": str(exc)[:200],
                }
            )
            try:
                chat = client.chat.completions.create(**stripped_kwargs)
            except Exception as retry_exc:  # pragma: no cover - API failure handling
                raise RuntimeError(f"OpenAI request failed: {retry_exc}") from retry_exc
        else:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    try:
        content = chat.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:  # pragma: no cover - response guard
        raise RuntimeError("OpenAI response missing content") from exc

    return content.strip()


class JSONPayloadError(ValueError):
    def __init__(
        self,
        message: str,
        raw: str,
        *,
        required_keys: Iterable[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.required_keys = tuple(required_keys or ())


def extract_json_payload(
    raw: object, *, required_keys: Iterable[str] | None = None
) -> Mapping[str, object] | None:
    if not isinstance(raw, str):
        return None

    required = tuple(required_keys or ())
    parsed = extract_json_object(raw, required_keys=required if required else None)
    if isinstance(parsed, Mapping):
        return parsed

    if not required:
        return None

    if all(key in raw for key in required):
        fallback: dict[str, object] = {}
        for key in required:
            pattern = rf'"{re.escape(key)}"\s*:\s*"(.+?)"'
            match = re.search(pattern, raw, re.DOTALL)
            if not match:
                fallback = {}
                break
            raw_value = match.group(1).strip()
            decoded_value: object = raw_value
            try:
                decoded_value = json.loads(f'"{raw_value}"')
            except json.JSONDecodeError:
                if isinstance(raw_value, str):
                    decoded_value = (
                        raw_value.replace("\\\"", '"')
                        .replace("\\n", "\n")
                        .replace("\\r", "\r")
                        .replace("\\t", "\t")
                    )
            if isinstance(decoded_value, str):
                decoded_value = decoded_value.strip()
            fallback[key] = decoded_value
        if fallback and len(fallback) == len(required):
            return fallback

    return None


def _call_openai_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.4,
    timeout: int = 120,
    required_keys: Iterable[str] | None = ("en",),
    max_tokens: int | None = None,
) -> dict[str, object]:
    text_kwargs: dict[str, object] = {
        "temperature": temperature,
        "timeout": timeout,
        "response_format": {"type": "json_object"},
    }
    if max_tokens is not None:
        text_kwargs["max_tokens"] = max_tokens

    raw = _call_openai_text(
        model,
        system_prompt,
        user_prompt,
        **text_kwargs,
    )
    parsed = extract_json_payload(raw, required_keys=required_keys)
    if not parsed:
        raise JSONPayloadError(
            f"Failed to parse JSON from model: {short_preview_of(raw)}",
            raw,
            required_keys=required_keys,
        )
    return parsed


def _choose_random(texts: Sequence[str], fallback: str = "") -> str:
    candidates = [t for t in texts if isinstance(t, str) and t.strip()]
    if not candidates:
        return fallback
    return random.choice(candidates)


def _translate_description(english_text: str) -> tuple[str, str, str, str]:
    if not english_text:
        return "", "", "", ""
    model = TRANSLATION_MODEL
    fallback_model = (
        TRANSLATION_MODEL_FALLBACK
        if TRANSLATION_MODEL_FALLBACK and TRANSLATION_MODEL_FALLBACK != model
        else None
    )
    raw_candidates: list[str] = []

    language_names = {
        "nl": "Dutch",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
    }

    def _format_language_list(labels: Sequence[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return " and ".join(labels)
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    def _run_models(
        text: str,
        attempt_label: str,
        languages: tuple[str, ...],
    ) -> dict[str, object] | None:
        local_raw: list[str] = []

        json_keys = ", ".join(f"\\\"{lang}\\\": \\\"...\\\"" for lang in languages)
        labels = [f"{language_names.get(lang, lang)} ({lang})" for lang in languages]
        language_list = _format_language_list(labels)
        system_prompt = (
            "You translate professional musical instrument copy. Respond ONLY "
            f"with JSON: {{{json_keys}}}. Do not wrap inside 'description' or 'name'. "
            "No markdown. No prose."
        )
        user_prompt = (
            "Translate the following English guitar product description into "
            f"{language_list}. Use natural marketing language and keep line breaks.\n\n"
            f"TEXT:\n{text.strip()}"
        )

        def _attempt(model_name: str) -> dict[str, object] | None:
            try:
                return _call_openai_json(
                    model_name,
                    system_prompt,
                    user_prompt,
                    temperature=0.2,
                    required_keys=languages,
                    max_tokens=2500,
                )
            except JSONPayloadError as exc:
                trace(
                    {
                        "where": "step3:translate_error",
                        "model": model_name,
                        "err": str(exc)[:200],
                        "preview": short_preview_of(exc.raw),
                        "attempt": attempt_label,
                    }
                )
                if exc.raw:
                    local_raw.append(exc.raw)
                return None

        fallback_tried = False
        payload: dict[str, object] | None = None

        try:
            try:
                payload = _attempt(model)
            except Exception as exc:
                trace(
                    {
                        "where": "step3:translate_error",
                        "model": model,
                        "err": str(exc)[:200],
                        "attempt": attempt_label,
                    }
                )
                if fallback_model:
                    fallback_tried = True
                    try:
                        payload = _attempt(fallback_model)
                    except Exception as fallback_exc:
                        trace(
                            {
                                "where": "step3:translate_error",
                                "model": fallback_model,
                                "err": str(fallback_exc)[:200],
                                "attempt": attempt_label,
                            }
                        )
                        raise
                else:
                    raise

            if payload is None and fallback_model and not fallback_tried:
                try:
                    payload = _attempt(fallback_model)
                    fallback_tried = True
                except Exception as exc:
                    trace(
                        {
                            "where": "step3:translate_error",
                            "model": fallback_model,
                            "err": str(exc)[:200],
                            "attempt": attempt_label,
                        }
                    )
                    raise
        finally:
            raw_candidates.extend(local_raw)

        return payload

    results: dict[str, str] = {}

    def _translate_pair(languages: tuple[str, ...]) -> dict[str, object] | None:
        pair_label = "/".join(languages)
        payload = _run_models(english_text, f"full:{pair_label}", languages)
        if payload is None and len(english_text) > 4000:
            paragraphs = english_text.split("\n\n")
            shortened_text = "\n\n".join(paragraphs[:3]).strip()
            if shortened_text and len(shortened_text) < len(english_text):
                payload = _run_models(shortened_text, f"shortened:{pair_label}", languages)
        return payload

    for language_pair in (("nl", "de"), ("es", "fr")):
        payload = _translate_pair(language_pair)
        if payload:
            for lang in language_pair:
                value = payload.get(lang) or payload.get(lang.upper())
                results[lang] = _normalize_text(value)

    if not all(results.get(lang) for lang in ("nl", "de", "es", "fr")):
        nl_value = results.get("nl", "")
        de_value = results.get("de", "")
        es_value = results.get("es", "")
        fr_value = results.get("fr", "")
        for raw in raw_candidates:
            parsed_candidate = extract_json_payload(raw)
            if isinstance(parsed_candidate, Mapping):
                if not nl_value:
                    nl_candidate = parsed_candidate.get("nl") or parsed_candidate.get("NL")
                    nl_value = _normalize_text(nl_candidate)
                if not de_value:
                    de_candidate = parsed_candidate.get("de") or parsed_candidate.get("DE")
                    de_value = _normalize_text(de_candidate)
                if not es_value:
                    es_candidate = parsed_candidate.get("es") or parsed_candidate.get("ES")
                    es_value = _normalize_text(es_candidate)
                if not fr_value:
                    fr_candidate = parsed_candidate.get("fr") or parsed_candidate.get("FR")
                    fr_value = _normalize_text(fr_candidate)
            if not nl_value:
                partial_nl = extract_json_payload(raw, required_keys=("nl",))
                if isinstance(partial_nl, Mapping):
                    nl_candidate = partial_nl.get("nl") or partial_nl.get("NL")
                    nl_value = _normalize_text(nl_candidate)
            if not nl_value:
                partial_nl_upper = extract_json_payload(raw, required_keys=("NL",))
                if isinstance(partial_nl_upper, Mapping):
                    nl_candidate = partial_nl_upper.get("NL") or partial_nl_upper.get("nl")
                    nl_value = _normalize_text(nl_candidate)
            if not de_value:
                partial_de = extract_json_payload(raw, required_keys=("de",))
                if isinstance(partial_de, Mapping):
                    de_candidate = partial_de.get("de") or partial_de.get("DE")
                    de_value = _normalize_text(de_candidate)
            if not de_value:
                partial_de_upper = extract_json_payload(raw, required_keys=("DE",))
                if isinstance(partial_de_upper, Mapping):
                    de_candidate = partial_de_upper.get("DE") or partial_de_upper.get("de")
                    de_value = _normalize_text(de_candidate)
            if not es_value:
                partial_es = extract_json_payload(raw, required_keys=("es",))
                if isinstance(partial_es, Mapping):
                    es_candidate = partial_es.get("es") or partial_es.get("ES")
                    es_value = _normalize_text(es_candidate)
            if not es_value:
                partial_es_upper = extract_json_payload(raw, required_keys=("ES",))
                if isinstance(partial_es_upper, Mapping):
                    es_candidate = partial_es_upper.get("ES") or partial_es_upper.get("es")
                    es_value = _normalize_text(es_candidate)
            if not fr_value:
                partial_fr = extract_json_payload(raw, required_keys=("fr",))
                if isinstance(partial_fr, Mapping):
                    fr_candidate = partial_fr.get("fr") or partial_fr.get("FR")
                    fr_value = _normalize_text(fr_candidate)
            if not fr_value:
                partial_fr_upper = extract_json_payload(raw, required_keys=("FR",))
                if isinstance(partial_fr_upper, Mapping):
                    fr_candidate = partial_fr_upper.get("FR") or partial_fr_upper.get("fr")
                    fr_value = _normalize_text(fr_candidate)
            if nl_value and de_value and es_value and fr_value:
                break
        if nl_value or de_value or es_value or fr_value:
            trace(
                {
                    "where": "step3:translate_partial",
                    "nl_available": bool(nl_value),
                    "de_available": bool(de_value),
                    "es_available": bool(es_value),
                    "fr_available": bool(fr_value),
                }
            )
            return nl_value, de_value, es_value, fr_value
        raise RuntimeError("Failed to parse translation JSON")

    return (
        results.get("nl", ""),
        results.get("de", ""),
        results.get("es", ""),
        results.get("fr", ""),
    )


def _categorize_product(product: Step3Product) -> str:
    name_norm = _normalize_text(product.attribute_set_name).lower()

    def _matches(candidates: set[str]) -> bool:
        for candidate in candidates:
            if candidate and candidate in name_norm:
                return True
        return False

    if _matches(ACOUSTIC_SET_NAMES):
        return "acoustic"
    if _matches(ELECTRIC_SET_NAMES):
        return "electric"
    if _matches(BASS_SET_NAMES):
        return "bass"
    if _matches(AMP_SET_NAMES):
        return "amp"
    if _matches(EFFECT_SET_NAMES):
        return "effect"
    if _matches(ACCESSORY_SET_NAMES):
        return "accessory"

    try:
        attr_set_id = int(product.attribute_set_id) if product.attribute_set_id is not None else None
    except (TypeError, ValueError):
        attr_set_id = None
    if attr_set_id is not None and attr_set_id in ELECTRIC_SET_IDS:
        return "electric"
    return "accessory"


def _generate_acoustic_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    brand = product.attributes.get("brand")
    brand_accent = _resolve_brand_accent(brand, product.name)
    body_shape = _detect_body_shape(product.name, product.attributes)

    user_lines = [
        f"SKU: {product.sku}",
        f"Model name: {product.name}",
    ]
    set_name_clean = _normalize_text(product.attribute_set_name)
    if set_name_clean:
        user_lines.append(f"Attribute set: {set_name_clean}")
    if _normalize_text(brand):
        user_lines.append(f"Brand: {_normalize_text(brand)}")
    if body_shape:
        user_lines.append(f"Body shape hint: {body_shape}")
    if brand_accent:
        user_lines.append(f"Brand accent to reflect: {brand_accent}")
    if hint:
        user_lines.append(f"Merchandising hint: {hint}")
    if existing_body:
        user_lines.append(
            "Existing short description (use as factual input, rewrite completely):"
        )
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Useful attributes:\n" + attr_summary)

    user_lines.append(
        "Write a professional 3-4 paragraph HTML short description (250-400 words). "
        "Use <p> tags, mention tonewoods, top/back/sides, bracing pattern, neck profile, scale, "
        "electronics if present, and performance context. Close with the exact sentence "
        "'Every guitar is inspected and set up by the Max Guitar luthier team.'"
    )

    system_prompt = (
        "You are a copywriter for premium acoustic guitars. Use SEO-friendly, musical language, "
        "stay factual, and avoid inventing specs not provided."
    )

    english_body = _call_openai_text(
        ACOUSTIC_EN_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.65,
    )
    english_body = _ensure_html_description(english_body)

    nl, de, es, fr = _translate_description(english_body)
    es = _prepend_embed_if_valid(embed, es)
    fr = _prepend_embed_if_valid(embed, fr)
    final_en = _restore_embed(embed, english_body)
    return final_en, nl, de, es, fr


def _generate_electric_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    brand = _normalize_text(product.attributes.get("brand"))

    user_lines = [
        f"SKU: {product.sku}",
        f"Model name: {product.name}",
    ]
    if brand:
        user_lines.append(f"Brand: {brand}")
    set_name_clean = _normalize_text(product.attribute_set_name)
    if set_name_clean:
        user_lines.append(f"Attribute set: {set_name_clean}")
    if hint:
        user_lines.append(f"Merchandising hint: {hint}")
    if existing_body:
        user_lines.append("Existing short description (reference only, rewrite):")
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Relevant attributes:\n" + attr_summary)

    user_lines.append(
        "Produce a 250-400 word English HTML description (<p> tags) that naturally uses the phrase "
        "'electric guitar'. Cover body and neck materials, construction (bolt-on/set), pickup types, "
        "controls, tonal range (clean to driven), playability (neck profile, radius), and stage/studio use. "
        "End with 'Every guitar is inspected and set up by the Max Guitar luthier team.'"
    )

    system_prompt = (
        "You create SEO-oriented copy for electric guitars. Stay factual, vivid, and musician-focused."
    )

    english_body = _call_openai_text(
        ELECTRIC_EN_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.7,
    )
    english_body = _ensure_html_description(english_body)

    nl, de, es, fr = _translate_description(english_body)
    es = _prepend_embed_if_valid(embed, es)
    fr = _prepend_embed_if_valid(embed, fr)
    final_en = _restore_embed(embed, english_body)
    return final_en, nl, de, es, fr


def _generate_accessory_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    brand = _normalize_text(product.attributes.get("brand"))

    intro = _choose_random(ACCESSORY_INTROS, fallback=f"The {product.name} is a guitar accessory built for daily use.")
    outro = _choose_random(ACCESSORY_OUTROS, fallback="Checked by Max Guitar staff before it leaves the shop.")

    user_lines = [
        f"SKU: {product.sku}",
        f"Product name: {product.name}",
        f"Intro sentence to echo: {intro}",
        f"Closing sentence to include: {outro}",
    ]
    if brand:
        user_lines.append(f"Brand: {brand}")
    if hint:
        user_lines.append(f"Usage hint: {hint}")
    if existing_body:
        user_lines.append("Existing description snippet (for facts only):")
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Attributes:\n" + attr_summary)

    user_lines.append(
        "Write a persuasive 2-3 paragraph HTML (<p>) description, about 150-250 words, mentioning how the accessory "
        "supports guitarists (protection, convenience, tone, etc.). Use 'guitar accessory' and the accessory type in context."
    )

    system_prompt = (
        "You write marketing copy for guitar accessories. Keep tone premium yet practical, remain factual, and avoid hype."
    )

    english_body = _call_openai_text(
        ACCESSORY_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.6,
        max_tokens=800,
    )
    english_body = _ensure_html_description(english_body)

    nl, de, es, fr = _translate_description(english_body)
    es = _prepend_embed_if_valid(embed, es)
    fr = _prepend_embed_if_valid(embed, fr)
    final_en = _restore_embed(embed, english_body)
    return final_en, nl, de, es, fr


def _generate_bass_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    brand = _normalize_text(product.attributes.get("brand")) or product.name
    intro_template = _choose_random(BASS_INTROS)
    outro_template = _choose_random(BASS_OUTROS)
    angle = _choose_random(BASS_ANGLES)

    user_lines = [
        f"SKU: {product.sku}",
        f"Bass name: {product.name}",
        f"Intro to echo: {intro_template.format(name=product.name)}",
        f"Outro to include: {outro_template.format(brand=brand, name=product.name)}",
        angle,
    ]
    if hint:
        user_lines.append(f"Retailer hint: {hint}")
    if existing_body:
        user_lines.append("Existing text (reference only):")
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Attributes:\n" + attr_summary)

    user_lines.append(
        "Return JSON with keys en, nl, de, es, fr. The 'en' value must be 3-4 HTML paragraphs (<p>) "
        "totalling around 220-320 words. Describe tonewoods, neck feel, pickup configuration, onboard electronics, "
        "and playing techniques (fingerstyle, pick, slap). The 'nl', 'de', 'es', and 'fr' values should be faithful translations."
    )

    system_prompt = (
        "You write marketing copy for electric bass guitars. Use musical vocabulary and stay factual."
    )

    payload = _call_openai_json(
        BASS_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.6,
        required_keys=("en", "nl", "de", "es", "fr"),
    )

    en_body = _clean_description_value(payload.get("en"))
    nl_body = _clean_description_value(payload.get("nl"))
    de_body = _clean_description_value(payload.get("de"))
    es_body = _clean_description_value(payload.get("es"))
    fr_body = _clean_description_value(payload.get("fr"))
    if not en_body:
        raise RuntimeError("Bass description empty")
    es_body = _prepend_embed_if_valid(embed, es_body)
    fr_body = _prepend_embed_if_valid(embed, fr_body)
    final_en = _restore_embed(embed, en_body)
    return final_en, nl_body, de_body, es_body, fr_body


def _generate_amp_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    intro_template = _choose_random(AMP_INTROS)
    outro = _choose_random(AMP_OUTROS)
    angle = _choose_random(AMP_ANGLES)

    user_lines = [
        f"SKU: {product.sku}",
        f"Amplifier name: {product.name}",
        f"Intro to reflect: {intro_template.format(name=product.name)}",
        f"Outro to include: {outro}",
        angle,
    ]
    if hint:
        user_lines.append(f"Retailer hint: {hint}")
    if existing_body:
        user_lines.append("Existing description (reference only):")
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Attributes:\n" + attr_summary)

    user_lines.append(
        "Return JSON with keys en, nl, de, es, fr. The English text should be 3-4 HTML paragraphs (<p>) "
        "around 220-320 words covering amplifier format (combo/head/cab), power (watts), speaker configuration, "
        "tube vs. solid-state topology, channel/effects details, and pedalboard friendliness. Ensure the final "
        "sentence confirms technician inspection as provided. Provide accurate NL, DE, ES, and FR translations."
    )

    system_prompt = "You craft technical yet musical marketing copy for guitar amplifiers."

    payload = _call_openai_json(
        AMP_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.6,
        required_keys=("en", "nl", "de", "es", "fr"),
    )

    en_body = _clean_description_value(payload.get("en"))
    nl_body = _clean_description_value(payload.get("nl"))
    de_body = _clean_description_value(payload.get("de"))
    es_body = _clean_description_value(payload.get("es"))
    fr_body = _clean_description_value(payload.get("fr"))
    if not en_body:
        raise RuntimeError("Amp description empty")
    es_body = _prepend_embed_if_valid(embed, es_body)
    fr_body = _prepend_embed_if_valid(embed, fr_body)
    final_en = _restore_embed(embed, en_body)
    return final_en, nl_body, de_body, es_body, fr_body


def _generate_effect_copy(product: Step3Product) -> tuple[str, str, str, str, str]:
    embed, existing_body = _extract_video_embed(product.short_description)
    attr_summary = _format_attribute_summary(product.attributes)
    hint = _normalize_text(product.attributes.get("hint"))
    intro_template = _choose_random(EFFECTS_INTROS)
    outro = _choose_random(EFFECTS_OUTROS)
    angle = _choose_random(EFFECTS_ANGLES)
    effect_type = _normalize_text(product.attributes.get("effect_type"))

    user_lines = [
        f"SKU: {product.sku}",
        f"Effect name: {product.name}",
        f"Intro to reflect: {intro_template.format(name=product.name)}",
        f"Outro to include: {outro}",
        angle,
    ]
    if effect_type:
        user_lines.append(f"Effect type: {effect_type}")
    if hint:
        user_lines.append(f"Retailer hint: {hint}")
    if existing_body:
        user_lines.append("Existing copy (reference only):")
        user_lines.append(existing_body)
    if attr_summary:
        user_lines.append("Attributes:\n" + attr_summary)

    user_lines.append(
        "Return JSON with keys en, nl, de, es, fr. The English copy must be 2-3 HTML paragraphs (~180-260 words) "
        "describing circuit topology (analog/digital), control layout, tonal behaviour, power requirements, "
        "and live performance reliability. Include the keywords 'guitar effects pedal' and 'guitar effect' naturally. "
        "Provide accurate NL, DE, ES, and FR translations, keeping HTML."
    )

    system_prompt = "You write persuasive yet factual copy for guitar effects pedals."

    payload = _call_openai_json(
        EFFECT_MODEL,
        system_prompt,
        "\n".join(user_lines),
        temperature=0.55,
        required_keys=("en", "nl", "de", "es", "fr"),
    )

    en_body = _clean_description_value(payload.get("en"))
    nl_body = _clean_description_value(payload.get("nl"))
    de_body = _clean_description_value(payload.get("de"))
    es_body = _clean_description_value(payload.get("es"))
    fr_body = _clean_description_value(payload.get("fr"))
    if not en_body:
        raise RuntimeError("Effect description empty")
    es_body = _prepend_embed_if_valid(embed, es_body)
    fr_body = _prepend_embed_if_valid(embed, fr_body)
    final_en = _restore_embed(embed, en_body)
    return final_en, nl_body, de_body, es_body, fr_body


def _generate_description_for_product(product: Step3Product) -> tuple[str, str, str, str, str]:
    category = _categorize_product(product)
    if category == "acoustic":
        return _generate_acoustic_copy(product)
    if category == "electric":
        return _generate_electric_copy(product)
    if category == "bass":
        return _generate_bass_copy(product)
    if category == "amp":
        return _generate_amp_copy(product)
    if category == "effect":
        return _generate_effect_copy(product)
    return _generate_accessory_copy(product)


def _build_step3_products() -> list[Step3Product]:
    step2_state = _ensure_step2_state()
    _commit_step2_staged(step2_state)

    set_names = step2_state.get("set_names", {})
    wide_map = step2_state.get("wide", {})
    products: list[Step3Product] = []

    for set_id, wide_df in (wide_map or {}).items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue
        set_name_value = set_names.get(set_id, set_id)
        set_name_text = _normalize_text(set_name_value) or str(set_id)
        for _, row in wide_df.iterrows():
            row_dict = {key: row.get(key) for key in row.index}
            sku = _normalize_text(row_dict.get("sku"))
            if not sku:
                continue
            name = _clean_description_value(row_dict.get("name")) or sku
            attr_set_raw = row_dict.get("attribute_set_id", set_id)
            try:
                attr_set_id = int(attr_set_raw) if attr_set_raw not in (None, "") else int(set_id)
            except (TypeError, ValueError):
                try:
                    attr_set_id = int(set_id)
                except (TypeError, ValueError):
                    attr_set_id = None
            generate_flag = _coerce_bool(row_dict.get(GENERATE_DESCRIPTION_COLUMN, True))
            short_desc = _clean_description_value(row_dict.get("short_description"))
            attributes = {
                key: row_dict.get(key)
                for key in row_dict.keys()
                if key not in {"sku", "name"}
            }
            products.append(
                Step3Product(
                    sku=sku,
                    name=name,
                    attribute_set_id=attr_set_id,
                    attribute_set_name=set_name_text,
                    generate=generate_flag,
                    short_description=short_desc,
                    attributes=attributes,
                )
            )

    products.sort(key=lambda item: item.sku)
    return products


def _ensure_descriptions_initialized(products: Sequence[Step3Product]) -> None:
    descriptions = st.session_state.get("descriptions")
    if not isinstance(descriptions, dict):
        descriptions = {}

    for product in products:
        stored = descriptions.get(product.sku)
        if not isinstance(stored, dict):
            stored = {}
        en_text = stored.get("en")
        nl_text = stored.get("nl")
        de_text = stored.get("de")
        es_text = stored.get("es")
        fr_text = stored.get("fr")
        if not _clean_description_value(en_text):
            en_text = product.short_description
        descriptions[product.sku] = {
            "en": _clean_description_value(en_text),
            "nl": _clean_description_value(nl_text),
            "de": _clean_description_value(de_text),
            "es": _clean_description_value(es_text),
            "fr": _clean_description_value(fr_text),
        }

    st.session_state["descriptions"] = descriptions

from concurrent.futures import ThreadPoolExecutor, as_completed


def _generate_descriptions_for_products(products: Sequence[Step3Product]) -> tuple[dict[str, dict[str, str]], list[str]]:
    results: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    total = len(products)
    progress = st.progress(0.0, text=f"Generating descriptions and translations… {total} remaining") if total else None

    def _translate_existing(product: Step3Product):
        sku = product.sku
        embed, body = _extract_video_embed(product.short_description)
        base_text = body or product.short_description
        nl, de, es, fr = _translate_description(base_text)
        if es:
            es = _prepend_embed_if_valid(embed, es)
        if fr:
            fr = _prepend_embed_if_valid(embed, fr)
        return sku, {"en": product.short_description, "nl": nl, "de": de, "es": es, "fr": fr}

    def _generate_all(product: Step3Product):
        sku = product.sku
        en, nl, de, es, fr = _generate_description_for_product(product)
        return sku, {"en": en, "nl": nl, "de": de, "es": es, "fr": fr}

    if total:
        max_workers = min(5, total)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_product = {}
            for product in products:
                func = _generate_all if product.generate else _translate_existing
                future = executor.submit(func, product)
                future_to_product[future] = product

            done = 0
            for future in as_completed(future_to_product):
                prod = future_to_product[future]
                sku = prod.sku
                try:
                    sku, result = future.result()
                    results[sku] = result
                except Exception as exc:
                    errors.append(f"{sku}: {exc}")
                    fallback = {
                        "en": _clean_description_value("AI FAILED TO GENERATE"),
                        "nl": "", "de": "", "es": "", "fr": ""
                    }
                    results[sku] = fallback
                finally:
                    done += 1
                    if progress:
                        remaining = total - done
                        frac = done / total
                        msg = "Generation completed" if remaining <= 0 else f"Generating descriptions… {remaining} remaining"
                        progress.progress(frac, text=msg)

    if progress:
        progress.progress(1.0, text="Generation completed")
    return results, errors


def _apply_descriptions_to_state(descriptions_map: Mapping[str, Mapping[str, object]]) -> set[str]:
    step2_state = _ensure_step2_state()
    _commit_step2_staged(step2_state)
    wide_map = step2_state.get("wide", {})
    updated_skus: set[str] = set()

    if not isinstance(descriptions_map, Mapping):
        return updated_skus

    for set_id, wide_df in (wide_map or {}).items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue
        df_copy = wide_df.copy()
        changed = False
        for idx, row in df_copy.iterrows():
            sku = _normalize_text(row.get("sku"))
            if not sku or sku not in descriptions_map:
                continue
            en_value = _clean_description_value(descriptions_map[sku].get("en"))
            current = _clean_description_value(row.get("short_description"))
            if en_value != current:
                df_copy.at[idx, "short_description"] = en_value
                changed = True
                updated_skus.add(sku)
        if changed:
            step2_state["wide"][set_id] = df_copy

    return updated_skus


def _store_specific_url(base_url: str, store_code: str, sku: str) -> str:
    return f"{base_url.rstrip('/')}/rest/{store_code}/V1/products/{quote(str(sku), safe='')}"


def _save_translations(
    descriptions_map: Mapping[str, Mapping[str, object]],
    *,
    store_codes: Sequence[str] = ("nl", "de"),
) -> tuple[set[str], list[str]]:
    session = st.session_state.get("mg_session")
    base_url = st.session_state.get("mg_base_url")
    if not session or not base_url:
        raise RuntimeError("Magento session is not initialized")

    success: set[str] = set()
    errors: list[str] = []

    for sku, payload in descriptions_map.items():
        if not isinstance(payload, Mapping):
            continue
        for store_code in store_codes:
            text_value = _clean_description_value(payload.get(store_code))
            if not text_value:
                continue
            url = _store_specific_url(base_url, store_code, sku)
            data = {
                "product": {
                    "sku": sku,
                    "custom_attributes": [
                        {"attribute_code": "short_description", "value": text_value}
                    ],
                }
            }
            try:
                resp = session.put(
                    url,
                    json=data,
                    headers=build_magento_headers(session=session),
                    timeout=30,
                )
            except Exception as exc:  # pragma: no cover - network error handling
                errors.append(f"{sku} ({store_code}): {exc}")
                trace(
                    {
                        "where": "step3:save_translation_error",
                        "sku": sku,
                        "store": store_code,
                        "err": str(exc)[:200],
                    }
                )
                continue
            if not resp.ok:
                snippet = resp.text[:200] if hasattr(resp, "text") else str(resp)
                errors.append(f"{sku} ({store_code}): {resp.status_code} {snippet}")
                trace(
                    {
                        "where": "step3:save_translation_http_error",
                        "sku": sku,
                        "store": store_code,
                        "status": getattr(resp, "status_code", None),
                        "body": snippet,
                    }
                )
                continue
            success.add(f"{sku}:{store_code}")
            trace({"where": "step3:save_translation_ok", "sku": sku, "store": store_code})

    return success, errors


def save_step3_to_magento() -> None:
    descriptions_map = st.session_state.get("descriptions")
    if not isinstance(descriptions_map, Mapping) or not descriptions_map:
        st.warning("⚠️ Nothing to save. Please repeat Step 3.")
        return

    updated_skus = _apply_descriptions_to_state(descriptions_map)
    trace({"where": "step3:update_applied", "skus": sorted(updated_skus)})

    save_step2_to_magento()

    try:
        success, errors = _save_translations(descriptions_map)
    except Exception as exc:
        st.error(f"Failed to save translations: {exc}")
        trace({"where": "step3:save_translation_exception", "err": str(exc)[:200]})
        return

    if success:
        st.success("Translations saved for: " + ", ".join(sorted(success)))
    if errors:
        st.warning("Some translations failed:")
        st.write("\n".join(f"- {msg}" for msg in errors))


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

    if "value" in updated.columns:
        updated["value"] = updated["value"].astype(object)

    for row_idx, row in updated.iterrows():
        sku = row.get("sku")
        code = row.get("attribute_code")
        if not sku or not code:
            continue
        if sku not in wide_indexed.index:
            continue
        if code not in wide_indexed.columns:
            continue
        raw_value = wide_indexed.at[sku, code]
        if isinstance(raw_value, list):
            updated.at[row_idx, "value"] = ", ".join(str(v) for v in raw_value)
        else:
            updated.at[row_idx, "value"] = raw_value

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


def _apply_ai_suggestions_to_wide(
    wide_df: pd.DataFrame,
    suggestions: dict[str, dict[str, dict]],
    meta_map: dict[str, dict],
    meta_cache=None,
    session=None,
    api_base: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
        return wide_df, [], []
    if not isinstance(suggestions, dict):
        return wide_df, [], []

    DEBUG = bool(st.session_state.get("debug_ai", True))
    force_override_enabled = bool(
        st.session_state.get("ai_force_text_override")
    )

    def is_empty(v: object) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip() == ""
        if isinstance(v, (list, tuple, set, dict)):
            return len(v) == 0
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False

    updated = wide_df.copy(deep=True)

    ai_cells: dict[tuple[str, str], dict[str, object]] = {}
    ai_log: list[dict[str, object]] = []
    empty_mask_cache: dict[tuple[str, str], pd.Series] = {}
    skip_context: dict[str, object] = {}

    def _clear_empty_mask_cache_for(code_key: str) -> None:
        if not empty_mask_cache:
            return
        keys_to_remove = [
            cache_key for cache_key in empty_mask_cache if cache_key[0] == code_key
        ]
        for cache_key in keys_to_remove:
            empty_mask_cache.pop(cache_key, None)

    def _get_empty_mask_for(code_key: str, sku_value: str) -> pd.Series | None:
        cache_key = (code_key, sku_value)
        cached = empty_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        if code_key not in updated.columns:
            return None
        try:
            empties = updated[code_key].map(is_empty)
        except Exception:
            return None
        try:
            mask = (updated[sku_col].astype(str) == sku_value) & empties
        except Exception:
            return None
        empty_mask_cache[cache_key] = mask
        return mask

    def _shorten_value(value: object, limit: int = 120) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _preview_existing(mask: pd.Series, column: str) -> list:
        if column not in updated.columns:
            return []
        try:
            series = updated.loc[mask, column]
        except Exception:
            return []
        try:
            unique_values = pd.unique(series)
        except Exception:
            try:
                unique_values = series.unique()
            except Exception:
                return []
        preview: list = []
        for item in unique_values:
            preview.append(item)
            if len(preview) >= 3:
                break
        return preview

    def _mark_ai_filled(
        container: dict[tuple[str, str], dict[str, object]],
        *,
        row_idx: int | None,
        col_key: str,
        sku_value: str,
        value: object,
        log_entry: dict[str, object] | None = None,
    ) -> None:
        key = (sku_value, col_key)
        entry = container.setdefault(key, {"sku": sku_value, "code": col_key})
        if row_idx is not None:
            try:
                entry["row"] = int(row_idx)
            except Exception:
                pass
        entry["value"] = value
        if isinstance(log_entry, dict):
            for log_key, log_value in log_entry.items():
                if log_value in (None, [], {}):
                    continue
                entry[log_key] = log_value

    def _normalize_multiselect_list(value: object) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            iterable = value
        elif value in (None, ""):
            return []
        else:
            return _split_multiselect_input(value)

        normalized: list[str] = []
        for item in iterable:
            if item in (None, ""):
                continue
            if isinstance(item, float) and pd.isna(item):
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    sku_col = "sku" if "sku" in updated.columns else None
    if not sku_col:
        return updated, [], []

    sku_series = updated[sku_col].astype(str)

    for raw_sku, attrs in suggestions.items():
        if not isinstance(attrs, dict):
            continue
        sku_value = str(raw_sku)
        attrs_payload = dict(attrs)
        meta_payload = attrs_payload.pop("__meta__", None)
        regex_hint_map: Mapping[str, object] | dict[str, object] = {}
        if isinstance(meta_payload, Mapping):
            meta_entry = {"sku": sku_value}
            for key, value in meta_payload.items():
                if value is None:
                    continue
                meta_entry[key] = value
            regex_hints_candidate = meta_payload.get("regex_hints")
            if isinstance(regex_hints_candidate, Mapping):
                regex_hint_map = dict(regex_hints_candidate)
            ai_log.append(meta_entry)
        else:
            regex_hint_map = {}
        if isinstance(regex_hint_map, Mapping):
            for regex_code, regex_value in list(regex_hint_map.items()):
                if regex_code not in attrs_payload and not is_empty(regex_value):
                    attrs_payload[regex_code] = {
                        "value": regex_value,
                        "reason": "Regex pre-extract",
                    }
        row_mask = sku_series == sku_value
        if not row_mask.any():
            continue
        row_indices = updated.index[row_mask]
        row_context: dict[str, object] = {"sku": sku_value}
        first_row_idx = row_indices[0] if len(row_indices) else None
        if first_row_idx is not None and "name" in updated.columns:
            row_context["name"] = updated.at[first_row_idx, "name"]
        ai_suggested = attrs_payload if isinstance(attrs_payload, Mapping) else {}
        if DEBUG:
            try:
                suggested_len = len(ai_suggested)
            except Exception:
                suggested_len = 0
            print(
                f"[DEBUG AI APPLY] SKU={sku_value} → AI suggested {suggested_len} attrs"
            )
        for code, payload in ai_suggested.items():
            code_key = str(code)
            if code_key not in updated.columns:
                try:
                    updated[code_key] = pd.NA
                except Exception:
                    updated[code_key] = pd.Series(pd.NA, index=updated.index)
                _clear_empty_mask_cache_for(code_key)
                trace(
                    {"where": "apply_ai:new_col", "code": code_key, "sku": sku_value}
                )
            column_empty_mask = _get_empty_mask_for(code_key, sku_value)
            
            def _trace_skip(reason: str, extra: dict[str, object] | None = None) -> None:
                event = {
                    "where": "apply_ai:skip",
                    "sku": sku_value,
                    "code": code_key,
                    "reason": reason,
                    "existing_preview": _preview_existing(row_mask, code_key),
                }
                if skip_context:
                    event.update(skip_context)
                if isinstance(extra, Mapping):
                    event.update(extra)
                trace(event)
            suggestion_payload = payload if isinstance(payload, dict) else {"value": payload}
            ai_value = suggestion_payload.get("value")
            regex_candidate = None
            if isinstance(regex_hint_map, Mapping):
                regex_candidate = regex_hint_map.get(code_key)
            use_regex = False
            if not is_empty(regex_candidate):
                if is_empty(ai_value):
                    use_regex = True
                elif is_numeric_spec(code_key):
                    regex_text = str(regex_candidate).strip()
                    ai_text = str(ai_value).strip()
                    if regex_text and ai_text and regex_text != ai_text:
                        use_regex = True
            log_details: dict[str, object] | None = None
            if use_regex:
                ai_value = regex_candidate
                suggestion_payload = dict(suggestion_payload)
                suggestion_payload["value"] = ai_value
                if not suggestion_payload.get("reason"):
                    suggestion_payload["reason"] = "Regex pre-extract"
                log_details = {"used_regex": True, "value": ai_value}
            meta_candidate = meta_map.get(code_key) if isinstance(meta_map, dict) else {}
            if meta_candidate is None:
                meta = {"code": code_key, "frontend_input": "text"}
            elif isinstance(meta_candidate, Mapping):
                meta = meta_candidate
            else:
                meta = {"code": code_key, "frontend_input": "text"}
            allowed_for_set_info = suggestion_payload.get("allowed_for_set")
            allowed_for_set_map = (
                allowed_for_set_info
                if isinstance(allowed_for_set_info, Mapping)
                else {}
            )
            input_type_source = (
                meta.get("frontend_input")
                if isinstance(meta, Mapping)
                else "text"
            )
            if isinstance(meta, Mapping) and not input_type_source:
                input_type_source = meta.get("backend_type")
            input_type = str(input_type_source or "text").lower()
            if isinstance(allowed_for_set_info, Mapping):
                allowed_for_set_value = bool(allowed_for_set_info.get("value", True))
            elif allowed_for_set_info is None:
                allowed_for_set_value = True
            else:
                try:
                    allowed_for_set_value = bool(allowed_for_set_info)
                except Exception:
                    allowed_for_set_value = True
            meta_options = meta.get("options") if isinstance(meta, Mapping) else None
            options_count = len(meta_options) if isinstance(meta_options, Iterable) else None
            trace(
                {
                    "where": "apply_ai:meta",
                    "sku": sku_value,
                    "code": code_key,
                    "input_type": input_type,
                    "options_count": options_count,
                    "allowed_for_set": bool(allowed_for_set_value),
                }
            )
            skip_context.clear()
            skip_context.update(
                {
                    "suggested": _shorten_value(ai_value),
                    "meta_frontend_input": input_type,
                    "has_options": bool(options_count),
                    "allowed_for_set": bool(allowed_for_set_value),
                }
            )
            trace(
                {
                    "where": "apply_ai:plan",
                    "sku": sku_value,
                    "code": code_key,
                    "ai_raw": _shorten_value(ai_value),
                    "input_type": input_type,
                    "allowed_for_set": bool(allowed_for_set_value),
                    "force_text_override": force_override_enabled,
                }
            )
            skip_extra_base = {
                "allowed_for_set": bool(allowed_for_set_value),
            }
            if is_empty(ai_value):
                _trace_skip(
                    "empty-suggestion",
                    {
                        "ai_preview": None if ai_value is None else str(ai_value)[:120],
                        **skip_extra_base,
                    },
                )
                continue
            if DEBUG:
                options_count = (
                    len(meta.get("options", []))
                    if isinstance(meta, Mapping)
                    else "N/A"
                )
                allowed_debug = None
                if isinstance(allowed_for_set_map, Mapping):
                    allowed_debug = allowed_for_set_map.get(code_key)
                    if allowed_debug is None:
                        allowed_debug = allowed_for_set_map.get("value")
                print(
                    f"[DEBUG META] {code_key}: type={meta.get('frontend_input') if isinstance(meta, Mapping) else None} "
                    f"options={options_count} allowed_for_set={allowed_debug}"
                )
            allowed_multiselect_labels: dict[str, str] | None = None
            multiselect_values_to_labels: dict[str, str] | None = None
            allowed_select_labels: dict[str, str] | None = None
            select_values_to_labels: dict[str, str] | None = None
            if input_type == "multiselect" and isinstance(meta, dict):
                options = meta.get("options")
                if isinstance(options, list):
                    allowed_multiselect_labels = {}
                    multiselect_values_to_labels = {}
                    for option in options:
                        if not isinstance(option, dict):
                            continue
                        raw_label = option.get("label")
                        if not isinstance(raw_label, str):
                            continue
                        trimmed_label = raw_label.strip()
                        if not trimmed_label:
                            continue
                        allowed_multiselect_labels[trimmed_label] = trimmed_label
                        if "value" in option and option["value"] not in (None, ""):
                            value_text = str(option["value"]).strip()
                            if value_text:
                                multiselect_values_to_labels[value_text] = trimmed_label
            elif input_type == "select" and isinstance(meta, dict):
                options = meta.get("options")
                if isinstance(options, list):
                    allowed_select_labels = {}
                    select_values_to_labels = {}
                    for option in options:
                        if not isinstance(option, dict):
                            continue
                        raw_label = option.get("label")
                        canonical_label: str | None = None
                        if isinstance(raw_label, str):
                            trimmed_label = raw_label.strip()
                            if trimmed_label:
                                canonical_label = trimmed_label
                                allowed_select_labels.setdefault(trimmed_label, trimmed_label)
                                allowed_select_labels.setdefault(
                                    trimmed_label.casefold(), trimmed_label
                                )
                        raw_value = option.get("value")
                        if raw_value in (None, ""):
                            continue
                        value_text = str(raw_value).strip()
                        if not value_text:
                            continue
                        if canonical_label is None:
                            canonical_label = value_text
                        select_values_to_labels.setdefault(value_text, canonical_label)
                        select_values_to_labels.setdefault(
                            value_text.casefold(), canonical_label
                        )
                        if value_text.isdigit():
                            int_value = int(value_text)
                            select_values_to_labels.setdefault(str(int_value), canonical_label)
                            select_values_to_labels.setdefault(int_value, canonical_label)
            if input_type in {"multiselect", "select"}:
                options_list = meta.get("options") if isinstance(meta, Mapping) else None
                if not options_list:
                    _trace_skip("no-options", skip_extra_base)
                    ai_log.append(
                        {
                            "sku": sku_value,
                            "code": code_key,
                            "skip_reason": "no-options",
                            "suggested": _shorten_value(ai_value),
                        }
                    )
                    continue
            if input_type in {"multiselect", "select"} and not allowed_for_set_value:
                skip_reason = "not-applicable-for-set"
                if DEBUG:
                    print(
                        f"[DEBUG SKIP] {code_key} skipped — reason={skip_reason} "
                        f"existing={_preview_existing(row_mask, code_key)}"
                    )
                _trace_skip(skip_reason, skip_extra_base)
                ai_log.append(
                    {
                        "sku": sku_value,
                        "code": code_key,
                        "skip_reason": "not-applicable-for-set",
                        "suggested": ai_value,
                    }
                )
                continue

            coerced_value = _coerce_ai_value(ai_value, meta)
            trace(
                {
                    "where": "apply_ai:coerce",
                    "sku": sku_value,
                    "code": code_key,
                    "ai_raw_preview": str(ai_value)[:120],
                    "coerced_preview": str(coerced_value)[:120],
                }
            )
            if DEBUG:
                print(
                    f"[DEBUG VALUE] {code_key}: ai_raw={ai_value!r}, coerced={coerced_value!r}"
                )
            converted_value = coerced_value

            if input_type in TEXT_INPUT_TYPES:
                if converted_value in (None, ""):
                    skip_reason = "empty-after-coerce"
                    if DEBUG:
                        print(
                            f"[DEBUG SKIP] {code_key} skipped — reason={skip_reason} "
                            f"existing={_preview_existing(row_mask, code_key)}"
                        )
                    _trace_skip(skip_reason, skip_extra_base)
                    ai_log.append(
                        {
                            "sku": sku_value,
                            "code": code_key,
                            "skip_reason": "empty-after-coerce",
                            "suggested": _shorten_value(ai_value),
                        }
                    )
                    continue
                trimmed_value = str(converted_value).strip()
                if not trimmed_value:
                    skip_reason = "empty-after-coerce"
                    if DEBUG:
                        print(
                            f"[DEBUG SKIP] {code_key} skipped — reason={skip_reason} "
                            f"existing={_preview_existing(row_mask, code_key)}"
                        )
                    _trace_skip(skip_reason, skip_extra_base)
                    ai_log.append(
                        {
                            "sku": sku_value,
                            "code": code_key,
                            "skip_reason": "empty-after-coerce",
                            "suggested": _shorten_value(ai_value),
                        }
                    )
                    continue
                if code_key not in updated.columns:
                    updated[code_key] = pd.NA
                try:
                    empties = updated[code_key].map(is_empty)
                except Exception:
                    empties = pd.Series(False, index=updated.index)
                mask = (sku_series == sku_value) & empties
                force_override_applied = False
                soft_override_applied = False
                if not mask.any():
                    existing_series = updated.loc[row_mask, code_key]
                    existing_value = None
                    if isinstance(existing_series, pd.Series) and not existing_series.empty:
                        existing_value = existing_series.iloc[0]
                    should_soft_override = should_force_override(
                        code_key,
                        existing_value,
                        trimmed_value,
                        row_context,
                    )
                    if force_override_enabled or should_soft_override:
                        mask = sku_series == sku_value
                        force_override_applied = True
                        soft_override_applied = not force_override_enabled and should_soft_override
                        _force_mark(sku_value, code_key)
                    else:
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason=non-empty "
                                f"existing={_preview_existing(row_mask, code_key)}"
                            )
                        _trace_skip(
                            "non-empty",
                            {"existing_value": existing_value, **skip_extra_base},
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "skip_reason": "non-empty",
                                "existing_value": existing_value,
                            }
                        )
                        continue
                if DEBUG:
                    print(
                        f"[DEBUG APPLY] Writing {code_key}={trimmed_value!r} for {int(mask.sum())} rows"
                    )
                indices_to_update = mask[mask].index.tolist()
                row_positions: list[int] = []
                for idx in indices_to_update:
                    row_position_arr = updated.index.get_indexer([idx])
                    row_position: int | None = None
                    if len(row_position_arr) == 1 and row_position_arr[0] >= 0:
                        row_position = int(row_position_arr[0])
                        row_positions.append(row_position)
                    _mark_ai_filled(
                        ai_cells,
                        row_idx=row_position,
                        col_key=code_key,
                        sku_value=sku_value,
                        value=trimmed_value,
                        log_entry=log_details,
                    )
                updated.loc[mask, code_key] = trimmed_value
                _clear_empty_mask_cache_for(code_key)
                trace(
                    {
                        "where": "apply_ai:apply",
                        "sku": sku_value,
                        "code": code_key,
                        "value_preview": str(trimmed_value)[:200],
                        "row_idx": row_positions[0] if row_positions else None,
                        "rows_affected": len(indices_to_update),
                        "force_override": force_override_applied,
                        "soft_override": soft_override_applied,
                    }
                )
                continue

            for idx in row_indices:
                existing_raw = updated.at[idx, code_key]
                is_cell_empty = True
                if column_empty_mask is not None:
                    try:
                        is_cell_empty = bool(column_empty_mask.at[idx])
                    except Exception:
                        is_cell_empty = is_empty(existing_raw)
                else:
                    is_cell_empty = is_empty(existing_raw)
                row_position_arr = updated.index.get_indexer([idx])
                row_position: int | None = None
                if len(row_position_arr) == 1 and row_position_arr[0] >= 0:
                    row_position = int(row_position_arr[0])

                if code_key == "categories":
                    meta_categories = meta_map.get("categories") or {}
                    values_to_labels: dict[str, str] = {}
                    labels_to_values_map: dict[str, object] = {}
                    if isinstance(meta_categories, Mapping):
                        raw_v2l = meta_categories.get("values_to_labels")
                        if isinstance(raw_v2l, Mapping):
                            values_to_labels = {
                                str(key): str(value)
                                for key, value in raw_v2l.items()
                                if key not in (None, "") and value not in (None, "")
                            }
                        raw_l2v = meta_categories.get("labels_to_values")
                        if isinstance(raw_l2v, Mapping):
                            labels_to_values_map = dict(raw_l2v)

                    ai_raw = (
                        suggestion_payload.get("categories")
                        or suggestion_payload.get("category")
                        or ai_value
                    )
                    if isinstance(ai_raw, dict):
                        candidate = (
                            ai_raw.get("value")
                            or ai_raw.get("values")
                            or list(ai_raw.values())
                        )
                        ai_raw = candidate
                    if isinstance(ai_raw, (str, int, float)):
                        ai_raw = [ai_raw]
                    normalized_ai: list[str] = []
                    for item in ai_raw or []:
                        text = str(item).strip()
                        if not text:
                            continue
                        if text.isdigit():
                            normalized_ai.append(str(int(text)))
                        else:
                            normalized_ai.append(text)

                    found_ids: list[str] = []
                    ignored_tokens: list[str] = []

                    def _append_ignored(token: str) -> None:
                        candidate = str(token).strip()
                        if not candidate:
                            return
                        if candidate not in ignored_tokens:
                            ignored_tokens.append(candidate)

                    canonical_seen: set[str] = set()
                    for token in normalized_ai:
                        if token.isdigit():
                            if token in values_to_labels:
                                if token not in found_ids:
                                    found_ids.append(token)
                            else:
                                _append_ignored(token)
                            continue

                        canonical_label = normalize_category_label(token, meta_categories)
                        if canonical_label:
                            if canonical_label in canonical_seen:
                                continue
                            canonical_seen.add(canonical_label)
                            resolved_value = labels_to_values_map.get(canonical_label)
                            if resolved_value is None:
                                _append_ignored(token)
                                continue
                            resolved_str = str(resolved_value)
                            if resolved_str not in found_ids:
                                found_ids.append(resolved_str)
                        else:
                            _append_ignored(token)

                    unmatched_labels = _ensure_list_of_strings(
                        suggestion_payload.get("unmatched_labels")
                    )
                    for item in unmatched_labels:
                        _append_ignored(item)

                    normalized_added = _ensure_list_of_strings(
                        suggestion_payload.get("normalized_categories_added")
                    )

                    existing_raw_value = existing_raw if code_key in updated.columns else None
                    if (
                        existing_raw_value is None
                        or (isinstance(existing_raw_value, float) and pd.isna(existing_raw_value))
                        or existing_raw_value == ""
                    ):
                        existing_ids: list[str] = []
                    elif isinstance(existing_raw_value, (list, tuple, set)):
                        existing_ids = [str(item) for item in existing_raw_value]
                    elif isinstance(existing_raw_value, str):
                        parts = [
                            part.strip()
                            for part in re.split(r"[;,]\s*", existing_raw_value)
                            if part.strip() != ""
                        ]
                        existing_ids = [part for part in parts if part.isdigit()]
                    else:
                        existing_ids = []

                    combined_ids = existing_ids + [
                        cid for cid in found_ids if cid not in existing_ids
                    ]

                    trace(
                        {
                            "where": "apply_ai:categories_map",
                            "sku": sku_value,
                            "suggested": normalized_ai,
                            "found_ids": found_ids,
                            "existing_ids": existing_ids,
                            "combined_after": combined_ids,
                            "ignored_tokens": ignored_tokens,
                        }
                    )

                    if combined_ids != existing_ids:
                        if DEBUG:
                            print(
                                f"[DEBUG APPLY] Writing {code_key}={combined_ids!r} for {row_mask.sum()} rows"
                            )
                        trace(
                            {
                                "where": "apply_ai:apply",
                                "sku": sku_value,
                                "code": code_key,
                                "value_preview": str(combined_ids)[:200],
                                "row_idx": row_position,
                                "rows_affected": 1,
                                "force_override": False,
                            }
                        )
                        updated.at[idx, code_key] = combined_ids
                        if column_empty_mask is not None:
                            column_empty_mask.at[idx] = is_empty(combined_ids)
                        log_entry = {
                            "applied": found_ids,
                            "existing_before": existing_ids,
                            "combined_after": combined_ids,
                        }
                        if normalized_added:
                            log_entry["normalized_categories_added"] = normalized_added
                        _mark_ai_filled(
                            ai_cells,
                            row_idx=row_position,
                            col_key=code_key,
                            sku_value=sku_value,
                            value=combined_ids,
                            log_entry=log_entry,
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": "categories",
                                "applied": found_ids,
                                "existing_before": existing_ids,
                                "combined_after": combined_ids,
                                "normalized_categories_added": normalized_added,
                            }
                        )
                    else:
                        skip_reason = (
                            "non-empty" if existing_ids else "not-applicable-for-set"
                        )
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason={skip_reason} "
                                f"existing={existing_ids[:3]}"
                            )
                        _trace_skip(
                            skip_reason,
                            {
                                "found_ids": found_ids,
                                "ignored_tokens": ignored_tokens,
                                "suggested": normalized_ai,
                                **skip_extra_base,
                            },
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": "categories",
                                "skip_reason": (
                                    "non-empty"
                                    if existing_ids
                                    else "not-applicable-for-set"
                                ),
                                "suggested": normalized_ai or ai_raw,
                                "existing_before": existing_ids,
                            }
                        )

                    if ignored_tokens:
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": "categories",
                                "ignored_by_ai_filter": ignored_tokens,
                            }
                        )

                    continue

                if input_type == "multiselect":
                    existing_list = _normalize_multiselect_list(existing_raw)
                    ai_values = _normalize_multiselect_list(converted_value)
                    if (
                        allowed_multiselect_labels is not None
                        or multiselect_values_to_labels is not None
                    ):
                        filtered_ai_values: list[str] = []
                        for item in ai_values:
                            candidate = item.strip()
                            if not candidate:
                                continue
                            if (
                                allowed_multiselect_labels is not None
                                and candidate in allowed_multiselect_labels
                            ):
                                filtered_ai_values.append(
                                    allowed_multiselect_labels[candidate]
                                )
                                continue
                            if (
                                multiselect_values_to_labels is not None
                                and candidate in multiselect_values_to_labels
                            ):
                                filtered_ai_values.append(
                                    multiselect_values_to_labels[candidate]
                            )
                        ai_values = filtered_ai_values
                    if not ai_values:
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason=not-applicable-for-set "
                                f"existing={_preview_existing(row_mask, code_key)}"
                            )
                        _trace_skip(
                            "not-applicable-for-set",
                            {
                                "suggested": ai_value,
                                "ignored_tokens": ignored_tokens,
                                **skip_extra_base,
                            },
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "skip_reason": "not-applicable-for-set",
                                "suggested": ai_value,
                            }
                        )
                        continue
                    existing_normalized = [item.strip() for item in existing_list]
                    seen = set(existing_normalized)
                    new_items: list[str] = []
                    skipped_duplicates: list[str] = []
                    for item in ai_values:
                        normalized_item = item.strip()
                        if normalized_item in seen:
                            skipped_duplicates.append(item)
                            continue
                        seen.add(normalized_item)
                        new_items.append(item)
                    if new_items:
                        combined = existing_list + new_items
                        if DEBUG:
                            print(
                                f"[DEBUG APPLY] Writing {code_key}={combined!r} for {row_mask.sum()} rows"
                            )
                        trace(
                            {
                                "where": "apply_ai:apply",
                                "sku": sku_value,
                                "code": code_key,
                                "value_preview": str(combined)[:200],
                                "row_idx": row_position,
                                "rows_affected": 1,
                                "force_override": False,
                            }
                        )
                        updated.at[idx, code_key] = combined
                        if column_empty_mask is not None:
                            column_empty_mask.at[idx] = is_empty(combined)
                        log_entry = {
                            "applied": new_items,
                            "existing_before": existing_list,
                            "combined_after": combined,
                        }
                        if skipped_duplicates:
                            log_entry["skipped_duplicates"] = skipped_duplicates
                        _mark_ai_filled(
                            ai_cells,
                            row_idx=row_position,
                            col_key=code_key,
                            sku_value=sku_value,
                            value=combined,
                            log_entry=log_entry,
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "applied": new_items,
                                "existing_before": existing_list,
                                "combined_after": combined,
                                **(
                                    {"skipped_duplicates": skipped_duplicates}
                                    if skipped_duplicates
                                    else {}
                                ),
                            }
                        )
                    elif skipped_duplicates:
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason=duplicates "
                                f"existing={existing_list[:3]}"
                            )
                        _trace_skip(
                            "duplicates",
                            {
                                "suggested": ai_value,
                                "skipped_duplicates": skipped_duplicates,
                                **skip_extra_base,
                            },
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "existing_before": existing_list,
                                "skipped_duplicates": skipped_duplicates,
                            }
                        )
                    else:
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason=not-applicable-for-set "
                                f"existing={existing_list[:3]}"
                            )
                        _trace_skip(
                            "not-applicable-for-set",
                            {
                                "suggested": ai_value,
                                "ignored_tokens": ignored_tokens,
                                **skip_extra_base,
                            },
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "skip_reason": "not-applicable-for-set",
                                "suggested": ai_value,
                                **(
                                    {"ignored_tokens": ignored_tokens}
                                    if ignored_tokens
                                    else {}
                                ),
                            }
                        )
                    continue

                if input_type == "select":
                    candidate_raw = converted_value
                    if isinstance(candidate_raw, str):
                        candidate_text = candidate_raw.strip()
                    elif candidate_raw in (None, ""):
                        candidate_text = ""
                    else:
                        candidate_text = str(candidate_raw).strip()

                    if not candidate_text:
                        if DEBUG:
                            print(
                                f"[DEBUG SKIP] {code_key} skipped — reason=not-applicable-for-set "
                                f"existing={_preview_existing(row_mask, code_key)}"
                            )
                        _trace_skip(
                            "not-applicable-for-set",
                            {"suggested": candidate_raw, **skip_extra_base},
                        )
                        ai_log.append(
                            {
                                "sku": sku_value,
                                "code": code_key,
                                "skip_reason": "not-applicable-for-set",
                                "suggested": candidate_raw,
                            }
                        )
                        continue

                    canonical_label: str | None = None
                    if allowed_select_labels:
                        canonical_label = allowed_select_labels.get(candidate_text)
                        if canonical_label is None:
                            canonical_label = allowed_select_labels.get(
                                candidate_text.casefold()
                            )
                    if canonical_label is None and select_values_to_labels:
                        canonical_label = select_values_to_labels.get(candidate_text)
                        if canonical_label is None:
                            canonical_label = select_values_to_labels.get(
                                candidate_text.casefold()
                            )

                    if canonical_label is None:
                        canonical_label = candidate_text

                    converted_value = canonical_label

                if not is_empty(existing_raw):
                    if DEBUG:
                        print(
                            f"[DEBUG SKIP] {code_key} skipped — reason=non-empty "
                            f"existing={_preview_existing(row_mask, code_key)}"
                        )
                    _trace_skip(
                        "non-empty",
                        {"existing_value": existing_raw, **skip_extra_base},
                    )
                    ai_log.append(
                        {
                            "sku": sku_value,
                            "code": code_key,
                            "skip_reason": "non-empty",
                            "existing_value": existing_raw,
                        }
                    )
                    continue

                converted_value = normalize_units(code_key, converted_value)

                if DEBUG:
                    print(
                        f"[DEBUG APPLY] Writing {code_key}={converted_value!r} for {row_mask.sum()} rows"
                    )
                trace(
                    {
                        "where": "apply_ai:apply",
                        "sku": sku_value,
                        "code": code_key,
                        "value_preview": str(converted_value)[:200],
                    }
                )
                updated.at[idx, code_key] = converted_value
                if column_empty_mask is not None:
                    column_empty_mask.at[idx] = is_empty(converted_value)
                _mark_ai_filled(
                    ai_cells,
                    row_idx=row_position,
                    col_key=code_key,
                    sku_value=sku_value,
                    value=converted_value,
                    log_entry=log_details,
                )

    return updated, list(ai_cells.values()), ai_log

def _render_ai_highlight(
    df_view: pd.DataFrame,
    column_order: list[str],
    ai_cells: Iterable | None,
) -> None:
    if not isinstance(df_view, pd.DataFrame) or df_view.empty:
        return
    if "sku" not in df_view.columns:
        return

    df_reset = df_view.reset_index(drop=True)
    row_positions = {
        str(df_reset.at[idx, "sku"]): int(idx) for idx in df_reset.index
    }
    column_positions = {
        col: int(idx)
        for idx, col in enumerate(column_order)
        if col in df_view.columns
    }
    sku_index = column_positions.get("sku")
    if sku_index is None:
        return

    targets: list[dict[str, object]] = []
    seen_targets: set[tuple[str, str]] = set()

    def _iter_cells():
        if isinstance(ai_cells, dict):
            yield from ai_cells.values()
            return
        for item in ai_cells or []:
            yield item

    for cell in _iter_cells():
        row_idx_from_cell: int | None = None
        if isinstance(cell, dict):
            raw_sku = cell.get("sku")
            code_value = cell.get("code")
            raw_row = cell.get("row")
            if isinstance(raw_row, (int, float)) and not pd.isna(raw_row):
                try:
                    row_idx_from_cell = int(raw_row)
                except Exception:
                    row_idx_from_cell = None
        elif isinstance(cell, (list, tuple)) and len(cell) >= 2:
            raw_sku, code_value = cell[0], cell[1]
        else:
            continue
        sku_value = str(raw_sku)
        code_key = str(code_value)
        row_idx = row_idx_from_cell
        if row_idx is None:
            row_idx = row_positions.get(sku_value)
        col_idx = column_positions.get(code_key)
        if row_idx is None or col_idx is None:
            continue
        key = (sku_value, code_key)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        targets.append({"row": row_idx, "col": col_idx, "sku": sku_value})

    payload = {"targets": targets, "sku_index": sku_index}
    _inject_ai_highlight_script(payload)


def _inject_ai_highlight_script(payload: dict) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)

    st.markdown(
        """
<style id="ai-highlight-style">
  [data-ai-filled="true"], [data-ai-filled="true"] * {
    background-color: #fff7ae !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<script>
(function() {{
  const cfg = {payload_json};  // {{ targets: [{{row, col, sku}}...], sku_index }}
  const MAX_TRIES = 80;

  function findTableRoot() {{
    // стараемся найти самый свежий stDataEditor (последний)
    const editors = Array.from(document.querySelectorAll('div[data-testid="stDataEditor"]'));
    return editors.length ? editors[editors.length - 1] : null;
  }}

  function applyOnce() {{
    const root = findTableRoot();
    if (!root) return false;

    // Пытаемся подсветить по row/col индексам
    let changed = false;
    (cfg.targets || []).forEach(t => {{
      // сперва ищем ячейку по data-row-index/data-col-index (если в текущей версии они есть)
      let cell = root.querySelector(`[data-row-index="${{t.row}}"][data-col-index="${{t.col}}"]`);
      if (!cell) {{
        // fallback: ищем по тексту SKU в ряду (если передан sku_index)
        try {{
          const rows = root.querySelectorAll('tbody tr');
          const skuIdx = cfg.sku_index;
          if (rows && rows.length && Number.isInteger(skuIdx)) {{
            const tr = rows[t.row];
            if (tr) {{
              const cells = tr.querySelectorAll('td');
              if (cells && cells.length > Math.max(skuIdx, t.col)) {{
                const skuCell = cells[skuIdx];
                if (skuCell && (skuCell.innerText || '').trim() === (t.sku || '').trim()) {{
                  cell = cells[t.col];
                }}
              }}
            }}
          }}
        }} catch (e) {{}}
      }}
      if (cell) {{
        cell.setAttribute('data-ai-filled', 'true');
        changed = true;
      }}
    }});
    return changed;
  }}

  function scheduleApply(tries) {{
    if (applyOnce()) return;
    if (tries >= MAX_TRIES) return;
    setTimeout(() => scheduleApply(tries + 1), 100);
  }}

  // первая попытка сразу
  scheduleApply(0);

  // пересветка при изменениях DOM (скролл/перерисовка)
  const obs = new MutationObserver(() => scheduleApply(0));
  obs.observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
""",
        unsafe_allow_html=True,
    )

def _collect_ai_suggestions_log(
    ai_suggestions: dict | None,
    ai_cells: dict | None,
    applied_logs: dict | None = None,
    extra_log: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(ai_suggestions, dict):
        ai_suggestions = {}

    valid_cells: set[tuple[str, str, str]] = set()
    if isinstance(ai_cells, dict):
        for set_id, cells in ai_cells.items():
            if not isinstance(cells, Iterable):
                continue
            for cell in cells:
                if not isinstance(cell, Iterable):
                    continue
                try:
                    sku_value, code_value = cell
                except ValueError:
                    continue
                valid_cells.add((str(set_id), str(sku_value), str(code_value)))

    suggestions_log: dict[str, dict[str, object]] = {}
    for set_id, per_sku in ai_suggestions.items():
        if not isinstance(per_sku, dict):
            continue
        for raw_sku, attrs in per_sku.items():
            sku_key = str(raw_sku)
            if valid_cells and all(
                (str(set_id), sku_key, str(code)) not in valid_cells
                for code in (attrs or {}).keys()
            ):
                continue
            if not isinstance(attrs, dict):
                continue
            for code, payload in attrs.items():
                normalized_key = str(code)
                if valid_cells and (
                    str(set_id), sku_key, normalized_key
                ) not in valid_cells:
                    continue
                reason = None
                if isinstance(payload, Mapping):
                    value = payload.get("value")
                    reason = payload.get("reason")
                else:
                    value = payload
                if value in (None, ""):
                    continue
                if isinstance(value, (list, tuple, set)) and not value:
                    continue
                entry_payload: dict[str, object] = {"value": value}
                if reason:
                    entry_payload["reason"] = reason
                suggestions_log.setdefault(sku_key, {})[normalized_key] = entry_payload

    if suggestions_log:
        suggestions_log = dict(sorted(suggestions_log.items(), key=lambda item: item[0]))

    applied_entries: list[dict[str, object]] = []
    if isinstance(applied_logs, dict):
        for entries in applied_logs.values():
            if not isinstance(entries, Iterable):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    applied_entries.append(entry)

    result: dict[str, object] = {}
    if suggestions_log:
        result["suggestions"] = suggestions_log
    if applied_entries:
        result["applied"] = applied_entries
    if isinstance(extra_log, Mapping):
        errors = extra_log.get("errors")
        if isinstance(errors, list) and errors:
            result["errors"] = errors

    return result


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
    labels: list[str] = []
    for opt in meta.get("options") or []:
        if not isinstance(opt, dict):
            continue
        label = opt.get("label")
        if label is None:
            continue
        labels.append(str(label))

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
    wide_meta: dict[str, dict],
    sample_df: pd.DataFrame | None = None,
    set_id: int | str | None = None,
    set_name: str | None = None,
):
    cfg = {
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "name": st.column_config.TextColumn("Name", disabled=False),
    }

    for code, original_meta in list(wide_meta.items()):
        meta = original_meta or {}
        cfg[code] = colcfg_from_meta(code, meta)

    set_name_text = str(set_name or "")
    set_name_norm = set_name_text.strip().lower()
    set_id_value: int | None = None
    if set_id not in (None, ""):
        try:
            set_id_value = int(set_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            set_id_value = None

    is_electric = False
    if set_id_value is not None and set_id_value in ELECTRIC_SET_IDS:
        is_electric = True
    elif set_name_norm.startswith("electric"):
        is_electric = True

    if is_electric and "guitarstylemultiplechoice" in cfg:
        guitar_cfg = cfg["guitarstylemultiplechoice"]
        if hasattr(guitar_cfg, "label"):
            guitar_cfg.label = "Guitar style"
        elif hasattr(guitar_cfg, "_label"):
            guitar_cfg._label = "Guitar style"

    cfg[GENERATE_DESCRIPTION_COLUMN] = st.column_config.CheckboxColumn(
        "Generate description", default=True
    )

    return cfg


def _unique_preserve_str(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def safe_label(entry: Mapping[str, object] | None) -> str:
    if not isinstance(entry, Mapping):
        return ""
    label = entry.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    raw_value = entry.get("raw_value")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    if raw_value not in (None, ""):
        return str(raw_value)
    return ""


def _extract_attr_text(attr_rows: Mapping[str, dict], code: str) -> str:
    entry = attr_rows.get(code) if isinstance(attr_rows, Mapping) else None
    if not isinstance(entry, Mapping):
        return ""
    label = entry.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    raw_value = entry.get("raw_value")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    if raw_value not in (None, "", []):
        return str(raw_value)
    return ""


def _match_category_hint(
    candidate: str,
    labels_to_values: Mapping[str, object],
    values_to_labels: Mapping[str, str],
) -> str | None:
    if not isinstance(candidate, str):
        return None
    normalized = _norm_label(candidate)
    if not normalized:
        return None
    value_key = None
    if isinstance(labels_to_values, Mapping):
        for lookup in (candidate, candidate.casefold(), normalized):
            if lookup in labels_to_values:
                value_key = labels_to_values[lookup]
                break
    if value_key is None:
        return None
    if isinstance(values_to_labels, Mapping):
        resolved = values_to_labels.get(str(value_key)) or values_to_labels.get(value_key)
        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()
    return str(candidate).strip() or None


def _extract_model_series_from_name(name: object) -> list[str]:
    if not isinstance(name, str):
        return []
    pattern = re.compile(r"\b(series|model)\b[:#-]?\s*([A-Za-z0-9][A-Za-z0-9 '\-/&]+)", re.I)
    candidates: list[str] = []
    for match in pattern.finditer(name):
        candidate = match.group(2).strip()
        candidate = re.split(r"[()\[\],]| with | featuring | includes ", candidate)[0]
        candidate = candidate.strip("-:/ ")
        if candidate:
            candidates.append(candidate)
    return _unique_preserve_str(candidates)


def _build_ai_hints(
    attr_rows: Mapping[str, dict],
    set_name: str,
    product: Mapping[str, object],
    labels_to_values: Mapping[str, object],
    values_to_labels: Mapping[str, str],
    meta_cache: AttributeMetaCache | Mapping[str, object] | None,
) -> dict[str, object]:
    hints: dict[str, object] = {}
    set_name_clean = str(set_name).strip() if isinstance(set_name, str) else ""
    if set_name_clean:
        hints["attribute_set_name"] = set_name_clean

    brand_label = safe_label(attr_rows.get("brand"))
    if brand_label and isinstance(labels_to_values, Mapping):
        brand_lookup = brand_label.casefold()
        for raw_label in labels_to_values.keys():
            key_text = str(raw_label or "").strip()
            if not key_text:
                continue
            if key_text.casefold() == brand_lookup:
                hints["brand_hint"] = brand_label
                break

    if "brand_hint" not in hints:
        brand_meta: Mapping[str, object] | None = None
        if isinstance(meta_cache, AttributeMetaCache):
            brand_meta = meta_cache.get("brand")
        elif isinstance(meta_cache, Mapping):
            brand_meta = meta_cache.get("brand")
        brand_options = []
        if isinstance(brand_meta, Mapping):
            options = brand_meta.get("options")
            if isinstance(options, Sequence):
                brand_options = [
                    option
                    for option in options
                    if isinstance(option, Mapping)
                ]
        guessed_brand = brand_from_title(
            product.get("name") if isinstance(product, Mapping) else None,
            brand_options,
        )
        if guessed_brand:
            hints["brand_hint"] = guessed_brand
            hints["brand_hint_reason"] = "Brand found in product name"

    set_hint = _match_category_hint(set_name_clean, labels_to_values, values_to_labels)
    if set_hint:
        hints["set_hint"] = set_hint

    candidates: list[str] = []
    for key in ("model", "series"):
        text = _extract_attr_text(attr_rows, key)
        if text:
            candidates.append(text)
    candidates.extend(
        _extract_model_series_from_name(product.get("name"))
        if isinstance(product, Mapping)
        else []
    )
    candidates = _unique_preserve_str(candidates)
    if candidates:
        hints["model_series_candidates"] = candidates

    style_texts = list(candidates)
    product_name = product.get("name") if isinstance(product, Mapping) else None
    if isinstance(product_name, str) and product_name.strip():
        style_texts.append(product_name)
    style_hints = derive_styles_from_texts(style_texts)
    if style_hints:
        hints["style_hint"] = style_hints

    return hints


def build_attributes_df(
    df_changed: pd.DataFrame,
    session,
    api_base: str,
    attribute_sets: dict,
    attr_sets_map: dict,
    meta_cache: AttributeMetaCache,
    *,
    ai_api_key: str | None = None,
    ai_model: str | None = None,
    progress_callback=None,  # 👈 new parameter to update progress
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

    category_option_items: list[dict[str, str]] = []
    seen_category_ids: set[str] = set()
    for raw_value, raw_label in cat_values_to_labels.items():
        value_str = str(raw_value)
        if value_str in seen_category_ids:
            continue
        label_text = str(raw_label or "").strip()
        if not label_text:
            continue
        seen_category_ids.add(value_str)
        category_option_items.append({"label": label_text, "value": value_str})

    category_option_items.sort(key=lambda item: item["label"].casefold())
    cats_meta["options"] = [{"label": "", "value": ""}] + category_option_items
    normalized_values_to_labels: dict[str, str] = {}
    for raw_value, raw_label in cat_values_to_labels.items():
        label_text = str(raw_label or "").strip()
        if not label_text:
            continue
        normalized_values_to_labels[str(raw_value)] = label_text
    cat_values_to_labels = normalized_values_to_labels
    cats_meta["values_to_labels"] = cat_values_to_labels
    labels_to_values_map = cats_meta.get("labels_to_values") or {}
    if not isinstance(labels_to_values_map, dict):
        labels_to_values_map = {}
    if not labels_to_values_map:
        stored_meta = st.session_state.get("categories_meta")
        if isinstance(stored_meta, dict):
            alt_map = stored_meta.get("labels_to_values")
            if isinstance(alt_map, dict):
                labels_to_values_map = alt_map
    if isinstance(meta_cache, AttributeMetaCache):
        try:
            meta_cache.set_static("categories", cats_meta)
        except Exception:
            pass
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

    ai_enabled = bool(ai_api_key)
    ai_requests: list[
        tuple[int, str, dict, pd.DataFrame, list[str], dict[str, object]]
    ] = []
    ai_results: dict[int, dict[str, dict[str, dict[str, object]]]] = defaultdict(dict)
    ai_errors: list[str] = []
    ai_log_data: dict[str, list[dict[str, object]]] = {"errors": []}
    resolved_model = resolve_model_name(ai_model, default=DEFAULT_OPENAI_MODEL)
    if not resolved_model:
        resolved_model = st.session_state.get("ai_model_resolved")
    ai_model_name = resolved_model or DEFAULT_OPENAI_MODEL
    ai_conv = get_ai_conversation(ai_model_name) if ai_enabled else None

    total_rows = len(df_changed.index)
    processed = 0

    for _, row in df_changed.iterrows():
        processed += 1
        sku_value = str(row.get("sku", "")).strip()
        if not sku_value:
            continue

        # 👇 update outer progress bar per SKU
        if callable(progress_callback):
            progress_callback(
                processed,
                max(total_rows, 1),
                f"🔄 {sku_value}: loading attributes…",
            )

        name_value = row.get("name")
        attr_set_value = row.get("attribute set")
        try:
            # remove nested spinner to avoid noisy UX
            product = get_product_by_sku(session, api_base, sku_value)
        except Exception as exc:  # pragma: no cover - UI interaction
            st.warning(f"{sku_value}: failed to fetch attributes ({exc})")
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
        allowed.add("categories")

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

        name_text = str(product.get("name", "")).lower()
        curr_orient = (attr_rows.get("orientation", {}) or {}).get("label") or (
            (attr_rows.get("orientation", {}) or {}).get("raw_value")
        )
        if _is_blank_value(curr_orient) or str(curr_orient).strip().lower() in {
            "right",
            "right-handed",
        }:
            if any(k in name_text for k in ["left", "left-handed", "lefty", "lh "]):
                attr_rows["orientation"] = {"label": "Left", "raw_value": "Left"}
                _force_mark(sku_value, "orientation")

        def _infer_condition_from_sku(sku: str) -> str | None:
            s = (sku or "").upper()
            if "ART" in s:
                return "New"
            if "ARC" in s or "ARO" in s:
                return "Pre-owned"
            if "ARB" in s:
                return "B-Stock"
            return None

        curr_cond = (attr_rows.get("condition", {}) or {}).get("label") or (
            (attr_rows.get("condition", {}) or {}).get("raw_value")
        )
        if _is_blank_value(curr_cond):
            inferred = _infer_condition_from_sku(sku_value)
            if inferred:
                attr_rows["condition"] = {"label": inferred, "raw_value": inferred}
                _force_mark(sku_value, "condition")

        hints_payload: dict[str, object] = {}
        if ai_enabled:
            set_name_label = set_names.get(attr_set_id_int, "")
            hints_payload = _build_ai_hints(
                attr_rows,
                set_name_label,
                product,
                labels_to_values_map,
                cat_values_to_labels,
                meta_cache,
            )

        force_codes = {"categories"}
        bool_codes = {
            c
            for c in editor_codes
            if (meta_cache.get(c) or {}).get("frontend_input") == "boolean"
        }
        missing_codes = {
            c
            for c in editor_codes
            if c not in {"sku", "name", "attribute set", "price"}
            and _is_blank_value(
                (attr_rows.get(c, {}) or {}).get("label")
                or (attr_rows.get(c, {}) or {}).get("raw_value")
            )
        }
        ai_codes = sorted(force_codes | bool_codes | missing_codes)

        needs_ai = False
        if ai_enabled:
            for code in ai_codes:
                values = attr_rows.get(code, {})
                current_value = values.get("label") or values.get("raw_value")
                if _is_blank_value(current_value):
                    needs_ai = True
                    break
        if ai_enabled and needs_ai:
            ai_requests.append(
                (
                    attr_set_id_int,
                    sku_value,
                    product,
                    df_full.copy(deep=True),
                    list(ai_codes),
                    hints_payload,
                )
            )

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
                "value": categories_ids,
            }
        )

    if not rows_by_set:
        return {}, {}, {}, meta_cache, {}, {}, {}, [], ai_log_data

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

    if ai_enabled and ai_requests:
        # Ограничиваем кол-во потоков, чтобы не убить ни OpenAI, ни Magento
        max_workers = min(4, len(ai_requests))

        def _run_ai_task(args):
            (
                set_id,
                sku_value,
                product_data,
                df_full,
                editor_codes,
                hints_payload,
            ) = args
            # Важно: здесь только тяжёлый вызов infer_missing,
            # никакого Streamlit UI внутри потока.
            ai_df = infer_missing(
                product_data,
                df_full,
                session,
                api_base,
                editor_codes,
                ai_api_key,
                conv=ai_conv,
                hints=hints_payload,
                model=ai_model_name,
            )
            return set_id, sku_value, ai_df, hints_payload

        with st.spinner("🤖 Generating AI suggestions…"):
            # Параллельно запускаем infer_missing по всем SKU
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_args = {
                    executor.submit(_run_ai_task, args): args
                    for args in ai_requests
                }

                total = len(future_to_args)
                done_count = 0

                for future in as_completed(future_to_args):
                    (
                        set_id,
                        sku_value,
                        product_data,
                        df_full,
                        editor_codes,
                        hints_payload,
                    ) = future_to_args[future]

                    try:
                        set_id, sku_value, ai_df, hints_payload = future.result()
                    except Exception as exc:  # pragma: no cover - network error handling
                        raw_preview_source = (
                            getattr(exc, "raw_response", None)
                            or getattr(exc, "raw_response_retry", None)
                            or getattr(exc, "raw_response_sanitized", None)
                            or ""
                        )
                        try:
                            raw_preview_source = str(raw_preview_source)[:1000]
                        except Exception:
                            raw_preview_source = ""
                        trace(
                            {
                                "where": "ai:error",
                                "sku": sku_value,
                                "set_id": set_id,
                                "err": str(exc),
                                "raw_preview": raw_preview_source,
                                "tb": traceback.format_exc()[-600:],
                            }
                        )
                        if "json" in str(exc).lower():
                            ai_log_data.setdefault("errors", []).append(
                                {
                                    "sku": sku_value,
                                    "reason": "json-parse-failed",
                                    "model": ai_model_name,
                                }
                            )
                        error_message = f"AI suggestion failed for {sku_value}: {exc}"
                        st.warning(error_message)
                        ai_errors.append(error_message)
                        continue

                    # Лёгкая часть — уже в основном потоке
                    if isinstance(ai_df, pd.DataFrame):
                        preview_df = ai_df.head(50)
                        if {"code", "value"} <= set(preview_df.columns):
                            suggested_records = preview_df[["code", "value"]].to_dict(
                                orient="records"
                            )
                        else:
                            suggested_records = preview_df.to_dict(orient="records")
                    elif hasattr(ai_df, "to_dict"):
                        try:
                            suggested_records = list(ai_df.to_dict().items())
                        except Exception:
                            suggested_records = []
                    else:
                        suggested_records = []

                    trace(
                        {
                            "where": "ai:raw",
                            "sku": sku_value,
                            "set_id": set_id,
                            "suggested": suggested_records,
                        }
                    )

                    categories_meta_for_ai = step2_state.get("categories_meta")
                    if not isinstance(categories_meta_for_ai, Mapping):
                        categories_meta_for_ai = {}
                    attribute_set_label = set_names.get(set_id, "")
                    ai_df = enrich_ai_suggestions(
                        ai_df,
                        hints_payload,
                        categories_meta_for_ai,
                        attribute_set_label,
                        set_id,
                    )

                    sku_key = str(sku_value)
                    per_sku = ai_results.setdefault(set_id, {}).setdefault(sku_key, {})

                    metadata = getattr(ai_df, "attrs", {}).get("meta") if hasattr(ai_df, "attrs") else None
                    if isinstance(metadata, dict) and metadata:
                        per_sku["__meta__"] = metadata

                    trace(
                        {
                            "where": "ai:enriched",
                            "sku": sku_value,
                            "set_id": set_id,
                            "keys": list(per_sku.keys()),
                        }
                    )

                    if isinstance(ai_df, pd.DataFrame) and not ai_df.empty:
                        for _, suggestion in ai_df.iterrows():
                            code = suggestion.get("code")
                            if not code:
                                continue
                            value = suggestion.get("value")
                            if value is None:
                                continue
                            if isinstance(value, str) and not value.strip():
                                continue
                            reason = suggestion.get("reason")
                            per_sku[str(code)] = {
                                "value": value,
                                "reason": reason,
                            }

                    # Можно чуть-чуть обновлять прогресс (необязательно)
                    done_count += 1
                    try:
                        progress_value = 35 + int(15 * done_count / max(total, 1))
                        # Если есть pbar/_pupdate — здесь можно дергать их
                        # Но если не хочешь — этот блок можно убрать.
                    except Exception:
                        pass

    return (
        dfs,
        column_configs,
        disabled,
        meta_cache,
        row_meta_by_set,
        set_names,
        {key: value for key, value in ai_results.items()},
        ai_errors,
        ai_log_data,
    )

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


def _sync_ai_highlight_state(
    step2_state: dict,
    set_id: int,
    editor_df: pd.DataFrame | None,
) -> None:
    if not isinstance(step2_state, dict):
        return
    if not isinstance(editor_df, pd.DataFrame) or editor_df.empty:
        return
    ai_cells_map = step2_state.get("ai_cells")
    if not isinstance(ai_cells_map, dict):
        return
    existing_cells = ai_cells_map.get(set_id)
    if not existing_cells:
        return
    if "sku" not in editor_df.columns:
        return

    normalized = editor_df.copy(deep=True)
    normalized["sku"] = normalized["sku"].astype(str)

    keep: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for cell in existing_cells:
        if isinstance(cell, dict):
            raw_sku = cell.get("sku")
            code_value = cell.get("code")
            suggested_value = cell.get("value")
        elif isinstance(cell, (list, tuple)) and len(cell) >= 2:
            raw_sku, code_value = cell[0], cell[1]
            suggested_value = cell[2] if len(cell) > 2 else None
        else:
            continue

        sku_value = str(raw_sku or "")
        code_key = str(code_value or "")

        if not sku_value or not code_key:
            continue
        if code_key not in normalized.columns:
            continue

        matches = normalized[normalized["sku"] == sku_value]
        if matches.empty:
            continue

        current_value = matches.iloc[0][code_key]
        if suggested_value is not None and not _values_equal(current_value, suggested_value):
            continue

        key = (sku_value, code_key)
        if key in seen:
            continue
        seen.add(key)
        keep.append(
            {
                "sku": sku_value,
                "code": code_key,
                "value": suggested_value,
            }
        )

    if keep:
        ai_cells_map[set_id] = keep
    else:
        ai_cells_map.pop(set_id, None)


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


def apply_product_update(
    session,
    api_base: str,
    sku: str,
    attributes: dict,
    meta_by_attr: dict[str, dict[str, object]] | None = None,
):
    if not attributes:
        return

    attrs = dict(attributes or {})
    attr_set_id = attrs.get("attribute_set_id")

    row_payload = {"sku": sku, "attribute_set_id": attr_set_id}
    for code, value in attrs.items():
        if code == "attribute_set_id":
            continue
        row_payload[code] = value

    payload = build_magento_payload(row_payload, meta_by_attr or {})

    url = f"{api_base.rstrip('/')}/products/{quote(sku, safe='')}"
    product_payload = payload.get("product", {}) if isinstance(payload, dict) else {}
    custom_attributes = product_payload.get("custom_attributes", [])
    extension_attributes = product_payload.get("extension_attributes", {})
    if logger.isEnabledFor(logging.INFO):
        attr_codes = [
            item.get("attribute_code")
            for item in custom_attributes
            if isinstance(item, dict)
        ]
        payload_preview = json.dumps(payload, ensure_ascii=False)[:200]
        logger.info(
            "Magento PUT %s attrs=%s payload=%s",
            sku,
            ",".join(code for code in attr_codes if code),
            payload_preview,
        )
    _dbg = {
        "where": "save:http_put",
        "sku": sku,
        "url": url,
        "custom_attr_codes": [
            a.get("attribute_code") for a in custom_attributes if isinstance(a, dict)
        ],
        "ext_categories": [
            l.get("category_id")
            for l in extension_attributes.get("category_links", [])
        ],
    }
    try:
        trace(_dbg)
    except Exception:
        pass
    resp = session.put(
        url,
        json=payload,
        headers=build_magento_headers(session=session),
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Magento update failed for {sku}: {resp.status_code} {resp.text[:200]}"
        )


def _collect_step2_products_rows(
    df_filtered: pd.DataFrame | None = None,
    df_original: pd.DataFrame | None = None,
    row_meta_map: dict[int, dict[str, dict]] | None = None,
    step2_state: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if isinstance(df_filtered, pd.DataFrame):
        df = df_filtered.copy()
    else:
        # fallback
        products = st.session_state.get("step2_output_rows", [])
        if not products:
            return []
        df = pd.DataFrame(products)

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        sku = row.get("sku", "").strip()
        if not sku:
            continue
        for col in df.columns:
            if col == "sku":
                continue
            value = row.get(col)
            if _is_blank_value(value):
                continue

            if col == "categories":
                # Map category labels to IDs using categories_meta
                cats_meta = step2_state.get("categories_meta", {}) if step2_state else {}
                options_map = cats_meta.get("options_map", {}) if isinstance(cats_meta, dict) else {}
                # Ensure value is a list (e.g. split comma-separated strings or wrap single value)
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        # Split by comma for multiple labels, or treat as single label
                        labels = [v.strip() for v in value.split(",") if v.strip()]
                        value = labels or []
                if not isinstance(value, list):
                    value = [value]
                cleaned_list = []
                for cat in value:
                    if isinstance(cat, str):
                        label = cat.strip()
                        if not label:
                            continue
                        # Lookup label (case-insensitive) in options_map to get ID
                        mapped_id = options_map.get(label) or options_map.get(label.casefold())
                        if mapped_id:
                            try:
                                cleaned_list.append(int(mapped_id))
                            except Exception:
                                continue
                    else:
                        try:
                            cleaned_list.append(int(cat))
                        except Exception:
                            continue
                value = cleaned_list
                if not value:
                    continue

            rows.append({
                "sku": sku,
                "attribute_code": col,
                "value": value,
            })

    return rows


def _collect_step2_output_rows() -> list[dict[str, object]]:
    step2_state = _ensure_step2_state()
    _commit_step2_staged(step2_state)

    wide_map: dict[int, pd.DataFrame] = step2_state.get("wide", {})
    wide_meta_map: dict[int, dict[str, dict]] = step2_state.get("wide_meta", {})

    rows: list[dict[str, object]] = []
    attr_set_by_sku: dict[str, object] = {}

    for set_id, wide_df in (wide_map or {}).items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue

        meta_for_set = wide_meta_map.get(set_id, {}) if isinstance(wide_meta_map, dict) else {}
        attr_cols = [
            col
            for col in wide_df.columns
            if col not in {"sku", "name", "attribute_set_id", GENERATE_DESCRIPTION_COLUMN}
        ]

        for _, row in wide_df.iterrows():
            sku = _normalize_text(row.get("sku"))
            if not sku:
                continue

            attr_set_raw = row.get("attribute_set_id", set_id)
            if attr_set_raw in (None, ""):
                attr_set_by_sku.setdefault(sku, set_id)
            else:
                attr_set_by_sku.setdefault(sku, attr_set_raw)

            for col in attr_cols:
                meta = meta_for_set.get(col) or {}
                attr_code = meta.get("attribute_code") or col
                rows.append(
                    {
                        "sku": sku,
                        "store_view_code": "all",
                        "attribute_code": attr_code,
                        "value": row.get(col),
                    }
                )

    st.session_state["step2_output_rows"] = rows
    step2_state.setdefault("attr_set_by_sku", {}).update(attr_set_by_sku)
    return rows


def _save_specs_step2_to_magento(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> None:
    if rows is None:
        rows = st.session_state.get("step2_output_rows")

    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    if not isinstance(rows, list) or not rows:
        rows = _collect_step2_products_rows() or _collect_step2_output_rows()
    if not rows:
        st.info("No specs to save.")
        return

    session = st.session_state.get("mg_session")
    base_url = st.session_state.get("mg_base_url")
    if not session or not base_url:
        st.error("Magento session is not initialized")
        return

    api_base = st.session_state.get("ai_api_base") or get_api_base(base_url)
    st.session_state["ai_api_base"] = api_base

    step2_state = _ensure_step2_state()
    meta_cache = step2_state.get("meta_cache")
    wide_meta_map: dict[int, dict[str, dict]] = step2_state.get("wide_meta", {})
    attr_set_by_sku: dict[str, object] = step2_state.get("attr_set_by_sku", {})

    meta_by_code: dict[str, dict[str, object]] = {}

    if isinstance(meta_cache, AttributeMetaCache):
        meta_cache.build_and_set_static_for(["country_of_manufacture"], store_id=0)

    for set_meta in (wide_meta_map or {}).values():
        if not isinstance(set_meta, dict):
            continue
        for key, meta in set_meta.items():
            attr_code = (meta.get("attribute_code") if isinstance(meta, dict) else None) or key
            if not attr_code:
                continue
            if attr_code not in meta_by_code:
                meta_by_code[attr_code] = meta if isinstance(meta, dict) else {}

    payload_by_sku: dict[str, dict[str, object]] = {}

    for row in rows:
        sku = _normalize_text(row.get("sku"))
        attr_code = _normalize_text(row.get("attribute_code"))
        if not sku or not attr_code:
            continue
        entry = payload_by_sku.setdefault(
            sku,
            {
                "attribute_set_id": attr_set_by_sku.get(sku),
                "values": {},
                "meta": {},
            },
        )
        entry["values"][attr_code] = row.get("value")
        meta_for_attr = meta_by_code.get(attr_code) or {}
        if isinstance(meta_cache, AttributeMetaCache):
            cache_meta = meta_cache.get(attr_code)
            if isinstance(cache_meta, dict):
                meta_for_attr = cache_meta
        if meta_for_attr:
            entry["meta"][attr_code] = meta_for_attr

    processed_rows = 0
    errors: list[str] = []

    for sku, entry in payload_by_sku.items():
        values_map = entry.get("values") if isinstance(entry, dict) else {}
        meta_map = entry.get("meta") if isinstance(entry, dict) else {}
        if not isinstance(values_map, dict):
            continue
        attributes_payload: dict[str, object] = {}
        if entry.get("attribute_set_id") is not None:
            attributes_payload["attribute_set_id"] = entry.get("attribute_set_id")
        attributes_payload.update(values_map)
        try:
            apply_product_update(session, api_base, sku, attributes_payload, meta_map)
            processed_rows += len(values_map)
        except Exception as exc:  # pragma: no cover - network interaction
            errors.append(f"{sku}: {exc}")

    if errors:
        st.error("❌ Failed to save specs: " + "; ".join(errors))
        return

    st.success(f"✅ Specs saved to Magento ({processed_rows} rows)")


def _save_rows_to_magento(
    rows: Sequence[Mapping[str, object]] | None = None,
    *,
    session=None,
    trace=None,
    is_dry_run: bool | None = None,
) -> None:
    _save_specs_step2_to_magento(rows)


def save_rows_to_magento(
    rows: Sequence[Mapping[str, object]] | None = None,
    *,
    session=None,
    trace=None,
    is_dry_run: bool | None = None,
) -> None:
    _save_rows_to_magento(rows, session=session, trace=trace, is_dry_run=is_dry_run)


def save_step2_to_magento():
    step2_state = st.session_state.get("step2")
    if not isinstance(step2_state, dict):
        st.info("No changes to save.")
        return

    def _short_preview(value: object, limit: int = 120) -> str:
        if value in (None, ""):
            return ""
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    session = st.session_state.get("mg_session")
    base_url = st.session_state.get("mg_base_url")
    if not session or not base_url:
        st.error("Magento session is not initialized")
        return

    _commit_step2_staged(step2_state)

    wide_map: dict[int, pd.DataFrame] = step2_state.get("wide", {})
    wide_meta_map: dict[int, dict[str, dict]] = step2_state.get("wide_meta", {})
    baseline_map: dict[int, pd.DataFrame] = step2_state.get("wide_synced", {})
    meta_cache = step2_state.get("meta_cache")
    if meta_cache is None:
        meta_cache = step2_state.get("meta")

    force_map = st.session_state.get("force_send", {})
    if not isinstance(force_map, dict):
        force_map = {}

    if isinstance(meta_cache, AttributeMetaCache):
        meta_cache.build_and_set_static_for(["country_of_manufacture"], store_id=0)

    if not wide_map:
        st.info("No changes to save.")
        return

    payload_by_sku: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []

    for set_id, wide_df in wide_map.items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue

        attr_cols = [
            col
            for col in wide_df.columns
            if col
            not in (
                "sku",
                "name",
                "attribute_set_id",
                GENERATE_DESCRIPTION_COLUMN,
            )
        ]
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

            attr_set_value = row.get("attribute_set_id", set_id)
            if isinstance(attr_set_value, float) and pd.isna(attr_set_value):
                attr_set_value = set_id
            if attr_set_value in (None, ""):
                attr_set_value = set_id
            try:
                attr_set_clean = int(attr_set_value)
            except (TypeError, ValueError):
                attr_set_clean = attr_set_value
            entry = payload_by_sku.setdefault(
                sku_value,
                {"attribute_set_id": attr_set_clean, "values": {}, "meta": {}},
            )
            entry["attribute_set_id"] = attr_set_clean

            baseline_row: pd.Series | None
            if baseline_idx is not None and sku in baseline_idx.index:
                baseline_row = baseline_idx.loc[sku]
                if isinstance(baseline_row, pd.DataFrame):
                    baseline_row = baseline_row.iloc[-1]
            else:
                baseline_row = None

            sku_force_codes_raw = force_map.get(sku_value, set())
            if isinstance(sku_force_codes_raw, (set, list, tuple)):
                sku_force_codes = {str(item) for item in sku_force_codes_raw}
            elif sku_force_codes_raw in (None, ""):
                sku_force_codes = set()
            else:
                sku_force_codes = {str(sku_force_codes_raw)}

            for code in attr_cols:
                new_raw = row.get(code)
                old_raw = (
                    baseline_row.get(code)
                    if isinstance(baseline_row, pd.Series) and code in baseline_row.index
                    else None
                )

                must_force = str(code) in sku_force_codes
                if (not must_force) and _values_equal(new_raw, old_raw):
                    continue

                meta = meta_for_set.get(code) or {}
                attr_code = meta.get("attribute_code") or code

                meta_info: dict[str, object] = {}
                if isinstance(meta_cache, AttributeMetaCache):
                    cached_meta = meta_cache.get(attr_code)
                    if isinstance(cached_meta, dict):
                        meta_info = cached_meta
                elif isinstance(meta_cache, dict):  # pragma: no cover - safety fallback
                    cached_meta = meta_cache.get(attr_code)
                    if isinstance(cached_meta, dict):
                        meta_info = cached_meta
                if not meta_info and isinstance(meta, dict):
                    meta_info = meta

                input_type = (
                    meta_info.get("frontend_input")
                    or meta.get("frontend_input")
                    or meta_info.get("backend_type")
                    or meta.get("backend_type")
                    or ""
                )
                input_type = str(input_type).lower()

                prepared_value = new_raw
                if (
                    attr_code != "categories"
                    and input_type == "multiselect"
                ):
                    labels = _split_multiselect_input(new_raw)
                    label_to_value: dict[str, object] = {}

                    options_map = meta_info.get("options_map") if isinstance(meta_info, dict) else {}
                    if isinstance(options_map, dict):
                        for key, target in options_map.items():
                            if isinstance(key, str):
                                cleaned_key = key.strip()
                                if not cleaned_key:
                                    continue
                                label_to_value.setdefault(cleaned_key, target)
                                label_to_value.setdefault(cleaned_key.casefold(), target)
                            else:
                                key_str = str(key).strip()
                                if not key_str:
                                    continue
                                label_to_value.setdefault(key_str, target)

                    labels_to_values = meta_info.get("labels_to_values") if isinstance(meta_info, dict) else {}
                    if not isinstance(labels_to_values, dict):
                        labels_to_values = {}

                    if not labels_to_values:
                        values_to_labels = meta_info.get("values_to_labels") if isinstance(meta_info, dict) else {}
                        if isinstance(values_to_labels, dict):
                            for value_key, label in values_to_labels.items():
                                if not isinstance(label, str):
                                    continue
                                cleaned_label = label.strip()
                                if not cleaned_label:
                                    continue
                                labels_to_values.setdefault(cleaned_label, value_key)
                                labels_to_values.setdefault(cleaned_label.casefold(), value_key)
                    else:
                        normalized_labels_to_values: dict[str, object] = {}
                        for label_key, mapped_value in labels_to_values.items():
                            if not isinstance(label_key, str):
                                continue
                            cleaned_label = label_key.strip()
                            if not cleaned_label:
                                continue
                            normalized_labels_to_values.setdefault(cleaned_label, mapped_value)
                            normalized_labels_to_values.setdefault(cleaned_label.casefold(), mapped_value)
                        labels_to_values = normalized_labels_to_values

                    for label_key, mapped_value in labels_to_values.items():
                        label_to_value.setdefault(label_key, mapped_value)

                    mapped_ids: list[object] = []
                    for label in labels:
                        if not isinstance(label, str):
                            text_label = str(label).strip()
                        else:
                            text_label = label.strip()
                        if not text_label:
                            continue

                        candidate = label_to_value.get(text_label)
                        if candidate is None:
                            candidate = label_to_value.get(text_label.casefold())
                        if candidate is None:
                            candidate = label_to_value.get(text_label.lower())
                        if candidate is None:
                            try:
                                candidate = int(text_label)
                            except (TypeError, ValueError):
                                candidate = text_label

                        if candidate in (None, ""):
                            continue

                        if isinstance(candidate, str):
                            candidate_str = candidate.strip()
                            if not candidate_str:
                                continue
                            try:
                                candidate_val: object = int(candidate_str)
                            except (TypeError, ValueError):
                                candidate_val = candidate_str
                        else:
                            candidate_val = candidate

                        if candidate_val not in mapped_ids:
                            mapped_ids.append(candidate_val)

                    prepared_value = mapped_ids

                trace(
                    {
                        "where": "save:pre-normalize",
                        "sku": sku_value,
                        "code": attr_code,
                        "input_type": input_type,
                        "prepared_preview": str(prepared_value)[:200],
                        "old_vs_new": {
                            "old": _short_preview(old_raw),
                            "new": _short_preview(prepared_value),
                        },
                    }
                )
                try:
                    normalized = normalize_for_magento(attr_code, prepared_value, meta_cache)
                except Exception as exc:  # pragma: no cover - safety guard
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": repr(new_raw),
                            "hint_examples": f"normalize error: {exc}",
                            "expected_codes": "",
                        }
                    )
                    continue

                trace(
                    {
                        "where": "save:post-normalize",
                        "sku": sku_value,
                        "code": attr_code,
                        "normalized_preview": str(normalized)[:200],
                    }
                )

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
                    expected_codes: list[str] = []
                    values_to_labels = meta_info.get("values_to_labels")
                    if isinstance(values_to_labels, dict):
                        expected_codes = sorted(
                            [
                                str(value)
                                for value in values_to_labels.values()
                                if value not in (None, "")
                            ]
                        )[:10]
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": repr(new_raw),
                            "hint_examples": "\n".join(examples[:5]),
                            "expected_codes": ", ".join(expected_codes),
                        }
                    )
                    continue

                meta_for_payload: dict[str, object] = {}
                if isinstance(meta_info, dict):
                    meta_for_payload = dict(meta_info)

                options_map_raw = meta_for_payload.get("options_map")
                normalized_options_map: dict[str, str] = {}
                if isinstance(options_map_raw, dict):
                    for opt_key, opt_value in options_map_raw.items():
                        if opt_key in (None, ""):
                            continue
                        if isinstance(opt_key, str):
                            key_clean = opt_key.strip()
                            if not key_clean:
                                continue
                            normalized_options_map[key_clean.lower()] = str(opt_value)
                        else:
                            key_str = str(opt_key).strip()
                            if not key_str:
                                continue
                            normalized_options_map[key_str.lower()] = str(opt_value)
                elif isinstance(meta_for_payload.get("values_to_labels"), dict):
                    values_to_labels = meta_for_payload.get("values_to_labels") or {}
                    for value_key, label in values_to_labels.items():
                        if not isinstance(label, str):
                            continue
                        label_clean = label.strip()
                        if not label_clean:
                            continue
                        normalized_options_map[label_clean.lower()] = str(value_key)

                meta_for_payload["options_map"] = normalized_options_map
                meta_for_payload["frontend_input"] = input_type

                entry = payload_by_sku.setdefault(
                    sku_value,
                    {"attribute_set_id": attr_set_clean, "values": {}, "meta": {}},
                )
                entry.setdefault("values", {})[attr_code] = normalized
                entry.setdefault("meta", {})[attr_code] = meta_for_payload

            entry = payload_by_sku.get(sku_value)
            values_map = entry.get("values") if isinstance(entry, dict) else None
            if entry and (not values_map or not values_map.keys()):
                payload_by_sku.pop(sku_value, None)

    if not payload_by_sku and not errors:
        st.info("No changes to save.")
        return

    api_base = st.session_state.get("ai_api_base")
    if not api_base:
        st.warning("Magento API is not initialized.")
        return

    ok_skus: set[str] = set()
    for sku, entry in payload_by_sku.items():
        values_map = entry.get("values") if isinstance(entry, dict) else {}
        if not isinstance(values_map, dict):
            values_map = {}
        meta_map = entry.get("meta") if isinstance(entry, dict) else {}
        if not isinstance(meta_map, dict):
            meta_map = {}

        attr_keys = list(values_map.keys())
        trace(
            {
                "where": "save:payload",
                "sku": sku,
                "keys": sorted(attr_keys + ["attribute_set_id"]),
                "payload_preview": {
                    k: (
                        _short_preview(v)
                        if k != "categories"
                        else v
                    )
                    for k, v in values_map.items()
                },
                "attribute_set_id": entry.get("attribute_set_id"),
                "count_custom_attrs": len(attr_keys),
                "has_categories": "categories" in attr_keys,
            }
        )

        to_drop: list[str] = []
        for code, raw_value in list(values_map.items()):
            if code == "categories":
                continue
            meta_for_code = meta_map.get(code) if isinstance(meta_map, dict) else {}
            mapped_value = map_value_for_magento(code, raw_value, meta_for_code)
            frontend_input = str(
                (meta_for_code or {}).get("frontend_input") or ""
            ).lower()

            if frontend_input in {"select", "boolean"} and mapped_value in (None, ""):
                label_text = (
                    str(raw_value)[:120] if raw_value not in (None, "") else ""
                )
                errors.append(
                    {
                        "sku": sku,
                        "attribute": code,
                        "raw": repr(raw_value),
                        "hint_examples": f'Не найден option_id для "{label_text}" в атрибуте {code} — атрибут пропущен при импорте.',
                        "expected_codes": "",
                    }
                )
                to_drop.append(code)
            elif frontend_input == "multiselect" and mapped_value == "":
                label_text = (
                    ", ".join(
                        str(v).strip()
                        for v in (raw_value or [])
                        if v not in (None, "")
                    )
                    if isinstance(raw_value, (list, tuple, set))
                    else (
                        str(raw_value)[:120]
                        if raw_value not in (None, "")
                        else ""
                    )
                )
                errors.append(
                    {
                        "sku": sku,
                        "attribute": code,
                        "raw": repr(raw_value),
                        "hint_examples": f'Не найден option_id для "{label_text}" в атрибуте {code} — атрибут пропущен при импорте.',
                        "expected_codes": "",
                    }
                )
                to_drop.append(code)

        for code in to_drop:
            values_map.pop(code, None)

        attributes_payload: dict[str, object] = {}
        if entry.get("attribute_set_id") is not None:
            attributes_payload["attribute_set_id"] = entry.get("attribute_set_id")
        attributes_payload.update(values_map)

        try:
            apply_product_update(
                session,
                api_base,
                sku,
                attributes_payload,
                meta_map,
            )
            ok_skus.add(sku)
            trace({"where": "save:http_ok", "sku": sku, "status": "ok"})
        except Exception as exc:  # pragma: no cover - network interaction
            errors.append(
                {
                    "sku": sku,
                    "attribute": "*batch*",
                    "raw": repr(attributes_payload),
                    "hint_examples": f"{exc}\nKeys: {', '.join(sorted(attributes_payload.keys()))}",
                    "expected_codes": "",
                }
            )
            trace(
                {
                    "where": "save:http_err",
                    "sku": sku,
                    "err": _short_preview(exc, limit=200),
                }
            )

    if ok_skus:
        st.success("Updated SKUs:")
        st.markdown("\n".join(f"- `{sku}`" for sku in sorted(ok_skus)))
        for sku in ok_skus:
            force_map.pop(str(sku), None)
        st.session_state["force_send"] = force_map

    if errors:
        import pandas as _pd

        if any(err.get("attribute") == "categories" for err in errors):
            st.warning("Some categories are not saved because they do not exist in Magento.")

        st.warning("Some rows failed; review and fix:")
        st.dataframe(
            _pd.DataFrame(
                errors,
                columns=[
                    "sku",
                    "attribute",
                    "raw",
                    "hint_examples",
                    "expected_codes",
                ],
            ),
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
    skip_filters: bool = False,
):
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

    def _attr_set_name(product: dict, fallback_attr_set_id: int | None) -> str:
        attr_name = product.get("attribute_set_name")
        if attr_name:
            return str(attr_name)
        attr_value = product.get("attribute_set_id")
        if attr_value not in (None, ""):
            return str(attr_value)
        if fallback_attr_set_id not in (None, ""):
            return str(fallback_attr_set_id)
        return _DEF_ATTR_SET_NAME

    if skip_filters:
        rows: list[dict[str, str]] = []
        prog = st.progress(0.0, text="Loading products…")
        total_hint = 0

        if products is None:
            effective_attr_set_id = attr_set_id
            if effective_attr_set_id is None:
                effective_attr_set_id = get_attr_set_id(
                    session, base_url, name=_DEF_ATTR_SET_NAME
                )
            iterator = iter_products_by_attr_set(
                session, base_url, effective_attr_set_id
            )
            for product, total in iterator:
                total_hint = total or total_hint
                rows.append(
                    {
                        "sku": product.get("sku", ""),
                        "name": product.get("name", ""),
                        "attribute set": _attr_set_name(
                            product, effective_attr_set_id
                        ),
                        "created_at": product.get("created_at", ""),
                    }
                )
                if total_hint:
                    done = len(rows) / max(total_hint, 1)
                    done = min(done, 1.0)
                    prog.progress(
                        done, text=f"Loading products… {int(done * 100)}%"
                    )
        else:
            total_hint = len(products)
            for idx, product in enumerate(products, start=1):
                rows.append(
                    {
                        "sku": product.get("sku", ""),
                        "name": product.get("name", ""),
                        "attribute set": _attr_set_name(product, attr_set_id),
                        "created_at": product.get("created_at", ""),
                    }
                )
                if total_hint:
                    done = idx / max(total_hint, 1)
                    done = min(done, 1.0)
                    prog.progress(
                        done, text=f"Loading products… {int(done * 100)}%"
                    )

        prog.progress(1.0, text="Products loaded")

        df_skip = pd.DataFrame(rows)
        if df_skip.empty:
            return _empty_df()
        return df_skip.reindex(columns=["sku", "name", "attribute set", "created_at"])

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
        return _empty_df()

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
        return _empty_df()

    out["attribute set"] = _DEF_ATTR_SET_NAME
    return out[["sku", "name", "attribute set", "created_at"]]


magento_base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
magento_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
magento_session = get_session(auth_token=magento_token)
st.session_state["mg_session"] = magento_session
st.session_state["mg_base_url"] = magento_base_url

ai_api_key_secret = st.secrets.get("OPENAI_API_KEY", "")
if not ai_api_key_secret:
    ai_api_key_secret = os.getenv("OPENAI_API_KEY", "")
ai_model_secret_raw = st.secrets.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
if isinstance(ai_model_secret_raw, str):
    ai_model_secret_raw = ai_model_secret_raw.strip() or DEFAULT_OPENAI_MODEL
else:
    ai_model_secret_raw = DEFAULT_OPENAI_MODEL
ai_model_secret = resolve_model_name(ai_model_secret_raw, default=DEFAULT_OPENAI_MODEL)
st.session_state["ai_api_key"] = ai_api_key_secret
model_options: list[str] = []
for option in (
    DEFAULT_OPENAI_MODEL,
    OPENAI_MODEL_ALIASES.get("5-mini"),
    OPENAI_MODEL_ALIASES.get("5"),
    OPENAI_MODEL_ALIASES.get("4-mini"),
    OPENAI_MODEL_ALIASES.get("4"),
    OPENAI_MODEL_ALIASES.get("o4-mini"),
    OPENAI_MODEL_ALIASES.get("o4"),
):
    if option and option not in model_options:
        model_options.append(option)
if ai_model_secret and ai_model_secret not in model_options:
    model_options.append(ai_model_secret)
if not model_options:
    model_options = [DEFAULT_OPENAI_MODEL]

default_model = DEFAULT_OPENAI_MODEL
preferred_model = ai_model_secret if ai_model_secret in model_options else default_model
stored_model = st.session_state.get("ai_model")
if not stored_model:
    st.session_state["ai_model"] = preferred_model
    stored_model = preferred_model
current_model = stored_model
resolved_current = resolve_model_name(current_model, default=default_model)
if current_model not in model_options and resolved_current in model_options:
    current_model = resolved_current
if current_model not in model_options:
    current_model = preferred_model if preferred_model in model_options else default_model

selected_model = st.sidebar.selectbox(
    "AI model",
    model_options,
    index=model_options.index(current_model),
)

st.session_state["ai_model"] = selected_model
resolved_selected = resolve_model_name(selected_model, default=default_model)
st.session_state["ai_model_resolved"] = resolved_selected

if not st.session_state.get("_mg_auth_logged"):
    st.write(
        "Auth header:",
        bool(st.session_state["mg_session"].headers.get("Authorization")),
    )
    st.session_state["_mg_auth_logged"] = True

session = st.session_state.get("mg_session")
base_url = st.session_state.get("mg_base_url")
if not session or not base_url:
    st.error("Magento session is not initialized")
    st.stop()

st.info("Let’s find items need to be updated.")

if "step1_editor_mode" not in st.session_state:
    st.session_state["step1_editor_mode"] = "all"

step1_state = st.session_state.setdefault("step1", {})

c1, c2, c3 = st.columns(3)
btn_all = c1.button("📦 Load All", key="btn_load_all")
btn_50 = c2.button("⚡ New arrivals", key="btn_load_50_fast")
btn_test = c3.button("🧪 Test ART-22895", key="btn_test_art_22895")

if btn_all:
    requested_run_mode: str | None = "all"
elif btn_50:
    requested_run_mode = "fast50"
elif btn_test:
    requested_run_mode = "test_art_22895"
else:
    requested_run_mode = None

if requested_run_mode:
    run_mode = requested_run_mode
    st.session_state["step1_editor_mode"] = run_mode
    st.session_state["show_attributes_trigger"] = False
    if run_mode != "test_art_22895":
        step1_state.pop("products", None)

    all_modes = {"all", "fast50", "test_art_22895"}
    for prefix in ("default_products", "df_original", "df_edited", "attribute_sets"):
        for other_mode in all_modes - {run_mode}:
            st.session_state.pop(f"{prefix}_{other_mode}", None)

    cache_key = f"default_products_{run_mode}"
    df_original_key = f"df_original_{run_mode}"
    df_edited_key = f"df_edited_{run_mode}"
    attribute_sets_key = f"attribute_sets_{run_mode}"

    if run_mode == "test_art_22895":
        test_skus = ["ART-22895", "ARC-22895", "ARB-22895", "ARO-22895"]
        status = st.status("Loading test products…", expanded=True)
        pbar = st.progress(0)

        found_products: list[dict] = []
        missing_skus: list[str] = []

        load_failed = False
        try:
            api_base = st.session_state.get("ai_api_base")
            if not api_base:
                api_base = get_api_base(base_url)
                st.session_state["ai_api_base"] = api_base

            status.update(label="Fetching test products…")
            for idx, sku in enumerate(test_skus, start=1):
                try:
                    product = get_product_by_sku(session, api_base, sku)
                except Exception:
                    product = None

                if product and product.get("sku"):
                    found_products.append(product)
                else:
                    missing_skus.append(sku)

                progress_value = int(40 * idx / len(test_skus))
                if progress_value > 0:
                    pbar.progress(progress_value)
        except Exception as exc:  # pragma: no cover - network/UI error handling
            status.update(state="error", label="Failed to load test products")
            st.error(f"Error: {exc}")
            st.session_state.pop(df_original_key, None)
            st.session_state.pop(df_edited_key, None)
            st.session_state.pop(attribute_sets_key, None)
            _reset_step2_state()
            pbar.progress(100)
            load_failed = True

        if load_failed:
            st.stop()

        if not found_products:
            status.update(state="error", label="Failed to load test products")
            st.warning("Ни один из тестовых SKU не найден.")
            st.session_state.pop(df_original_key, None)
            st.session_state.pop(df_edited_key, None)
            st.session_state.pop(attribute_sets_key, None)
            _reset_step2_state()
            pbar.progress(100)
        else:
            if missing_skus:
                status.write(
                    "Следующие SKU не найдены и будут пропущены: "
                    + ", ".join(missing_skus)
                )

            step1_state["products"] = found_products
            step1_state["skus"] = [p.get("sku") for p in found_products if p.get("sku")]

            attr_sets_existing = st.session_state.get(attribute_sets_key, {})
            set_choices = step1_state.get("set_choices")
            if not isinstance(set_choices, dict):
                set_choices = {}

            for product in found_products:
                sku = product.get("sku")
                attr_set_id_value = product.get("attribute_set_id")
                try:
                    attr_set_id_int = int(attr_set_id_value)
                except (TypeError, ValueError):
                    attr_set_id_int = None

                attr_name = None
                if isinstance(attr_sets_existing, dict) and attr_sets_existing:
                    for name, attr_id in attr_sets_existing.items():
                        try:
                            if int(attr_id) == int(attr_set_id_int):
                                attr_name = name
                                break
                        except (TypeError, ValueError):
                            continue

                if not attr_name:
                    if attr_set_id_int is None:
                        attr_name = _DEF_ATTR_SET_NAME
                    else:
                        attr_name = str(attr_set_id_int)

                if sku:
                    set_choices[sku] = {
                        "attribute_set_id": attr_set_id_int,
                        "attribute_set_name": attr_name,
                    }
                product["attribute_set_name"] = attr_name

            step1_state["set_choices"] = set_choices

            data = found_products
            st.session_state[cache_key] = data

            pbar.progress(60)
            status.update(label="Parsing response…")
            df_items = load_items(
                session,
                base_url,
                attr_set_id=None,
                products=data,
                skip_filters=True,
            )
            pbar.progress(90)
            status.update(label="Building table…")
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
                df_ui = df_ui.sort_values("created_at", ascending=False).reset_index(
                    drop=True
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
                cols_order = [
                    "sku",
                    "name",
                    "attribute set",
                    "attribute_set_id",
                    "hint",
                    "created_at",
                ]
                column_order = [
                    col for col in cols_order if col in df_for_edit.columns
                ]
                lead_cols = [c for c in ("sku", "name") if c in column_order]
                tail_cols = [c for c in column_order if c not in ("sku", "name")]
                column_order = lead_cols + tail_cols
                df_for_edit = df_for_edit[column_order]
                st.session_state[df_edited_key] = df_for_edit.copy()
                _reset_step2_state()

            pbar.progress(100)
            status.update(
                state="complete",
                label=f"Loaded {len(data or [])} items",
            )
            st.success(f"Loaded {len(data or [])} test item(s).")

    else:
        limit = 50 if run_mode == "fast50" else None
        enabled_only = None
        minimal_fields = run_mode == "fast50"
        extra_params: dict[str, str] = {}
        if run_mode == "fast50":
            extra_params = {
                "searchCriteria[sortOrders][0][field]": "created_at",
                "searchCriteria[sortOrders][0][direction]": "DESC",
                "searchCriteria[pageSize]": "50",
            }

        status = st.status("Loading default products…", expanded=True)
        pbar = st.progress(0)
        data: list[dict] | None = None
        try:
            pbar.progress(5)
            status.update(label="Preparing request…")
            attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)
            data = get_default_products(
                session,
                base_url,
                attr_set_id=attr_set_id,
                qty_min=0,
                limit=limit,
                minimal_fields=minimal_fields,
                enabled_only=enabled_only,
                **extra_params,
            )
            st.session_state[cache_key] = data
            pbar.progress(60)
            status.update(label="Parsing response…")
            df_items = load_items(
                session,
                base_url,
                attr_set_id=attr_set_id,
                products=data,
            )
            pbar.progress(90)
            status.update(label="Building table…")
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
                df_ui = df_ui.sort_values("created_at", ascending=False).reset_index(
                    drop=True
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
                cols_order = [
                    "sku",
                    "name",
                    "attribute set",
                    "attribute_set_id",
                    "hint",
                    "created_at",
                ]
                column_order = [
                    col for col in cols_order if col in df_for_edit.columns
                ]
                lead_cols = [c for c in ("sku", "name") if c in column_order]
                tail_cols = [c for c in column_order if c not in ("sku", "name")]
                column_order = lead_cols + tail_cols
                df_for_edit = df_for_edit[column_order]
                st.session_state[df_edited_key] = df_for_edit.copy()
                _reset_step2_state()
            pbar.progress(100)
            status.update(
                state="complete",
                label=f"Loaded {len(data or [])} items",
            )
        except Exception as exc:  # pragma: no cover - UI error handling
            status.update(state="error", label="Failed to load products")
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
            cols_order = [
                "sku",
                "name",
                "attribute set",
                "attribute_set_id",
                "hint",
                "created_at",
            ]
            column_order = [col for col in cols_order if col in df_init.columns]
            lead_cols = [c for c in ("sku", "name") if c in column_order]
            tail_cols = [c for c in column_order if c not in ("sku", "name")]
            column_order = lead_cols + tail_cols
            df_init = df_init[column_order]
            st.session_state[df_edited_key] = df_init.copy()

        df_base = st.session_state[df_edited_key].copy()

        if "hint" not in df_base.columns:
            df_base["hint"] = ""

        df_view = df_base.copy()

        column_order = build_column_order("", df_view.columns.tolist())
        if "attribute_set_id" in df_view.columns:
            if "attribute_set_id" not in column_order:
                column_order.append("attribute_set_id")
            if "attribute set" in column_order:
                column_order = [
                    col for col in column_order if col != "attribute_set_id"
                ]
                insert_idx = column_order.index("attribute set") + 1
                column_order.insert(insert_idx, "attribute_set_id")
        df_view = df_view[column_order]
        options = list(attribute_sets.keys())
        options.extend(
            name
            for name in df_base["attribute set"].dropna().unique()
            if name not in attribute_sets
        )
        options = list(dict.fromkeys(options))


        st.header("Assign the product type (attribute set).")
        st.caption(
            "For each item, choose which product type (attribute set) it belongs to. "
            "This lets us show the correct attributes later. "
            "Edit the **Product Type (attribute set)** column: replace **default** with the right value from the dropdown."
        )

        step2_active = st.session_state.get("show_attributes_trigger", False)

        if not step2_active:
            step1_column_config = {
                "sku": st.column_config.Column(
                    label="SKU", disabled=True, width="small"
                ),
                "name": st.column_config.TextColumn(
                    label="Name", disabled=False, width="medium"
                ),
                "attribute set": st.column_config.SelectboxColumn(
                    label="Product Type (attribute set)",
                    help="Change attribute set",
                    options=options,
                    required=True,
                ),
                "attribute_set_id": st.column_config.NumberColumn(
                    label="Attribute Set ID",
                    help="Change attribute set identifier",
                    step=1,
                ),
                "hint": st.column_config.TextColumn("Hint"),
                "created_at": st.column_config.DatetimeColumn("Created At", disabled=True),
            }
            st.session_state["step1_column_config_cache"] = step1_column_config
            st.session_state["step1_disabled_cols_cache"] = [
                "sku",
                "created_at",
            ]
            col_cfg, disabled_cols = _build_column_config_for_step1_like(step="step1")
            if "sku" in df_view.columns:
                col_cfg["sku"] = st.column_config.Column(
                    label="SKU", disabled=True, width="small"
                )
            if "name" in df_view.columns:
                col_cfg["name"] = st.column_config.TextColumn(
                    label="Name", disabled=False, width="medium"
                )
            if "attribute_set_id" in df_view.columns:
                col_cfg["attribute_set_id"] = st.column_config.NumberColumn(
                    label="Attribute Set ID",
                    help="Change attribute set identifier",
                    step=1,
                )
            disabled_cols = [
                col
                for col in disabled_cols
                if col != "name" and col in df_view.columns
            ]
            if "attribute set" in col_cfg:
                cfg = col_cfg["attribute set"]
                if hasattr(cfg, "label"):
                    cfg.label = "Product Type (attribute set)"
                elif hasattr(cfg, "_label"):
                    cfg._label = "Product Type (attribute set)"
            if "attribute_set" in col_cfg:
                cfg = col_cfg["attribute_set"]
                if hasattr(cfg, "label"):
                    cfg.label = "Product Type (attribute set)"
                elif hasattr(cfg, "_label"):
                    cfg._label = "Product Type (attribute set)"
            if "price" in df_view.columns:
                col_cfg["price"] = st.column_config.NumberColumn(
                    label="Price", disabled=True
                )
            column_config_view = {
                key: value for key, value in col_cfg.items() if key in df_view.columns
            }
            edited_df = st.data_editor(
                df_view,
                column_config=column_config_view,
                disabled=disabled_cols,
                column_order=column_order,
                use_container_width=True,
                num_rows="fixed",
                key=editor_key,
            )

            go_attrs = st.button(
                "🔎 Show attributes",
                key="btn_show_attrs",
                help="Build attribute editor for selected items",
            )
            if isinstance(edited_df, pd.DataFrame) and go_attrs:
                updated_df = df_base.copy()
                for column in edited_df.columns:
                    updated_df[column] = edited_df[column]
                if "attribute_set_id" in edited_df.columns and "attribute set" in updated_df.columns:
                    id_lookup: dict[int, str] = {}
                    for name, attr_id in attribute_sets.items():
                        try:
                            id_lookup[int(attr_id)] = name
                        except (TypeError, ValueError):
                            continue

                    def _coerce_attr_id(value):
                        if value is None:
                            return None
                        if pd.isna(value):
                            return None
                        if isinstance(value, str):
                            if not value.strip():
                                return None
                            try:
                                return int(float(value))
                            except (TypeError, ValueError):
                                return value.strip()
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return value

                    for idx, raw_value in edited_df["attribute_set_id"].items():
                        coerced = _coerce_attr_id(raw_value)
                        if coerced is None:
                            continue
                        if isinstance(coerced, int):
                            name = id_lookup.get(coerced)
                            updated_df.loc[idx, "attribute set"] = (
                                name if name else str(coerced)
                            )
                        else:
                            updated_df.loc[idx, "attribute set"] = str(coerced)
                st.session_state[df_edited_key] = updated_df
                st.session_state["show_attributes_trigger"] = True
                st.session_state["_go_step2_requested"] = True
                _reset_step2_state()
                st.rerun()
        else:
            st.header("Filling in attributes.")
            if st.session_state.get("_go_step2_requested"):
                st.session_state["_go_step2_requested"] = False

            if df_edited_key not in st.session_state or df_original_key not in st.session_state:
                st.warning("No updated products.")
                st.session_state["show_attributes_trigger"] = False
            else:
                df_new = st.session_state[df_edited_key].copy()
                df_old = st.session_state[df_original_key].copy()

                required_cols = {"sku", "attribute set"}
                if not (
                    required_cols.issubset(df_new.columns)
                    and required_cols.issubset(df_old.columns)
                ):
                    st.warning("No updated products.")
                    st.session_state["show_attributes_trigger"] = False
                else:
                    df_new_idx = df_new.set_index("sku")
                    df_old_idx = df_old.set_index("sku")
                    common_skus = df_new_idx.index.intersection(df_old_idx.index)
                    if common_skus.empty:
                        st.warning("No updated products.")
                        st.session_state["show_attributes_trigger"] = False
                    else:
                        df_new_common = df_new_idx.loc[common_skus]
                        attr_sets_name_to_id = st.session_state.get(
                            attribute_sets_key, {}
                        )

                        def _coerce_set_id(val):
                            try:
                                if val is None or (
                                    isinstance(val, float) and pd.isna(val)
                                ):
                                    raise ValueError
                                return int(val)
                            except Exception:
                                name = str(val or "").strip()
                                if not name:
                                    return None
                                if name.casefold() == "default":
                                    return 4
                                raw = attr_sets_name_to_id.get(name)
                                try:
                                    return int(raw) if raw is not None else None
                                except Exception:
                                    return None

                        if "attribute_set_id" in df_new_common.columns:
                            eff_ids = df_new_common["attribute_set_id"].apply(
                                _coerce_set_id
                            )
                        else:
                            eff_ids = df_new_common["attribute set"].apply(
                                _coerce_set_id
                            )

                        df_changed = df_new_common.loc[eff_ids.ne(4)].reset_index()

                        if df_changed.empty:
                            st.warning("No updated products.")
                            st.session_state["show_attributes_trigger"] = False
                        else:
                            api_base = st.session_state.get("ai_api_base")
                            attr_sets_map = st.session_state.get("ai_attr_sets_map")
                            setup_failed = False

                            html_error = False
                            try:
                                if not api_base:
                                    with st.spinner("🔌 Connecting to Magento…"):
                                        api_base = probe_api_base(session, base_url)
                                    st.session_state["ai_api_base"] = api_base
                                if not attr_sets_map and api_base:
                                    with st.spinner("📚 Loading attribute sets…"):
                                        attr_sets_map = get_attribute_sets_map(
                                            session, api_base
                                        )
                                    st.session_state["ai_attr_sets_map"] = attr_sets_map
                            except HTMLResponseError as exc:
                                st.error(str(exc))
                                html_error = True
                            except Exception as exc:  # pragma: no cover - UI interaction
                                st.warning(
                                    f"Failed to prepare Magento connection: {exc}"
                                )
                                setup_failed = True

                            if html_error:
                                st.session_state["show_attributes_trigger"] = False
                                st.stop()

                            if setup_failed or not api_base or not attr_sets_map:
                                st.session_state["show_attributes_trigger"] = False
                            else:
                                status2 = st.status(
                                    "Building attribute editor…", expanded=True
                                )
                                progress = st.progress(0)

                                def _pupdate(pct: int, msg: str) -> None:
                                    clamped = max(0, min(int(pct), 100))
                                    progress.progress(clamped)
                                    status2.update(label=msg)

                                _pupdate(5, "Collecting selected sets…")
                                selected_set_ids: list[int] = []
                                step2_state = _ensure_step2_state()
                                step2_state.setdefault("staged", {})

                                meta_cache: AttributeMetaCache | None = step2_state.get(
                                    "meta_cache"
                                )
                                if not isinstance(meta_cache, AttributeMetaCache):
                                    meta_cache = AttributeMetaCache(session, api_base)
                                    step2_state["meta_cache"] = meta_cache

                                categories_meta = step2_state.get("categories_meta")
                                if not isinstance(categories_meta, dict):
                                    categories_meta = {}

                                if not categories_meta.get("options"):
                                    try:
                                        with st.spinner("📂 Loading categories…"):
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
                                            ) or {}
                                        st.session_state["step2_categories_failed"] = False
                                    except Exception as exc:  # pragma: no cover - UI interaction
                                        st.warning(
                                            f"Failed to load categories: {exc}"
                                        )
                                        categories_meta = (
                                            meta_cache.get("categories")
                                            if isinstance(meta_cache, AttributeMetaCache)
                                            else {}
                                        ) or {}
                                        st.session_state["step2_categories_failed"] = True
                                else:
                                    st.session_state["step2_categories_failed"] = False

                                normalized_options: list[dict[str, object]] = []
                                seen_category_ids: set[str] = set()
                                raw_options = categories_meta.get("options") or []
                                if not isinstance(raw_options, list):
                                    try:
                                        raw_options = list(raw_options)
                                    except TypeError:
                                        raw_options = []
                                for opt in raw_options:
                                    if not isinstance(opt, dict):
                                        continue
                                    label_raw = opt.get("label")
                                    value_raw = opt.get("value")
                                    label = str(label_raw or "").strip()
                                    if not label:
                                        continue
                                    try:
                                        value_int = int(value_raw)
                                    except (TypeError, ValueError):
                                        continue
                                    value_str = str(value_int)
                                    if value_str in seen_category_ids:
                                        continue
                                    seen_category_ids.add(value_str)
                                    normalized_options.append(
                                        {"label": label, "value": value_int}
                                    )
                                normalized_options.sort(
                                    key=lambda item: item["label"].casefold()
                                )

                                values_to_labels = categories_meta.get("values_to_labels")
                                if not isinstance(values_to_labels, dict):
                                    values_to_labels = {}
                                labels_to_values = categories_meta.get("labels_to_values")
                                if not isinstance(labels_to_values, dict):
                                    labels_to_values = {}
                                options_map = categories_meta.get("options_map")
                                if not isinstance(options_map, dict):
                                    options_map = {}

                                for opt in normalized_options:
                                    label = opt["label"]
                                    value_int = opt["value"]
                                    value_str = str(value_int)
                                    values_to_labels[value_str] = label
                                    normalized_label = _norm_label(label)
                                    if normalized_label:
                                        labels_to_values[normalized_label] = value_str
                                    labels_to_values[label] = value_str
                                    labels_to_values[label.casefold()] = value_str
                                    options_map[label] = value_int
                                    options_map[label.casefold()] = value_int
                                    options_map[value_str] = value_int

                                categories_meta["options"] = normalized_options
                                categories_meta["values_to_labels"] = values_to_labels
                                categories_meta["labels_to_values"] = labels_to_values
                                categories_meta["options_map"] = options_map

                                step2_state["categories_meta"] = categories_meta

                                if isinstance(meta_cache, AttributeMetaCache):
                                    try:
                                        meta_cache.set_static("categories", categories_meta)
                                    except Exception:
                                        pass

                                if not step2_state["dfs"]:
                                    (
                                        dfs_by_set,
                                        column_configs,
                                        disabled_cols_map,
                                        meta_cache,
                                        row_meta_map,
                                        set_names,
                                        ai_suggestions_map,
                                        ai_errors,
                                        ai_log_data,
                                    ) = build_attributes_df(
                                        df_changed,
                                        session,
                                        api_base,
                                        attribute_sets,
                                        attr_sets_map,
                                        meta_cache,
                                        ai_api_key=st.session_state.get("ai_api_key", ""),
                                        ai_model=(
                                            st.session_state.get("ai_model_resolved")
                                            or st.session_state.get("ai_model")
                                        ),
                                        # 👇 bind SKU progress to the shared Step 2 progress bar
                                        progress_callback=lambda idx, total, msg: _pupdate(
                                            20 + int(15 * idx / max(total, 1)),
                                            msg,
                                        ),
                                    )
                                    step2_state["ai_suggestions"] = (
                                        ai_suggestions_map or {}
                                    )
                                    step2_state["ai_errors"] = ai_errors
                                    step2_state["ai_log_data"] = ai_log_data or {"errors": []}
                                    if ai_errors:
                                        preview = "\n".join(
                                            f"- {msg}" for msg in ai_errors[:5]
                                        )
                                        if len(ai_errors) > 5:
                                            preview += (
                                                f"\n… {len(ai_errors) - 5} more items"
                                            )
                                        st.warning(
                                            "AI suggestions failed for some products:\n"
                                            + preview
                                        )
                                    selected_set_ids = sorted(dfs_by_set.keys())
                                    total_sets = len(selected_set_ids) or 1

                                    for i, set_id in enumerate(selected_set_ids, start=1):
                                        df_set = dfs_by_set.get(set_id)
                                        if not isinstance(df_set, pd.DataFrame):
                                            continue

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
                                        if "categories" in wide_df.columns:
                                            wide_df["categories"] = wide_df["categories"].apply(
                                                _cat_to_ids
                                            )
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        if (
                                            GENERATE_DESCRIPTION_COLUMN
                                            not in wide_df.columns
                                        ):
                                            wide_df[GENERATE_DESCRIPTION_COLUMN] = True
                                        for column in wide_df.columns:
                                            wide_df[column] = wide_df[column].astype(object)
                                        baseline_df = wide_df.copy(deep=True)
                                        step2_state["wide"][set_id] = baseline_df.copy(
                                            deep=True
                                        )
                                        step2_state["wide_orig"][set_id] = baseline_df.copy(
                                            deep=True
                                        )
                                        step2_state["wide_synced"][set_id] = baseline_df.copy(
                                            deep=True
                                        )
                                        attr_cols = [
                                            col
                                            for col in wide_df.columns
                                            if col
                                            not in (
                                                "sku",
                                                "name",
                                                GENERATE_DESCRIPTION_COLUMN,
                                            )
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
                                        effective_codes: list[str] = []
                                        pct_resolve = 10 + int(
                                            20 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_resolve,
                                            f"Resolving codes for set {set_id} ({i}/{total_sets})…",
                                        )
                                        if cacheable:
                                            resolved_map = meta_cache.resolve_codes_in_set(
                                                set_id, ["brand", "condition"]
                                            )
                                            seen_effective: set[str] = set()
                                            for code in attr_cols:
                                                if not isinstance(code, str):
                                                    continue
                                                effective = resolved_map.get(code, code)
                                                if effective and effective not in seen_effective:
                                                    seen_effective.add(effective)
                                                    effective_codes.append(effective)
                                        pct_meta = 30 + int(
                                            10 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_meta,
                                            f"Fetching metadata (set {set_id})…",
                                        )
                                        if cacheable:
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
                                        categories_meta_for_ai = step2_state.get(
                                            "categories_meta"
                                        )
                                        if isinstance(categories_meta_for_ai, dict):
                                            meta_map["categories"] = dict(
                                                categories_meta_for_ai
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "DEBUG categories before AI:",
                                                    meta_map.get("categories"),
                                                )

                                        _apply_categories_fallback(meta_map)
                                        ai_cells_map = step2_state.setdefault(
                                            "ai_cells", {}
                                        )
                                        pipeline_snapshots = step2_state.setdefault(
                                            "pipeline_snapshots", {}
                                        )
                                        suggestions_for_set = (
                                            step2_state.get("ai_suggestions") or {}
                                        ).get(set_id)
                                        if (
                                            suggestions_for_set
                                            and set_id not in ai_cells_map
                                            and isinstance(
                                                step2_state["wide"].get(set_id), pd.DataFrame
                                            )
                                        ):
                                            df_before_ai = step2_state["wide"].get(set_id)
                                            snapshot_entry = pipeline_snapshots.setdefault(
                                                set_id,
                                                {},
                                            )
                                            if isinstance(df_before_ai, pd.DataFrame):
                                                snapshot_entry[
                                                    "wide_before_ai"
                                                ] = df_before_ai.copy(deep=True)
                                            (
                                                updated_df,
                                                filled_cells,
                                                ai_log_entries,
                                            ) = (
                                                _apply_ai_suggestions_to_wide(
                                                    step2_state["wide"].get(set_id),
                                                    suggestions_for_set,
                                                    meta_map,
                                                    meta_cache=meta_cache,
                                                    session=session,
                                                    api_base=api_base,
                                                )
                                            )
                                            if isinstance(updated_df, pd.DataFrame):
                                                step2_state["wide"][set_id] = updated_df
                                                if isinstance(updated_df, pd.DataFrame):
                                                    snapshot_entry[
                                                        "wide_after_ai"
                                                    ] = updated_df.copy(deep=True)
                                                if filled_cells:
                                                    ai_cells_map[set_id] = filled_cells
                                                ai_logs_map = step2_state.setdefault(
                                                    "ai_logs", {}
                                                )
                                                if ai_log_entries:
                                                    ai_logs_map[set_id] = ai_log_entries
                                                else:
                                                    ai_logs_map.pop(set_id, None)
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

                                        set_label = (
                                            step2_state.get("set_names", {}).get(
                                                set_id, str(set_id)
                                            )
                                        )
                                        step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                            meta_for_debug,
                                            sample_df=step2_state["wide"].get(set_id),
                                            set_id=set_id,
                                            set_name=set_label,
                                        )

                                step2_state["meta_cache"] = meta_cache

                                if not selected_set_ids:
                                    selected_set_ids = sorted(step2_state["dfs"].keys())

                                for set_id, df0 in step2_state["dfs"].items():
                                    if "value" in df0.columns:
                                        df0["value"] = df0["value"].astype(object)

                                if step2_state["dfs"] and not step2_state["wide"]:
                                    ordered_sets = (
                                        selected_set_ids
                                        or sorted(step2_state["dfs"].keys())
                                    )
                                    total_sets = len(ordered_sets) or 1
                                    for i, set_id in enumerate(ordered_sets, start=1):
                                        df_existing = step2_state["dfs"].get(set_id)
                                        if not isinstance(df_existing, pd.DataFrame):
                                            continue
                                        wide_df = make_wide(df_existing)
                                        if "categories" in wide_df.columns:
                                            wide_df["categories"] = wide_df["categories"].apply(
                                                _cat_to_ids
                                            )
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        if (
                                            GENERATE_DESCRIPTION_COLUMN
                                            not in wide_df.columns
                                        ):
                                            wide_df[GENERATE_DESCRIPTION_COLUMN] = True
                                        for column in wide_df.columns:
                                            wide_df[column] = wide_df[column].astype(object)
                                        baseline_df = wide_df.copy(deep=True)
                                        step2_state["wide"][set_id] = baseline_df.copy(
                                            deep=True
                                        )
                                        step2_state["wide_orig"][set_id] = baseline_df.copy(
                                            deep=True
                                        )
                                        if set_id not in step2_state["wide_synced"]:
                                            step2_state["wide_synced"][set_id] = (
                                                baseline_df.copy(deep=True)
                                            )
                                        attr_cols = [
                                            col
                                            for col in wide_df.columns
                                            if col
                                            not in (
                                                "sku",
                                                "name",
                                                GENERATE_DESCRIPTION_COLUMN,
                                            )
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
                                        effective_codes: list[str] = []
                                        pct_resolve = 10 + int(
                                            20 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_resolve,
                                            f"Resolving codes for set {set_id} ({i}/{total_sets})…",
                                        )
                                        if cacheable:
                                            resolved_map = meta_cache.resolve_codes_in_set(
                                                set_id, ["brand", "condition"]
                                            )
                                            seen_effective: set[str] = set()
                                            for code in attr_cols:
                                                if not isinstance(code, str):
                                                    continue
                                                effective = resolved_map.get(code, code)
                                                if effective and effective not in seen_effective:
                                                    seen_effective.add(effective)
                                                    effective_codes.append(effective)
                                        pct_meta = 30 + int(
                                            10 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_meta,
                                            f"Fetching metadata (set {set_id})…",
                                        )
                                        if cacheable:
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
                                        categories_meta_for_ai = step2_state.get(
                                            "categories_meta"
                                        )
                                        if isinstance(categories_meta_for_ai, dict):
                                            meta_map["categories"] = dict(
                                                categories_meta_for_ai
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "DEBUG categories before AI:",
                                                    meta_map.get("categories"),
                                                )

                                        _apply_categories_fallback(meta_map)
                                        ai_cells_map = step2_state.setdefault(
                                            "ai_cells", {}
                                        )
                                        pipeline_snapshots = step2_state.setdefault(
                                            "pipeline_snapshots", {}
                                        )
                                        suggestions_for_set = (
                                            step2_state.get("ai_suggestions") or {}
                                        ).get(set_id)
                                        if (
                                            suggestions_for_set
                                            and set_id not in ai_cells_map
                                            and isinstance(
                                                step2_state["wide"].get(set_id), pd.DataFrame
                                            )
                                        ):
                                            snapshot_entry = pipeline_snapshots.setdefault(
                                                set_id,
                                                {},
                                            )
                                            df_before_ai = step2_state["wide"].get(set_id)
                                            if isinstance(df_before_ai, pd.DataFrame):
                                                snapshot_entry[
                                                    "wide_before_ai"
                                                ] = df_before_ai.copy(deep=True)
                                            (
                                                updated_df,
                                                filled_cells,
                                                ai_log_entries,
                                            ) = (
                                                _apply_ai_suggestions_to_wide(
                                                    step2_state["wide"].get(set_id),
                                                    suggestions_for_set,
                                                    meta_map,
                                                    meta_cache=meta_cache,
                                                    session=session,
                                                    api_base=api_base,
                                                )
                                            )
                                            if isinstance(updated_df, pd.DataFrame):
                                                step2_state["wide"][set_id] = updated_df
                                                snapshot_entry["wide_after_ai"] = (
                                                    updated_df.copy(deep=True)
                                                )
                                                if filled_cells:
                                                    ai_cells_map[set_id] = filled_cells
                                                ai_logs_map = step2_state.setdefault(
                                                    "ai_logs", {}
                                                )
                                                if ai_log_entries:
                                                    ai_logs_map[set_id] = ai_log_entries
                                                else:
                                                    ai_logs_map.pop(set_id, None)
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

                                        set_label = (
                                            step2_state.get("set_names", {}).get(
                                                set_id, str(set_id)
                                            )
                                        )
                                        step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                            meta_for_debug,
                                            sample_df=step2_state["wide"].get(set_id),
                                            set_id=set_id,
                                            set_name=set_label,
                                        )

                                if step2_state["wide"]:
                                    selected_set_ids = sorted(
                                        set(selected_set_ids)
                                        | set(step2_state["wide"].keys())
                                    )

                                _pupdate(40, "Fetching metadata from Magento…")

                                categories_meta = step2_state.get("categories_meta")
                                if not isinstance(categories_meta, dict):
                                    categories_meta = {}

                                _pupdate(45, "Preparing categories (full tree)…")

                                st.session_state["show_attrs"] = bool(
                                    step2_state["wide"]
                                )

                                completed = False
                                if not step2_state["wide"]:
                                    st.warning("No attributes to display.")
                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )
                                    completed = True
                                elif st.session_state.get("show_attrs"):
                                    if st.session_state.get("ai_rerun_requested", False):
                                        st.session_state["ai_rerun_requested"] = False
                                        suggestions_map = (
                                            step2_state.get("ai_suggestions") or {}
                                        )
                                        wide_map = step2_state.get("wide", {})
                                        wide_meta_map = step2_state.get("wide_meta", {})
                                        ai_cells_map = step2_state.setdefault(
                                            "ai_cells", {}
                                        )
                                        ai_logs_map = step2_state.setdefault(
                                            "ai_logs", {}
                                        )
                                        pipeline_snapshots = step2_state.setdefault(
                                            "pipeline_snapshots", {}
                                        )
                                        visible_sets = (
                                            selected_set_ids
                                            or sorted(step2_state.get("wide", {}).keys())
                                        )
                                        for current_set_id in visible_sets:
                                            wide_current = wide_map.get(current_set_id)
                                            suggestions_for_set = suggestions_map.get(
                                                current_set_id
                                            )
                                            if not (
                                                isinstance(wide_current, pd.DataFrame)
                                                and suggestions_for_set
                                            ):
                                                continue
                                            meta_for_set = (
                                                wide_meta_map.get(current_set_id, {})
                                                if isinstance(wide_meta_map, dict)
                                                else {}
                                            )
                                            snapshot_entry = pipeline_snapshots.setdefault(
                                                current_set_id,
                                                {},
                                            )
                                            snapshot_entry["wide_before_ai"] = wide_current.copy(
                                                deep=True
                                            )
                                            (
                                                updated_df,
                                                filled_cells,
                                                ai_log_entries,
                                            ) = _apply_ai_suggestions_to_wide(
                                                wide_current,
                                                suggestions_for_set,
                                                meta_for_set,
                                                meta_cache=meta_cache,
                                                session=session,
                                                api_base=api_base,
                                            )
                                            if not isinstance(updated_df, pd.DataFrame):
                                                continue
                                            step2_state["wide"][current_set_id] = updated_df
                                            snapshot_entry["wide_after_ai"] = updated_df.copy(
                                                deep=True
                                            )
                                            if filled_cells:
                                                ai_cells_map[current_set_id] = filled_cells
                                            else:
                                                ai_cells_map.pop(current_set_id, None)
                                            if ai_log_entries:
                                                ai_logs_map[current_set_id] = ai_log_entries
                                            else:
                                                ai_logs_map.pop(current_set_id, None)
                                            df_view_rerun = _coerce_for_ui(
                                                updated_df, meta_for_set
                                            )
                                            if isinstance(df_view_rerun, pd.DataFrame):
                                                snapshot_entry[
                                                    "ui_before_editor"
                                                ] = df_view_rerun.copy(deep=True)
                                                display_df = df_view_rerun.copy()
                                                if "attribute_set_id" in display_df.columns:
                                                    display_df = display_df.drop(
                                                        columns=["attribute_set_id"]
                                                    )
                                                _sync_ai_highlight_state(
                                                    step2_state,
                                                    current_set_id,
                                                    display_df,
                                                )
                                        st.rerun()
                                    wide_meta = step2_state.setdefault("wide_meta", {})
                                    categories_meta = step2_state.get("categories_meta", {})
                                    if not isinstance(categories_meta, dict):
                                        categories_meta = {}
                                    labels_all = [
                                        opt.get("label")
                                        for opt in (categories_meta.get("options") or [])
                                        if isinstance(opt, dict) and opt.get("label")
                                    ]
                                    total_rows = sum(
                                        len(df)
                                        for df in step2_state["wide"].values()
                                        if isinstance(df, pd.DataFrame)
                                    )
                                    normalized_rows = 0
                                    if total_rows == 0:
                                        _pupdate(70, "Normalizing values…")
                                    config_stage_logged = False
                                    render_counter = 0
                                    total_sets_render = len(selected_set_ids) or 1
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

                                        if isinstance(categories_meta, dict):
                                            meta_map["categories"] = dict(categories_meta)
                                            if DEBUG:
                                                st.write(
                                                    "DEBUG categories before UI:",
                                                    meta_map.get("categories"),
                                                )

                                        _apply_categories_fallback(meta_map)
                                        pipeline_snapshots = step2_state.setdefault(
                                            "pipeline_snapshots", {}
                                        )
                                        df_ref = _coerce_for_ui(wide_df, meta_map)
                                        if isinstance(df_ref, pd.DataFrame):
                                            missing_columns = [
                                                col
                                                for col in wide_df.columns
                                                if col not in df_ref.columns
                                                and col != "attribute_set_id"
                                            ]
                                            if missing_columns:
                                                trace(
                                                    {
                                                        "where": "ui:missing_col",
                                                        "set_id": set_id,
                                                        "codes": missing_columns,
                                                    }
                                                )
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
                                            if isinstance(df_ref, pd.DataFrame)
                                            and col in df_ref.columns
                                        ]
                                        if string_cols:
                                            df_ref[string_cols] = df_ref[
                                                string_cols
                                            ].astype("string")

                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "sku" in df_ref.columns
                                            and total_rows > 0
                                        ):
                                            sku_values = df_ref["sku"].tolist()
                                            for idx, sku in enumerate(sku_values, start=1):
                                                normalized_rows += 1
                                                if (
                                                    idx == len(sku_values)
                                                    or idx % 25 == 0
                                                    or normalized_rows == total_rows
                                                ):
                                                    pct = 45 + int(
                                                        25
                                                        * normalized_rows
                                                        / max(total_rows, 1)
                                                    )
                                                    _pupdate(
                                                        pct,
                                                        f"Normalizing values… SKU {sku} "
                                                        f"({normalized_rows}/{total_rows})",
                                                    )

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

                                        if not config_stage_logged:
                                            _pupdate(75, "Composing column configs…")
                                            config_stage_logged = True

                                        set_label = (
                                            step2_state.get("set_names", {}).get(
                                                set_id, str(set_id)
                                            )
                                        )
                                        column_config = build_wide_colcfg(
                                            meta_map,
                                            sample_df=df_ref,
                                            set_id=set_id,
                                            set_name=set_label,
                                        )
                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "categories" in df_ref.columns
                                        ):
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

                                        if not isinstance(df_ref, pd.DataFrame):
                                            continue

                                        attribute_set_names = step2_state.get(
                                            "set_names", {}
                                        )

                                        for current_set_id, group in df_ref.groupby(
                                            "attribute_set_id"
                                        ):
                                            render_counter += 1
                                            pct_render = 80 + int(
                                                15
                                                * render_counter
                                                / max(total_sets_render, 1)
                                            )
                                            _pupdate(
                                                pct_render,
                                                f"Rendering table: set {current_set_id} "
                                                f"({render_counter}/{total_sets_render})…",
                                            )

                                            raw_title = step2_state["set_names"].get(
                                                current_set_id, str(current_set_id)
                                            )
                                            set_name = (
                                                raw_title
                                                if isinstance(raw_title, str)
                                                and raw_title.strip()
                                                else str(current_set_id)
                                            )
                                            icon = _attr_set_icon(set_name)
                                            st.subheader(
                                                f"{icon} {set_name} ({len(group)} items)"
                                            )

                                            group = group.reset_index(drop=True)
                                            df_view = group.copy()
                                            if "attribute_set_id" in df_view.columns:
                                                df_view = df_view.drop(
                                                    columns=["attribute_set_id"]
                                                )
                                            if (
                                                GENERATE_DESCRIPTION_COLUMN
                                                not in df_view.columns
                                            ):
                                                df_view[GENERATE_DESCRIPTION_COLUMN] = True

                                            column_order = build_column_order(
                                                set_name,
                                                df_view.columns.tolist(),
                                            )

                                            set_name_key = (
                                                set_name
                                                if isinstance(set_name, str)
                                                else str(set_name)
                                            )
                                            allowed_for_set = (
                                                set(BASE_FIRST)
                                                | set(ALWAYS_ATTRS)
                                                | set(SET_ATTRS.get(set_name_key, []))
                                                | set(
                                                    SET_ATTRS.get(
                                                        str(current_set_id), []
                                                    )
                                                )
                                            )
                                            allowed_for_set.add(GENERATE_DESCRIPTION_COLUMN)

                                            allowed_columns = [
                                                col
                                                for col in df_view.columns
                                                if col in allowed_for_set
                                            ]
                                            df_view = df_view[allowed_columns]
                                            column_order = [
                                                c
                                                for c in column_order
                                                if c in allowed_for_set
                                            ]
                                            if column_order:
                                                df_view = df_view[column_order]
                                            else:
                                                column_order = (
                                                    df_view.columns.tolist()
                                                )

                                            snapshot_entry = pipeline_snapshots.setdefault(
                                                current_set_id,
                                                {},
                                            )
                                            if isinstance(df_view, pd.DataFrame):
                                                snapshot_entry[
                                                    "ui_before_editor"
                                                ] = df_view.copy(deep=True)
                                                trace(
                                                    {
                                                        "where": "ui:df_before_editor",
                                                        "set_id": current_set_id,
                                                        "cols": df_view.columns.tolist(),
                                                        "sample": df_view.head(3).to_dict(
                                                            orient="records"
                                                        ),
                                                    }
                                                )

                                            set_name_norm = set_name_key.strip().lower()
                                            current_set_id_value: int | None = None
                                            try:
                                                current_set_id_value = int(current_set_id)
                                            except (TypeError, ValueError):
                                                current_set_id_value = None
                                            is_electric = False
                                            if (
                                                current_set_id_value is not None
                                                and current_set_id_value in ELECTRIC_SET_IDS
                                            ):
                                                is_electric = True
                                            elif set_name_norm.startswith("electric"):
                                                is_electric = True

                                            column_config_final = dict(column_config or {})
                                            if "sku" in df_view.columns:
                                                column_config_final[
                                                    "sku"
                                                ] = st.column_config.Column(
                                                    label="SKU",
                                                    disabled=True,
                                                    width="small",
                                                )
                                            if "name" in df_view.columns:
                                                column_config_final[
                                                    "name"
                                                ] = st.column_config.TextColumn(
                                                    label="Name",
                                                    disabled=False,
                                                    width="medium",
                                                )
                                            if "price" in df_view.columns:
                                                column_config_final[
                                                    "price"
                                                ] = st.column_config.NumberColumn(
                                                    label="Price", disabled=True
                                                )
                                            if is_electric and "guitarstylemultiplechoice" in column_config_final:
                                                cfg = column_config_final[
                                                    "guitarstylemultiplechoice"
                                                ]
                                                if hasattr(cfg, "label"):
                                                    cfg.label = "Guitar style"
                                                elif hasattr(cfg, "_label"):
                                                    cfg._label = "Guitar style"

                                            editor_df = st.data_editor(
                                                df_view,
                                                key=f"editor_set_{current_set_id}",
                                                column_config={
                                                    key: value
                                                    for key, value in column_config_final.items()
                                                    if key in df_view.columns
                                                },
                                                column_order=column_order,
                                                use_container_width=True,
                                                hide_index=True,
                                                num_rows="fixed",
                                            )

                                            if isinstance(editor_df, pd.DataFrame):
                                                for column in editor_df.columns:
                                                    editor_df[column] = editor_df[
                                                        column
                                                    ].astype(object)
                                                st.session_state["step2_products"] = (
                                                    editor_df.copy(deep=True)
                                                )
                                                _sync_ai_highlight_state(
                                                    step2_state,
                                                    current_set_id,
                                                    editor_df,
                                                )
                                                base_synced = (
                                                    step2_state.get("wide_synced", {})
                                                    .get(current_set_id)
                                                )
                                                if (
                                                    isinstance(
                                                        base_synced, pd.DataFrame
                                                    )
                                                    and "attribute_set_id"
                                                    in base_synced.columns
                                                ):
                                                    base_synced = base_synced.drop(
                                                        columns=["attribute_set_id"]
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

                                            ai_cells_for_set = (
                                                step2_state.get("ai_cells", {}).get(
                                                    current_set_id
                                                )
                                                or []
                                            )
                                            _render_ai_highlight(
                                                df_view,
                                                column_order,
                                                ai_cells_for_set,
                                            )

                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )
                                    completed = True
                                if not completed:
                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )

                                if st.session_state.get("show_attrs"):
                                    trace_events = st.session_state.get(
                                        "_trace_events", []
                                    )
                                    payload_events = [
                                        event
                                        for event in trace_events
                                        if isinstance(event, dict)
                                        and event.get("where") == "save:payload"
                                    ]
                                    ai_log_payload = _collect_ai_suggestions_log(
                                        step2_state.get("ai_suggestions"),
                                        step2_state.get("ai_cells"),
                                        step2_state.get("ai_logs"),
                                        step2_state.get("ai_log_data"),
                                    )
                                    with st.expander("🧠 AI suggestions log"):
                                        if ai_log_payload:
                                            st.json(ai_log_payload)
                                        else:
                                            st.caption(
                                                "AI не вносил подсказки в этой сессии."
                                            )

                                    with st.expander("🧬 Data pipeline snapshots"):
                                        tab_ai, tab_stage, tab_payload = st.tabs(
                                            [
                                                "AI → Wide",
                                                "Staged vs Synced",
                                                "Payload log",
                                            ]
                                        )
                                        pipeline_snapshots = step2_state.get(
                                            "pipeline_snapshots", {}
                                        )
                                        set_names = step2_state.get("set_names", {})
                                        with tab_ai:
                                            if not pipeline_snapshots:
                                                st.caption("Пока нет снапшотов AI.")
                                            else:
                                                for snap_set_id in sorted(
                                                    pipeline_snapshots.keys()
                                                ):
                                                    snap_entry = pipeline_snapshots.get(
                                                        snap_set_id, {}
                                                    )
                                                    set_label = set_names.get(
                                                        snap_set_id, str(snap_set_id)
                                                    )
                                                    st.markdown(
                                                        f"**Set {snap_set_id} — {set_label}**"
                                                    )
                                                    before_ai = snap_entry.get(
                                                        "wide_before_ai"
                                                    )
                                                    after_ai = snap_entry.get(
                                                        "wide_after_ai"
                                                    )
                                                    ui_before = snap_entry.get(
                                                        "ui_before_editor"
                                                    )
                                                    if isinstance(before_ai, pd.DataFrame):
                                                        st.write(
                                                            "DF BEFORE AI",
                                                            before_ai.head(10),
                                                        )
                                                    else:
                                                        st.caption(
                                                            "Нет данных DF BEFORE AI"
                                                        )
                                                    if isinstance(after_ai, pd.DataFrame):
                                                        for col in after_ai.columns:
                                                            if after_ai[col].apply(
                                                                lambda x: isinstance(
                                                                    x, (list, dict)
                                                                )
                                                            ).any():
                                                                after_ai[col] = after_ai[
                                                                    col
                                                                ].astype(str)
                                                        try:
                                                            st.write(
                                                                "DF AFTER AI",
                                                                after_ai.head(10),
                                                            )
                                                        except Exception as e:
                                                            st.error(
                                                                f"Could not render dataframe: {e}"
                                                            )
                                                            st.text(
                                                                after_ai.head(10).to_string()
                                                            )
                                                    else:
                                                        st.caption(
                                                            "Нет данных DF AFTER AI"
                                                        )
                                                    if isinstance(ui_before, pd.DataFrame):
                                                        st.write(
                                                            "DF BEFORE EDITOR",
                                                            ui_before.head(10),
                                                        )
                                                    st.markdown("---")
                                        with tab_stage:
                                            staged_map = step2_state.get("staged", {})
                                            synced_map = step2_state.get("wide_synced", {})
                                            all_sets = sorted(
                                                set(staged_map.keys())
                                                | set(synced_map.keys())
                                            )
                                            if not all_sets:
                                                st.caption("Нет staged изменений.")
                                            for snap_set_id in all_sets:
                                                staged_df = staged_map.get(snap_set_id)
                                                if staged_df is None:
                                                    staged_df = step2_state.get("wide", {}).get(
                                                        snap_set_id
                                                    )
                                                synced_df = synced_map.get(snap_set_id)
                                                st.markdown(
                                                    f"**Set {snap_set_id} — {set_names.get(snap_set_id, snap_set_id)}**"
                                                )
                                                col_a, col_b = st.columns(2)
                                                with col_a:
                                                    st.write("Staged")
                                                    if isinstance(staged_df, pd.DataFrame):
                                                        st.dataframe(
                                                            staged_df,
                                                            use_container_width=True,
                                                        )
                                                    else:
                                                        st.caption("Нет данных")
                                                with col_b:
                                                    st.write("Synced baseline")
                                                    if isinstance(synced_df, pd.DataFrame):
                                                        st.dataframe(
                                                            synced_df,
                                                            use_container_width=True,
                                                        )
                                                    else:
                                                        st.caption("Нет данных")

                                                diff_columns: list[str] = []
                                                if isinstance(
                                                    staged_df, pd.DataFrame
                                                ) and isinstance(
                                                    synced_df, pd.DataFrame
                                                ):
                                                    all_cols = sorted(
                                                        set(staged_df.columns)
                                                        | set(synced_df.columns)
                                                    )
                                                    for col in all_cols:
                                                        if col == "attribute_set_id":
                                                            continue
                                                        left_series = (
                                                            staged_df[col]
                                                            if col in staged_df
                                                            else pd.Series(
                                                                [None]
                                                                * len(staged_df)
                                                            )
                                                        )
                                                        right_series = (
                                                            synced_df[col]
                                                            if col in synced_df
                                                            else pd.Series(
                                                                [None]
                                                                * len(synced_df)
                                                            )
                                                        )
                                                        if len(left_series) != len(
                                                            right_series
                                                        ):
                                                            diff_columns.append(col)
                                                            continue
                                                        values_differ = False
                                                        for left_val, right_val in zip(
                                                            left_series.astype(object),
                                                            right_series.astype(object),
                                                        ):
                                                            if not _values_equal(
                                                                left_val, right_val
                                                            ):
                                                                values_differ = True
                                                                break
                                                        if values_differ:
                                                            diff_columns.append(col)
                                                st.write(
                                                    "Changed columns:",
                                                    diff_columns
                                                    if diff_columns
                                                    else "Нет различий",
                                                )
                                                st.markdown("---")
                                        with tab_payload:
                                            if payload_events:
                                                payload_df = pd.DataFrame(
                                                    [
                                                        {
                                                            "sku": event.get("sku"),
                                                            "keys": ", ".join(
                                                                str(item)
                                                                for item in (
                                                                    event.get("keys", [])
                                                                    if isinstance(
                                                                        event.get("keys"),
                                                                        (list, tuple, set),
                                                                    )
                                                                    else [
                                                                        event.get(
                                                                            "keys"
                                                                        )
                                                                    ]
                                                                )
                                                                if item not in (None, "")
                                                            ),
                                                            "preview": str(
                                                                event.get(
                                                                    "payload_preview"
                                                                )
                                                            ),
                                                        }
                                                        for event in payload_events
                                                    ]
                                                )
                                                st.dataframe(
                                                    payload_df,
                                                    use_container_width=True,
                                                )
                                            else:
                                                st.caption("Payload еще не собирался.")

                                    st.markdown("---")
                                    c1, c2, c3 = st.columns([1, 1, 1])
                                    btn_generate = c1.button(
                                        "🌐 Generate Descriptions/Translation",
                                        key="btn_step2_generate_bottom",
                                    )
                                    btn_reset = c3.button(
                                        "🔄 Reset all",
                                        key="btn_step2_reset_bottom",
                                    )

                                    if btn_generate:
                                        products = _build_step3_products()
                                        _ensure_descriptions_initialized(products)
                                        step3_state = st.session_state.setdefault("step3", {})
                                        step3_state["products"] = products
                                        step3_state["errors"] = []
                                        st.session_state["step3_active"] = True
                                        st.session_state["step3_generation_pending"] = True
                                        st.rerun()

                                    with c2:
                                        if st.button(
                                            "💾 Save changes to Magento",
                                            key="btn_step2_save_specs_bottom",
                                        ):
                                            st.toast(
                                                "⏳ Saving changes to Magento…",
                                                icon="⏳",
                                            )

                                            df_filtered = st.session_state.get(
                                                "step2_products"
                                            )
                                            df_initial = step2_state.get("wide_orig")

                                            rows_to_save = _collect_step2_products_rows(
                                                df_filtered,
                                                df_original=df_initial,
                                                row_meta_map=step2_state.get(
                                                    "row_meta_map"
                                                ),
                                                step2_state=step2_state,
                                            )

                                            try:
                                                save_rows_to_magento(
                                                    rows_to_save,
                                                    session=session,
                                                    trace=trace,
                                                    is_dry_run=False,
                                                )
                                                st.success(
                                                    "✅ Saved successfully to Magento"
                                                )
                                            except Exception as exc:
                                                st.error(
                                                    f"❌ Failed to save to Magento: {exc}"
                                                )
                                                trace(
                                                    {
                                                        "where": "save:error",
                                                        "err": str(exc),
                                                    }
                                                )

                                    if btn_reset:
                                        _reset_step2_state()
                                        st.rerun()

                                    trace_json = json.dumps(
                                        trace_events,
                                        ensure_ascii=False,
                                        indent=2,
                                        default=str,
                                    )
                                    payload_json = json.dumps(
                                        payload_events,
                                        ensure_ascii=False,
                                        indent=2,
                                        default=str,
                                    )
                                    st.download_button(
                                        "⬇️ Download trace.json",
                                        data=trace_json,
                                        file_name="trace.json",
                                        mime="application/json",
                                    )
                                    st.download_button(
                                        "⬇️ Download payload.json",
                                        data=payload_json,
                                        file_name="payload.json",
                                        mime="application/json",
                                    )

        if st.session_state.get("step3_active"):
            st.markdown("---")
            st.header("Step 3. Generate and review descriptions")

            step3_state = st.session_state.setdefault("step3", {})
            _try_restore_step3_autosave()
            products = step3_state.get("products")
            if not products:
                products = _build_step3_products()
                step3_state["products"] = products
            _ensure_descriptions_initialized(products or [])

            if st.session_state.get("step3_generation_pending") and not products:
                st.session_state["step3_generation_pending"] = False

            if st.session_state.get("step3_generation_pending") and products:
                results, errors = _generate_descriptions_for_products(products)
                descriptions_map = st.session_state.get("descriptions")
                if not isinstance(descriptions_map, dict):
                    descriptions_map = {}
                for sku, payload in results.items():
                    descriptions_map[sku] = payload
                st.session_state["descriptions"] = descriptions_map
                step3_state["errors"] = errors
                st.session_state["step3_generation_pending"] = False

            errors = step3_state.get("errors") or []
            if errors:
                st.warning(
                    "Some descriptions could not be generated automatically. "
                    "The NL/DE fields for these items have been left blank."
                )
                st.write("\n".join(f"- {msg}" for msg in errors))

            descriptions_map = st.session_state.get("descriptions", {})
            table_rows: list[dict[str, object]] = []
            for product in products or []:
                entry = descriptions_map.get(product.sku, {}) if isinstance(descriptions_map, Mapping) else {}
                nl_value = _clean_description_value(entry.get("nl"))
                de_value = _clean_description_value(entry.get("de"))
                table_rows.append(
                    {
                        "Name": product.name,
                        "SKU": product.sku,
                        "EN Description": _clean_description_value(entry.get("en")),
                        "NL Description": _decorate_truncated_translation(nl_value),
                        "DE Description": _decorate_truncated_translation(de_value),
                        "ES Description": _clean_description_value(entry.get("es")),
                        "FR Description": _clean_description_value(entry.get("fr")),
                    }
                )

            st.session_state["step3_output_rows"] = table_rows
            if isinstance(descriptions_map, Mapping) and descriptions_map:
                _save_step3_autosave(descriptions_map, table_rows)

            if not table_rows:
                st.info("No products ready for description generation.")
            else:
                df_step3 = pd.DataFrame(table_rows)
                column_config = {
                    "Name": st.column_config.TextColumn("Name", disabled=True, width="medium"),
                    "SKU": st.column_config.TextColumn("SKU", disabled=True, width="small"),
                    "EN Description": st.column_config.TextColumn("EN Description", width="large"),
                    "NL Description": st.column_config.TextColumn("NL Description", width="large"),
                    "DE Description": st.column_config.TextColumn("DE Description", width="large"),
                    "ES Description": st.column_config.TextColumn("ES Description", width="large"),
                    "FR Description": st.column_config.TextColumn("FR Description", width="large"),
                }
                editor_df = st.data_editor(
                    df_step3,
                    column_config=column_config,
                    use_container_width=True,
                    num_rows="fixed",
                    key="step3_editor",
                )

                if isinstance(editor_df, pd.DataFrame):
                    new_map: dict[str, dict[str, str]] = {}
                    for row in editor_df.to_dict(orient="records"):
                        sku = _normalize_text(row.get("SKU"))
                        if not sku:
                            continue
                        new_map[sku] = {
                            "en": _clean_description_value(row.get("EN Description")),
                            "nl": _strip_truncation_warning(row.get("NL Description")),
                            "de": _strip_truncation_warning(row.get("DE Description")),
                            "es": _clean_description_value(row.get("ES Description")),
                            "fr": _clean_description_value(row.get("FR Description")),
                        }
                    if new_map:
                        descriptions_map = st.session_state.get("descriptions")
                        if not isinstance(descriptions_map, dict):
                            descriptions_map = {}
                        descriptions_map.update(new_map)
                        st.session_state["descriptions"] = descriptions_map
                        _save_step3_autosave(descriptions_map)

                btn_step3_save = st.button("💾 Save to Magento", key="btn_step3_save")
                if btn_step3_save:
                    save_step3_to_magento()
