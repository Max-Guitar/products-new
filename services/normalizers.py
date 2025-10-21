"""Normalization helpers for Magento attribute payloads."""
from __future__ import annotations

import math
import re
from typing import Any


INCH_PER_MM = 1 / 25.4

MEASURE_ATTRS = {
    "scale_mensur",
    "neck_radius",
    "neck_nutwidth",
    "nut_width",
}

TEXT_INPUT_TYPES = {
    "text",
    "textarea",
    "varchar",
    "string",
    "date",
    "datetime",
}


def _format_inch_value(value: float) -> str:
    rounded = round(value, 2)
    if math.isclose(rounded, int(round(rounded))):
        return f"{int(round(rounded))}\""
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return f"{text}\""


def normalize_units(attr: str, value: Any) -> Any:
    """Normalize unit-based attributes to inches with two decimals."""

    if attr not in MEASURE_ATTRS:
        return value
    if value in (None, ""):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_inch_value(float(value))

    text = str(value).strip()
    if not text:
        return value

    mm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", text, re.IGNORECASE)
    if mm_match:
        try:
            numeric = float(mm_match.group(1))
        except ValueError:
            return value
        inches = numeric * INCH_PER_MM
        return _format_inch_value(inches)

    inch_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:in|inch|\")", text, re.IGNORECASE)
    if inch_match:
        try:
            numeric = float(inch_match.group(1))
        except ValueError:
            return value
        return _format_inch_value(numeric)

    # Plain numeric string without unit
    try:
        numeric = float(text)
    except ValueError:
        return value
    return _format_inch_value(numeric)


def _coerce_ai_value(value: object, meta: dict | None) -> object:
    """Coerce an AI suggestion based on Magento attribute metadata."""

    meta = meta or {}
    input_type = (
        meta.get("frontend_input")
        or meta.get("backend_type")
        or "text"
    )
    input_type = str(input_type).lower()

    if isinstance(value, dict) and "value" in value:
        value = value.get("value")

    if input_type in TEXT_INPUT_TYPES:
        if value in (None, ""):
            return ""
        text_value = str(value).strip()
        return text_value if text_value else ""

    if input_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return None

    if input_type in {"int", "integer"}:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return value

    if input_type in {"decimal", "price", "float"}:
        try:
            return float(value)
        except Exception:
            return value

    if input_type == "multiselect":
        if isinstance(value, (list, tuple, set)):
            normalized = [
                str(item).strip()
                for item in value
                if item not in (None, "") and str(item).strip()
            ]
            return normalized
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    if value is None:
        return None

    return value

from connectors.magento.attributes import AttributeMetaCache


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):  # pragma: no cover - defensive NaN handling
        return str(value) == "nan"
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _sanitize_target(target: Any) -> Any:
    if target is None:
        return None
    if isinstance(target, str):
        cleaned = target.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except (TypeError, ValueError):
            return cleaned
    if isinstance(target, bool):  # pragma: no cover - defensive
        return int(target)
    if isinstance(target, int):
        return int(target)
    if isinstance(target, float):
        if target.is_integer():
            return int(target)
        return target
    return target


def _match_option_value(raw_value: Any, info: dict[str, Any]) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, set)):
        return None

    if isinstance(raw_value, str):
        candidate = raw_value.strip()
    else:
        candidate = str(raw_value).strip()

    if not candidate:
        return None

    options_map = info.get("options_map") or {}
    values_to_labels = info.get("values_to_labels") or {}

    if candidate in options_map:
        return _sanitize_target(options_map[candidate])

    if candidate in values_to_labels:
        return _sanitize_target(candidate)

    candidate_fold = candidate.casefold()

    for key, mapped in (options_map or {}).items():
        if isinstance(key, str) and key.casefold() == candidate_fold:
            return _sanitize_target(mapped)

    for stored_value, label in (values_to_labels or {}).items():
        if isinstance(label, str) and label.casefold() == candidate_fold:
            return _sanitize_target(stored_value)
        if isinstance(stored_value, str) and stored_value.casefold() == candidate_fold:
            return _sanitize_target(stored_value)

    return None


def _to_iterable(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_for_magento(code: str, val: Any, meta: AttributeMetaCache | None):
    """Normalize ``val`` for the Magento attribute identified by ``code``."""
    if code == "no_strings" and _is_blank(val):
        return None

    was_blank = _is_blank(val)

    val = normalize_units(code, val)

    info: dict[str, Any] | None = None
    if isinstance(meta, AttributeMetaCache):
        cached = meta.get(code)
        if isinstance(cached, dict):
            info = cached
    elif isinstance(meta, dict):  # pragma: no cover - fallback safety
        maybe = meta.get(code)
        if isinstance(maybe, dict):
            info = maybe
    elif hasattr(meta, "get") and callable(getattr(meta, "get")):
        maybe = meta.get(code)
        if isinstance(maybe, dict):
            info = maybe

    if info is None:
        info = {}

    ftype = info.get("frontend_input") or info.get("backend_type") or "text"
    ftype = str(ftype).lower()

    def to_id(value: Any):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
        elif isinstance(value, float) and str(value) == "nan":  # pragma: no cover - safety
            return None
        elif isinstance(value, (list, tuple, set)):
            return None
        matched = _match_option_value(value, info)
        if matched is not None:
            return matched
        s = str(value).strip()
        if not s:
            return None
        try:
            return int(s)
        except (TypeError, ValueError):
            return _sanitize_target(s)

    if ftype in TEXT_INPUT_TYPES:
        if val in (None, ""):
            return ""
        text_value = str(val).strip()
        return text_value if text_value else ""

    if was_blank:
        return None

    if ftype in {"select", "boolean", "int"}:
        if ftype == "boolean":
            truthy = {"1", "true", "yes", "y", "on"}
            falsy = {"0", "false", "no", "n", "off"}
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in truthy:
                    return 1
                if lowered in falsy:
                    return 0
            if isinstance(val, bool):
                return int(val)
            matched_bool = _match_option_value(val, info)
            if matched_bool is not None:
                return matched_bool
        return to_id(val)

    if ftype == "multiselect":
        items = _to_iterable(val)
        ids = []
        for item in items:
            matched = _match_option_value(item, info)
            if matched is None:
                matched = to_id(item)
            if matched is not None:
                ids.append(matched)
        cleaned = [item for item in ids if item is not None]
        return cleaned or None

    if isinstance(val, str):
        return val.strip()
    return val
