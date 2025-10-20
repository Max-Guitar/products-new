"""Normalization helpers for Magento attribute payloads."""
from __future__ import annotations

from typing import Any

from connectors.magento.attributes import AttributeMetaCache


TEXT_PASS_THROUGH = {
    "finish",
    "controls",
    "tuning_machines",
    "bridge",
    "top_material",
    "neck_material",
    "fretboard_material",
    "short_description",
}


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
    if val is None:
        return None
    if isinstance(val, str) and not val.strip():
        return None

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

    if code in TEXT_PASS_THROUGH:
        if isinstance(val, str):
            cleaned = val.strip()
            return cleaned
        return val

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
