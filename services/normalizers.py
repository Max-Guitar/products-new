"""Normalization helpers for Magento attribute payloads."""
from __future__ import annotations

from typing import Any

from connectors.magento.attributes import AttributeMetaCache


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

    info = meta.get(code) if isinstance(meta, AttributeMetaCache) else None
    if info is None:
        info = {}

    ftype = info.get("frontend_input") or info.get("backend_type") or "text"
    ftype = str(ftype).lower()
    options_map = info.get("options_map") or {}

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
        s = str(value).strip()
        if not s:
            return None
        if s in options_map:
            target = options_map[s]
            try:
                return int(target)
            except (TypeError, ValueError):
                return None
        lower = s.lower()
        if lower in options_map:
            target = options_map[lower]
            try:
                return int(target)
            except (TypeError, ValueError):
                return None
        try:
            return int(s)
        except (TypeError, ValueError):
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
        return to_id(val)

    if ftype == "multiselect":
        items = _to_iterable(val)
        ids = [to_id(item) for item in items]
        cleaned = [item for item in ids if item is not None]
        return cleaned or None

    if isinstance(val, str):
        return val.strip()
    return val
