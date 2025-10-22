"""Helpers for building Magento product update payloads."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping

from services.country_aliases import normalize_country


_BLANK_VALUES = {None, ""}


def _is_blank(value: Any) -> bool:
    if value in _BLANK_VALUES:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def map_value_for_magento(
    attr_code: str, value: Any, meta: Mapping[str, Any] | None
) -> Any:
    """Map UI value to Magento-compatible representation."""
    meta = meta or {}
    input_type = str(meta.get("frontend_input") or "").lower()
    options_map = meta.get("options_map") or {}
    if isinstance(options_map, Mapping):
        normalized_map: Dict[str, str] = {}
        for key, target in options_map.items():
            if key is None:
                continue
            if isinstance(key, str):
                key_clean = key.strip()
                if not key_clean:
                    continue
                normalized_map.setdefault(key_clean.lower(), str(target))
            else:
                key_str = str(key).strip()
                if not key_str:
                    continue
                normalized_map.setdefault(key_str.lower(), str(target))
        options_map = normalized_map
    else:
        options_map = {}

    if attr_code == "categories":
        return value

    if input_type in {"select", "boolean"}:
        if _is_blank(value):
            return None
        val_str = str(value).strip()
        if not val_str:
            return None
        if val_str.isdigit():
            return val_str
        mapped = options_map.get(val_str.lower())
        return str(mapped) if mapped is not None else None

    if input_type == "multiselect":
        if value in (None, "", []):
            return ""
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            parts = [item for item in value]
        else:
            parts = [v.strip() for v in str(value).split(",") if v.strip()]
        ids: list[str] = []
        for part in parts:
            pstr = str(part).strip()
            if not pstr:
                continue
            if pstr.isdigit():
                ids.append(pstr)
                continue
            mapped = options_map.get(pstr.lower())
            if mapped is not None:
                ids.append(str(mapped))
        return ",".join(ids)

    if attr_code == "scale_mensur" and value not in (None, ""):
        return str(value).replace('"', "").strip()

    return value


def _clean_attribute_set_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_magento_payload(
    row: Mapping[str, Any], meta_by_attr: Mapping[str, Mapping[str, Any]] | None
) -> Dict[str, Any]:
    """Build Magento product payload with mapped custom attributes."""
    meta_by_attr = meta_by_attr or {}
    data: MutableMapping[str, Any] = dict(row or {})

    sku = str(data.get("sku", "")).strip()
    if not sku:
        raise ValueError("SKU is required to build Magento payload")

    attr_set_value = data.get("attribute_set_id")
    attr_set_id = _clean_attribute_set_id(attr_set_value)

    if "country_of_manufacture" in data and not _is_blank(
        data.get("country_of_manufacture")
    ):
        data["country_of_manufacture"] = normalize_country(
            data.get("country_of_manufacture")
        )

    categories_value = data.get("categories")

    custom_attributes: list[dict[str, Any]] = []
    for code, value in data.items():
        if code in {"sku", "name", "attribute_set_id", "status", "type_id"}:
            continue
        meta = meta_by_attr.get(code) if isinstance(meta_by_attr, Mapping) else {}
        if code == "categories":
            continue
        mapped = map_value_for_magento(code, value, meta)
        if mapped in (None, ""):
            continue
        custom_attributes.append({"attribute_code": code, "value": mapped})

    payload: Dict[str, Any] = {"sku": sku, "custom_attributes": custom_attributes}
    if attr_set_id is not None:
        payload["attribute_set_id"] = attr_set_id

    extension_attributes: Dict[str, Any] = {}
    if isinstance(categories_value, Iterable) and not isinstance(
        categories_value, (str, bytes)
    ):
        links = []
        for idx, cat in enumerate(categories_value):
            try:
                cat_id = int(cat)
            except (TypeError, ValueError):
                continue
            links.append({"position": idx, "category_id": cat_id})
        if links:
            extension_attributes["category_links"] = links

    if extension_attributes:
        payload["extension_attributes"] = extension_attributes

    return {"product": payload}
