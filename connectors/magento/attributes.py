"""Magento attribute metadata cache with alias-aware option lookup."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import requests

try:
    from connectors.magento.client import magento_get
except ModuleNotFoundError:
    from connectors.magento import client as _client

    magento_get = _client.magento_get
from services.country_aliases import country_aliases


class AttributeMetaCache:
    """Cache Magento attribute metadata and build option alias maps."""

    def __init__(self, session: requests.Session, base_url: str):
        self._session = session
        self._base_url = base_url
        self._data: Dict[str, Dict[str, Any]] = {}

    def load(self, codes: Iterable[str]) -> None:
        """Fetch metadata for the provided attribute codes if not cached."""
        unique: list[str] = []
        for code in codes:
            code_str = str(code).strip() if isinstance(code, str) else ""
            if not code_str or code_str in self._data or code_str in unique:
                continue
            unique.append(code_str)

        for code in unique:
            try:
                meta = magento_get(
                    self._session,
                    self._base_url,
                    f"/products/attributes/{code}",
                )
            except Exception:  # pragma: no cover - network failure safety
                meta = {}
            prepared = self._prepare_meta(code, meta or {})
            self._data[code] = prepared

    def get(self, code: str) -> Optional[Dict[str, Any]]:
        """Return cached metadata for ``code`` if available."""
        if code not in self._data:
            self.load([code])
        return self._data.get(code)

    def set_static(self, code: str, info: Dict[str, Any]) -> None:
        """Store synthetic metadata (e.g. for categories) in the cache."""
        self._data[code] = info

    @staticmethod
    def _prepare_meta(code: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        options = meta.get("options") or []
        alias_map: Dict[Any, Any] = {}
        value_to_label: Dict[Any, str] = {}
        valid_examples: list[str] = []

        for opt in options:
            label_raw = opt.get("label")
            label = str(label_raw).strip() if isinstance(label_raw, str) else ""
            value = opt.get("value")
            if value in (None, ""):
                continue

            str_value = str(value).strip()
            if not str_value:
                continue

            int_value: Optional[int]
            try:
                int_value = int(str_value)
            except (TypeError, ValueError):
                int_value = None

            target = int_value if int_value is not None else str_value

            if label:
                value_to_label[str_value] = label
                if int_value is not None:
                    value_to_label[int_value] = label
                AttributeMetaCache._add_alias(alias_map, label, target)
                AttributeMetaCache._add_alias(alias_map, label.lower(), target)
                if label not in valid_examples and len(valid_examples) < 5:
                    valid_examples.append(label)
            AttributeMetaCache._add_alias(alias_map, str_value, target)
            if int_value is not None:
                AttributeMetaCache._add_alias(alias_map, int_value, target)
                AttributeMetaCache._add_alias(alias_map, str(int_value), target)

            aliases = opt.get("aliases") or []
            for alias in aliases:
                AttributeMetaCache._add_alias(alias_map, alias, target)

            if code == "country_of_manufacture" and label:
                for alias in country_aliases(label):
                    AttributeMetaCache._add_alias(alias_map, alias, target)
                    AttributeMetaCache._add_alias(alias_map, alias.upper(), target)

        prepared = dict(meta)
        prepared.setdefault("attribute_code", code)
        prepared["options_map"] = alias_map
        prepared["values_to_labels"] = value_to_label
        prepared["valid_examples"] = valid_examples
        prepared["frontend_input"] = (meta.get("frontend_input") or "text").lower()
        backend_type = meta.get("backend_type") or ""
        if backend_type:
            prepared["backend_type"] = str(backend_type).lower()
        return prepared

    @staticmethod
    def _add_alias(alias_map: Dict[Any, Any], alias: Any, target: Any) -> None:
        if alias is None:
            return
        if isinstance(alias, str):
            cleaned = alias.strip()
            if not cleaned:
                return
            alias_map.setdefault(cleaned, target)
            alias_map.setdefault(cleaned.lower(), target)
        else:
            try:
                int_alias = int(alias)
            except (TypeError, ValueError):
                return
            alias_map.setdefault(int_alias, target)
            alias_map.setdefault(str(int_alias), target)
