"""Magento attribute metadata cache with alias-aware option lookup."""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests

try:
    from connectors.magento.client import magento_get
except ModuleNotFoundError:  # pragma: no cover - fallback for notebooks
    from connectors.magento import client as _client

    magento_get = _client.magento_get
from services.country_aliases import country_aliases


CANDIDATE_BRAND_CODES = ["brand", "manufacturer", "brand_name"]


class AttributeMetaCache:
    """Cache Magento attribute metadata and build option alias maps."""

    def __init__(self, session: requests.Session, base_url: str):
        self._session = session
        self._base_url = base_url
        self._store: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}

    def load(self, codes: list[str]) -> None:
        """Fetch metadata for the provided attribute codes if not cached."""

        unique: list[str] = []
        for raw in codes or []:
            code = str(raw).strip() if isinstance(raw, str) else ""
            if not code or code in unique or code in self._store:
                continue
            unique.append(code)

        if "brand" in unique:
            best_code: Optional[str] = None
            best_size = -1
            for candidate in CANDIDATE_BRAND_CODES:
                options = self._fetch_options(candidate)
                if len(options) > best_size:
                    best_code = candidate
                    best_size = len(options)
            if best_code:
                self._aliases["brand"] = best_code

        for code in unique:
            lookup_code = self._aliases.get(code, code)
            try:
                meta = magento_get(
                    self._session,
                    self._base_url,
                    f"/products/attributes/{lookup_code}",
                )
            except RuntimeError:
                raise
            except Exception:  # pragma: no cover - network failure safety
                meta = {}
            prepared = self._prepare_meta(code, meta or {})
            self._store[code] = prepared

    def get(self, code: str) -> Optional[Dict[str, Any]]:
        """Return cached metadata for ``code`` if available."""

        if code not in self._store:
            self.load([code])
        return self._store.get(code)

    def set_static(self, code: str, info: Dict[str, Any]) -> None:
        """Store synthetic metadata (e.g. for categories) in the cache."""

        self._store[code] = info

    def _fetch_options(self, code: str, store_id: int = 0) -> list[dict]:
        for path in (
            f"/products/attributes/{code}/options?storeId={store_id}",
            f"/products/attributes/{code}/options",
            f"/products/attributes/{code}/options?storeId=1",
        ):
            try:
                data = magento_get(self._session, self._base_url, path) or []
                if isinstance(data, list):
                    return data
            except RuntimeError:
                raise
            except Exception:
                continue
        return []

    def _prepare_meta(self, code: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        code_eff = self._aliases.get(code, code)
        options_raw = meta.get("options")
        if not options_raw:
            options_raw = self._fetch_options(code_eff)

        prepared: Dict[str, Any] = {
            "attribute_code": code,
            "frontend_input": (meta.get("frontend_input") or "text").lower(),
            "options": [],
            "options_map": {},
            "values_to_labels": {},
            "valid_examples": [],
        }

        cleaned_options: list[Dict[str, Any]] = []
        values_to_labels: Dict[Any, str] = {}
        options_map: Dict[Any, Any] = {}
        valid_examples: list[str] = []

        for opt in options_raw or []:
            if not isinstance(opt, dict):
                continue
            value = opt.get("value")
            if value in (None, ""):
                continue

            raw_label = opt.get("label", "")
            label_clean = str(raw_label).strip()
            if not label_clean:
                label_clean = str(value).strip()
            if not label_clean:
                continue

            cleaned_options.append({"label": label_clean, "value": value})

            str_value = str(value).strip()
            if not str_value:
                continue

            try:
                int_value: Optional[int] = int(str_value)
            except (TypeError, ValueError):
                int_value = None

            target = int_value if int_value is not None else str_value

            values_to_labels[str_value] = label_clean
            if int_value is not None:
                values_to_labels[int_value] = label_clean

            if label_clean not in valid_examples and len(valid_examples) < 5:
                valid_examples.append(label_clean)

            self._add_alias(options_map, label_clean, target)
            self._add_alias(options_map, label_clean.lower(), target)
            self._add_alias(options_map, str_value, target)
            if int_value is not None:
                self._add_alias(options_map, int_value, target)
                self._add_alias(options_map, str(int_value), target)

            for alias in opt.get("aliases") or []:
                self._add_alias(options_map, alias, target)

            if code == "country_of_manufacture":
                for alias in country_aliases(label_clean):
                    self._add_alias(options_map, alias, target)
                    self._add_alias(options_map, alias.upper(), target)

        prepared["options"] = cleaned_options
        prepared["values_to_labels"] = values_to_labels
        prepared["options_map"] = options_map
        prepared["valid_examples"] = valid_examples

        backend_type = meta.get("backend_type")
        if backend_type:
            prepared["backend_type"] = str(backend_type).lower()
        if meta.get("is_required") is not None:
            prepared["is_required"] = bool(meta.get("is_required"))

        if cleaned_options and prepared["frontend_input"] in {"", "text", "varchar", "static"}:
            prepared["frontend_input"] = "select"

        yesno = {opt["label"].lower() for opt in cleaned_options}
        if cleaned_options and yesno and (
            yesno <= {"yes", "no"} or yesno <= {"0", "1", "true", "false"}
        ):
            prepared["frontend_input"] = "boolean"

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
