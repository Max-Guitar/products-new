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
        self._debug_last: Dict[str, Dict[str, Any]] = {}

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
                raw_meta = (
                    magento_get(
                        self._session,
                        self._base_url,
                        f"/products/attributes/{lookup_code}",
                    )
                    or {}
                )
            except RuntimeError:
                raise
            except Exception:  # pragma: no cover - network failure safety
                raw_meta = {}

            try:
                raw_opts = magento_get(
                    self._session,
                    self._base_url,
                    f"/products/attributes/{lookup_code}/options?storeId=0",
                )
            except RuntimeError:
                raise
            except Exception:
                raw_opts = []

            self._debug_last[code] = {
                "raw_meta_frontend_input": raw_meta.get("frontend_input"),
                "raw_options_count": len(raw_opts) if isinstance(raw_opts, list) else -1,
                "raw_options_sample": (raw_opts or [])[:5],
            }

            meta: Dict[str, Any] = dict(raw_meta or {})
            meta.update(
                {
                    "attribute_code": code,
                    "frontend_input": raw_meta.get("frontend_input"),
                    "options": raw_opts,
                }
            )

            prepared = self._prepare_meta(code, meta)
            self._debug_last[code].update(
                {
                    "prepared_frontend_input": prepared.get("frontend_input"),
                    "prepared_options_count": len(prepared.get("options") or []),
                    "prepared_sample": (prepared.get("options") or [])[:3],
                }
            )
            self._store[code] = prepared

    def get(self, code: str) -> Optional[Dict[str, Any]]:
        """Return cached metadata for ``code`` if available."""

        if code not in self._store:
            self.load([code])
        return self._store.get(code)

    def set_static(self, code: str, info: Dict[str, Any]) -> None:
        """Store synthetic metadata (e.g. for categories) in the cache."""

        self._store[code] = info

    def build_and_set_static_for(self, codes: list[str], store_id: int = 0):
        """Build synthetic metadata for ``codes`` and cache it as static."""

        unique: list[str] = []
        for raw in codes or []:
            if not isinstance(raw, str):
                continue
            code = raw.strip()
            if not code or code in unique:
                continue
            unique.append(code)

        for code in unique:
            raw_meta: Dict[str, Any] = {}
            try:
                raw_meta = (
                    magento_get(
                        self._session,
                        self._base_url,
                        f"/products/attributes/{code}",
                    )
                    or {}
                )
            except RuntimeError:
                raise
            except Exception:
                raw_meta = {}

            options_raw: list[dict] = []
            options_source = "fallback"
            for path in (
                f"/products/attributes/{code}/options?storeId={store_id}",
                f"/products/attributes/{code}/options",
                f"/products/attributes/{code}/options?storeId=1",
            ):
                try:
                    data = magento_get(self._session, self._base_url, path)
                except RuntimeError:
                    raise
                except Exception:
                    continue
                if isinstance(data, list):
                    options_raw = data
                    options_source = "magento"
                    break

            cleaned: list[Dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for opt in options_raw or []:
                if not isinstance(opt, dict):
                    continue
                value = opt.get("value")
                if value in (None, ""):
                    continue
                label = str(opt.get("label", "")).strip()
                if not label:
                    label = str(value).strip()
                if not label:
                    continue
                key = (str(value), label)
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append({"label": label, "value": value})

            frontend_input = str(raw_meta.get("frontend_input") or "text").lower()
            if (raw_meta.get("frontend_input") or "").lower() == "boolean":
                frontend_input = "boolean"
            elif cleaned and frontend_input != "multiselect":
                frontend_input = "select"

            options_map: Dict[Any, Any] = {}
            values_to_labels: Dict[Any, str] = {}
            valid_examples: list[str] = []

            for option in cleaned:
                label = option.get("label")
                value = option.get("value")
                if not isinstance(label, str):
                    continue
                label_clean = label.strip()
                if not label_clean:
                    continue
                if label_clean not in valid_examples and len(valid_examples) < 10:
                    valid_examples.append(label_clean)

                str_value = str(value)
                values_to_labels[str_value] = label_clean
                try:
                    int_value = int(str_value)
                except (TypeError, ValueError):
                    int_value = None
                if int_value is not None:
                    values_to_labels[int_value] = label_clean

                self._add_alias(options_map, label_clean, value)
                self._add_alias(options_map, label_clean.lower(), value)
                self._add_alias(options_map, str_value, value)
                if int_value is not None:
                    self._add_alias(options_map, int_value, value)
                    self._add_alias(options_map, str(int_value), value)

            prepared = {
                "attribute_code": code,
                "frontend_input": frontend_input,
                "options": cleaned,
                "options_map": options_map,
                "values_to_labels": values_to_labels,
                "valid_examples": valid_examples,
                "options_source": options_source,
            }

            if frontend_input == "boolean" and not cleaned:
                prepared.setdefault("options", [])

            self.set_static(code, prepared)

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
