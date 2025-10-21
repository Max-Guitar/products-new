"""Client helpers for Magento REST API requests."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import requests

from utils.http import build_magento_url, magento_get as _magento_get
from utils.http import magento_get_logged as _magento_get_logged

MAGENTO_ADMIN_TOKEN: str = os.environ.get("MAGENTO_ADMIN_TOKEN", "")


_LOGGER = logging.getLogger(__name__)


_ALL_ATTRIBUTES_CACHE: Dict[tuple[str, int], Dict[str, Dict[str, Any]]] = {}
_ATTRIBUTE_SET_CACHE: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
_ATTRIBUTE_META_CACHE: Dict[tuple[str, Optional[int], str], Dict[str, Any]] = {}


def _with_query(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    if not params:
        return path
    query = urlencode(params, doseq=True)
    if not query:
        return path
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}{query}"


def get_api_base(base_url: str) -> str:
    """Return the canonical Magento REST base for ``base_url``."""

    return build_magento_url(base_url, "").rstrip("/")


def _token_value() -> Optional[str]:
    token = MAGENTO_ADMIN_TOKEN.strip()
    return token or None


def magento_get(
    session: requests.Session,
    base_url: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Any:
    """Perform a GET request to a Magento REST endpoint with auth headers."""

    return _magento_get(
        session,
        base_url,
        path,
        headers=headers,
        timeout=timeout,
        token=_token_value(),
    )


def magento_get_logged(
    session: requests.Session,
    base_url: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Tuple[int, Optional[str], str, requests.Response]:
    """Perform a GET request returning response metadata without raising."""

    return _magento_get_logged(
        session,
        base_url,
        path,
        headers=headers,
        timeout=timeout,
        token=_token_value(),
    )


def _cache_base_key(base_url: str) -> str:
    return (base_url or "").rstrip("/")


def _normalize_code(code: object) -> Optional[str]:
    if not isinstance(code, str):
        if code in (None, ""):
            return None
        code = str(code)
    cleaned = code.strip()
    return cleaned or None


def _safe_options(
    session: requests.Session,
    base_url: str,
    code: str,
    *,
    store_id: int = 0,
) -> List[Dict[str, Any]]:
    for path in (
        f"/products/attributes/{quote(code, safe='')}/options?storeId={store_id}",
        f"/products/attributes/{quote(code, safe='')}/options",
    ):
        try:
            data = magento_get(session, base_url, path) or []
        except RuntimeError:
            raise
        except Exception:
            continue
        if isinstance(data, list):
            return data
    return []


def get_all_attributes(
    session: requests.Session,
    base_url: str,
    *,
    store_id: int = 0,
    use_cache: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of attribute code to global metadata with options."""

    cache_key = (_cache_base_key(base_url), store_id)
    if use_cache and cache_key in _ALL_ATTRIBUTES_CACHE:
        return _ALL_ATTRIBUTES_CACHE[cache_key]

    collected: Dict[str, Dict[str, Any]] = {}
    page = 1
    page_size = 200

    while True:
        params: Dict[str, Any] = {
            "searchCriteria[currentPage]": page,
            "searchCriteria[pageSize]": page_size,
            "fields": "items[attribute_code,frontend_input,backend_type,"
            "frontend_label,is_required,options],total_count",
        }
        path = _with_query("/products/attributes", params)
        try:
            payload = magento_get(session, base_url, path)
        except RuntimeError:
            raise
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("attributes") or []
        else:
            items = payload or []

        if not isinstance(items, list):
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue
            code = _normalize_code(item.get("attribute_code"))
            if not code:
                continue
            meta = dict(item)
            options = meta.get("options")
            if not isinstance(options, list):
                options = _safe_options(session, base_url, code, store_id=store_id)
            meta["attribute_code"] = code
            meta.setdefault("code", code)
            meta["options"] = options or []
            collected[code] = meta

        if len(items) < page_size:
            break
        page += 1

    if use_cache:
        _ALL_ATTRIBUTES_CACHE[cache_key] = collected

    return collected


def get_attribute_set_attributes(
    session: requests.Session,
    base_url: str,
    set_id: int,
    *,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Return attribute metadata for the specified set."""

    try:
        set_key = int(set_id)
    except (TypeError, ValueError):
        return []

    cache_key = (_cache_base_key(base_url), set_key)
    if use_cache and cache_key in _ATTRIBUTE_SET_CACHE:
        return _ATTRIBUTE_SET_CACHE[cache_key]

    path = f"/products/attribute-sets/{set_key}/attributes"
    try:
        payload = magento_get(session, base_url, path)
    except RuntimeError:
        raise
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("attributes") or []
    else:
        items = payload or []

    if not isinstance(items, list):
        items = []

    if use_cache:
        _ATTRIBUTE_SET_CACHE[cache_key] = items

    return items


def get_attribute_meta(
    session: requests.Session,
    base_url: str,
    code: str,
    set_id: Optional[int] = None,
    *,
    store_id: int = 0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Return attribute metadata with per-set fallback to global."""

    normalized_code = _normalize_code(code)
    if not normalized_code:
        return {}

    normalized_set_id: Optional[int]
    if set_id is None:
        normalized_set_id = None
    else:
        try:
            normalized_set_id = int(set_id)
        except (TypeError, ValueError):
            normalized_set_id = None

    cache_key = (_cache_base_key(base_url), normalized_set_id, normalized_code)
    if use_cache and cache_key in _ATTRIBUTE_META_CACHE:
        return _ATTRIBUTE_META_CACHE[cache_key]

    set_meta: Dict[str, Any] | None = None
    if normalized_set_id is not None:
        for attr in get_attribute_set_attributes(
            session,
            base_url,
            normalized_set_id,
            use_cache=use_cache,
        ):
            if not isinstance(attr, dict):
                continue
            code_candidate = _normalize_code(attr.get("attribute_code"))
            if code_candidate == normalized_code:
                set_meta = dict(attr)
                break

    global_meta = get_all_attributes(
        session,
        base_url,
        store_id=store_id,
        use_cache=use_cache,
    ).get(normalized_code)

    if set_meta:
        result: Dict[str, Any] = dict(global_meta or {})
        result.update(set_meta)
    else:
        result = dict(global_meta or {})

    if not result:
        result = {"code": normalized_code, "attribute_code": normalized_code}
    else:
        result.setdefault("code", normalized_code)
        result.setdefault("attribute_code", normalized_code)

    global_options = []
    if isinstance(global_meta, dict):
        global_options = global_meta.get("options") or []

    options = result.get("options")
    if not isinstance(options, list) or not options:
        result["options"] = list(global_options) if isinstance(global_options, list) else []

    if use_cache:
        _ATTRIBUTE_META_CACHE[cache_key] = result

    return result


def get_default_products(
    session: requests.Session,
    base_url: str,
    *,
    attr_set_id: Optional[int] = None,
    qty_min: int = 0,
    limit: Optional[int] = None,
    minimal_fields: bool = False,
    enabled_only: bool | None = None,
    **kwargs: Any,
) -> List[dict]:
    """Return products for the default attribute set with optional limit.

    Parameters
    ----------
    session, base_url:
        Magento REST client configuration.
    qty_min:
        Minimum quantity filter applied after fetching stock information. When
        ``qty_min`` is ``0`` the filter is skipped and no stock calls are made.
    limit:
        Optional number of items to request server-side via ``pageSize``.
    minimal_fields:
        When ``True`` only a subset of fields is requested to speed up the
        response.
    attr_set_id:
        Optional attribute set identifier to filter server-side.
    enabled_only:
        When ``True`` adds a status filter restricting results to enabled
        products. When ``False`` or ``None`` (default) no status filter is
        applied.
    kwargs:
        Extra query parameters merged into the Magento search criteria.
    """

    if limit is not None and limit <= 0:
        return []

    base_params: Dict[str, Any] = {}

    filter_index = 0
    if enabled_only:
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][field]"
        ] = "status"
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][value]"
        ] = "1"
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][condition_type]"
        ] = "eq"
        filter_index += 1

    if attr_set_id is not None:
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][field]"
        ] = "attribute_set_id"
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][value]"
        ] = attr_set_id
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][condition_type]"
        ] = "eq"
        filter_index += 1

    if kwargs:
        base_params.update(kwargs)

    if minimal_fields:
        base_params[
            "fields"
        ] = "items[sku,name,attribute_set_id,created_at,type_id,custom_attributes],total_count"

    collected: List[dict] = []
    remaining = limit
    page = 1
    default_page_size = 1000 if limit is None else min(limit, 1000)

    while True:
        page_size = default_page_size
        if remaining is not None:
            page_size = max(1, min(page_size, remaining))

        params = dict(base_params)
        params["searchCriteria[currentPage]"] = page
        params["searchCriteria[pageSize]"] = page_size

        path = _with_query("/products", params)
        final_url = build_magento_url(base_url, path)
        _LOGGER.debug("get_default_products request URL: %s", final_url)

        data = _magento_get(
            session,
            base_url,
            path,
            timeout=30,
            token=_token_value(),
        )

        if isinstance(data, dict):
            items = data.get("items") or []
        else:
            items = data or []

        if not isinstance(items, list):
            items = []

        collected.extend(items)

        if remaining is not None:
            remaining -= len(items)
            if remaining <= 0:
                break

        if len(items) < page_size:
            break

        page += 1

    if remaining is not None and len(collected) > limit:
        collected = collected[:limit]

    if qty_min and qty_min > 0:
        filtered: List[dict] = []
        for item in collected:
            sku = item.get("sku")
            if not sku:
                continue
            try:
                stock = _magento_get(
                    session,
                    base_url,
                    f"/stockItems/{quote(str(sku), safe='')}",
                    timeout=30,
                    token=_token_value(),
                )
            except Exception:
                stock = {}

            ext = item.setdefault("extension_attributes", {})
            if isinstance(ext, dict):
                ext.setdefault("stock_item", stock)

            qty_value = 0
            if isinstance(stock, dict):
                qty_value = stock.get("qty") or 0
            try:
                qty_float = float(qty_value)
            except (TypeError, ValueError):
                qty_float = 0.0
            if qty_float >= float(qty_min):
                filtered.append(item)
        collected = filtered

    return collected
