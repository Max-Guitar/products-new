"""Client helpers for Magento REST API requests."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode

import requests

from utils.http import build_magento_url, magento_get as _magento_get
from utils.http import magento_get_logged as _magento_get_logged

MAGENTO_ADMIN_TOKEN: str = os.environ.get("MAGENTO_ADMIN_TOKEN", "")


_LOGGER = logging.getLogger(__name__)


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


def get_default_products(
    session: requests.Session,
    base_url: str,
    *,
    attr_set_id: Optional[int] = None,
    qty_min: int = 0,
    limit: Optional[int] = None,
    minimal_fields: bool = False,
    enabled_only: bool | None = None,
    type_ids: Iterable[str] | None = None,
    created_at_from: str | None = None,
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
    type_ids:
        Optional iterable of product ``type_id`` values to filter server-side.
        When provided the values are joined with ``OR`` within the same filter
        group.
    created_at_from:
        Optional ISO timestamp string passed as a ``gteq`` filter for the
        ``created_at`` field.
    kwargs:
        Extra query parameters merged into the Magento search criteria.
    """

    if limit is not None and limit <= 0:
        return []

    base_params: Dict[str, Any] = {}

    filter_index = 0
    if enabled_only is True:
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

    if type_ids:
        values = [str(v) for v in type_ids if v not in (None, "")]
        if values:
            for idx, type_value in enumerate(values):
                base_params[
                    f"searchCriteria[filter_groups][{filter_index}][filters][{idx}][field]"
                ] = "type_id"
                base_params[
                    f"searchCriteria[filter_groups][{filter_index}][filters][{idx}][value]"
                ] = type_value
                base_params[
                    f"searchCriteria[filter_groups][{filter_index}][filters][{idx}][condition_type]"
                ] = "eq"
            filter_index += 1

    if created_at_from:
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][field]"
        ] = "created_at"
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][value]"
        ] = created_at_from
        base_params[
            f"searchCriteria[filter_groups][{filter_index}][filters][0][condition_type]"
        ] = "gteq"
        filter_index += 1

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
