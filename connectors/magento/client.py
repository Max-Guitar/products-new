"""Client helpers for Magento REST API requests."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

from utils.http import build_magento_url, magento_get as _magento_get
from utils.http import magento_get_logged as _magento_get_logged

MAGENTO_ADMIN_TOKEN: str = os.environ.get("MAGENTO_ADMIN_TOKEN", "")


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
