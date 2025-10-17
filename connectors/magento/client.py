"""Client helpers for Magento REST API requests."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

API_SUFFIX = "/rest/V1"
MAGENTO_ADMIN_TOKEN: str = os.environ.get("MAGENTO_ADMIN_TOKEN", "")


def get_api_base(base_url: str) -> str:
    """Return the canonical Magento REST base for ``base_url``."""

    return base_url.rstrip("/") + API_SUFFIX


def magento_get(
    session: requests.Session,
    base_url: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Perform a GET request to a Magento REST endpoint with auth headers."""

    if not path.startswith("/"):
        path = "/" + path
    url = get_api_base(base_url) + path

    auth_headers: Dict[str, str] = {"Accept": "application/json"}
    token = MAGENTO_ADMIN_TOKEN.strip()
    if token:
        auth_headers["Authorization"] = f"Bearer {token}"
    elif session.headers.get("Authorization"):
        auth_headers["Authorization"] = session.headers["Authorization"]

    if headers:
        auth_headers.update(headers)

    response = session.get(url, headers=auth_headers, timeout=30)
    response.raise_for_status()
    return response.json()
