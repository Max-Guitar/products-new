"""Client helpers for Magento REST API requests."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

API_PREFIX = "/rest/V1"


def magento_get(
    session: requests.Session,
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Perform a GET request to a Magento REST endpoint."""

    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}{API_PREFIX}{path}"
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()
