"""HTTP utilities for configuring reusable sessions."""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)

# Magento REST endpoints may live under different prefixes depending on
# deployment configuration and whether ``index.php`` is exposed.  The order is
# important – we try the "all" scope first to keep store context explicit.
MAGENTO_REST_PATH_CANDIDATES: tuple[str, ...] = (
    "/rest/all/V1",
    "/index.php/rest/all/V1",
    "/rest/V1",
    "/index.php/rest/V1",
)


_LOGGER = logging.getLogger(__name__)
_MAGENTO_REST_PREFIX = "rest/all/V1"


def build_magento_headers(
    *, token: Optional[str] = None, session: Optional[requests.Session] = None
) -> dict[str, str]:
    """Return default Magento REST headers with optional bearer token."""

    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    auth_value: Optional[str] = None
    if token:
        auth_value = f"Bearer {token}"
    elif session is not None:
        auth_value = session.headers.get("Authorization")
    if auth_value:
        headers["Authorization"] = auth_value
    return headers


def get_session(auth_token: Optional[str] = None) -> requests.Session:
    """Return a configured :class:`requests.Session` with retries.

    Parameters
    ----------
    auth_token:
        Magento admin bearer token. When provided, the ``Authorization`` header
        will be configured automatically.
    """

    session = requests.Session()
    session.headers.update(build_magento_headers(token=auth_token))

    retry = Retry(
        total=3,
        status_forcelist=DEFAULT_STATUS_FORCELIST,
        backoff_factor=0.4,
        allowed_methods=("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=80, pool_maxsize=80, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def normalize_magento_path(path: str) -> str:
    """Return ``path`` without redundant ``rest/*/V1`` prefixes."""

    if not isinstance(path, str):
        path = str(path or "")
    cleaned = re.sub(r"^/?(rest/(all|default)/V1/)+", "", path)
    return cleaned.lstrip("/")


def build_magento_url(base_url: str, path: str) -> str:
    """Construct the canonical REST URL for ``path`` under ``base_url``."""

    normalized = normalize_magento_path(path)
    base = base_url.rstrip("/")
    if not normalized:
        return f"{base}/{_MAGENTO_REST_PREFIX}"
    return f"{base}/{_MAGENTO_REST_PREFIX}/{normalized}"


def magento_get(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    token: Optional[str] = None,
) -> dict | list:
    """Perform an authenticated GET request to the Magento REST API."""

    final_url = build_magento_url(base_url, path)
    _LOGGER.debug("magento_get URL=%s", final_url)
    auth_headers: Dict[str, str] = build_magento_headers(
        token=(token.strip() if isinstance(token, str) else None), session=session
    )
    if headers:
        auth_headers.update(headers)

    response = session.get(final_url, headers=auth_headers, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        snippet = response.text[:200]
        raise RuntimeError(f"Non-JSON from Magento for {final_url}: {snippet}")

    return response.json()


def magento_get_logged(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    token: Optional[str] = None,
) -> Tuple[int, Optional[str], str, requests.Response]:
    """Perform a GET request capturing response metadata without raising."""

    final_url = build_magento_url(base_url, path)
    _LOGGER.debug("magento_get_logged URL=%s", final_url)
    auth_headers: Dict[str, str] = build_magento_headers(
        token=(token.strip() if isinstance(token, str) else None), session=session
    )
    if headers:
        auth_headers.update(headers)

    response = session.get(final_url, headers=auth_headers, timeout=timeout)
    status = response.status_code
    content_type = response.headers.get("Content-Type")
    body_snippet = response.text

    return status, content_type, body_snippet, response
