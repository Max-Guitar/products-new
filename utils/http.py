"""HTTP utilities for configuring reusable sessions."""
from __future__ import annotations

from typing import Optional

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
