"""HTTP utilities for configuring reusable sessions."""
from __future__ import annotations

from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def get_session(auth_token: Optional[str] = None) -> requests.Session:
    """Return a configured :class:`requests.Session` with retries.

    Parameters
    ----------
    auth_token:
        Magento admin bearer token. When provided, the ``Authorization`` header
        will be configured automatically.
    """

    session = requests.Session()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    session.headers.update(headers)

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
