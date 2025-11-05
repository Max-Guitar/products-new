"""Utilities for sanitizing HTML fragments returned by LLM responses."""

from __future__ import annotations

import re

__all__ = ["sanitize_html"]


# Symbols that often appear before the HTML payload (BOM, emojis, whitespace).
_LEADING_JUNK = "\ufeff\u200b\u200c\u200d\ufe0f\u26a0⚠️ \t\n\r"


def sanitize_html(html: str) -> str:
    """Return ``html`` stripped from leading junk and balanced ``<p>`` tags.

    The AI occasionally responds with leading emojis/BOM characters or forgets
    to close ``<p>`` tags. Streamlit's markdown renderer is sensitive to these
    issues and may emit warnings or break the layout. This helper removes the
    unwanted prefixes and appends missing closing tags while leaving otherwise
    valid markup untouched.
    """

    if not isinstance(html, str):
        html = "" if html is None else str(html)

    sanitized = html.lstrip(_LEADING_JUNK)

    open_tags = len(
        re.findall(r"<p(?=[\s>])[^>]*>", sanitized, flags=re.IGNORECASE)
    )
    close_tags = len(re.findall(r"</p>", sanitized, flags=re.IGNORECASE))
    if close_tags < open_tags:
        sanitized += "</p>" * (open_tags - close_tags)

    return sanitized
