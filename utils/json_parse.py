"""Helpers for parsing JSON responses from LLMs."""

from __future__ import annotations

import json
import re
from typing import Iterable


def extract_json_object(
    s: object, *, required_keys: Iterable[str] | None = None
) -> dict | list | None:
    """Best-effort extraction of the first JSON object embedded in ``s``.

    Parameters
    ----------
    s:
        Source text that potentially contains a JSON object or array.
    required_keys:
        Optional iterable with keys that must be present in the parsed result
        (only applied when the parsed value is a mapping).

    Returns
    -------
    dict | list | None
        Parsed JSON object (or array) or ``None`` if parsing fails.
    """

    if not isinstance(s, str):
        return None

    try:
        # remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE)
        start = cleaned.find("{")
        alt_start = cleaned.find("[")
        if (start == -1 or (0 <= alt_start < start)) and alt_start != -1:
            start = alt_start
        end = cleaned.rfind("}")
        alt_end = cleaned.rfind("]")
        if (end == -1 or (alt_end > end)) and alt_end != -1:
            end = alt_end
        if start == -1 or end == -1 or end <= start:
            return None
        sliced = cleaned[start : end + 1]
        obj = json.loads(sliced)
        if required_keys and isinstance(obj, dict):
            if not all(key in obj for key in required_keys):
                return None
        return obj
    except Exception:
        return None


def short_preview_of(value: object, *, max_len: int = 120) -> str:
    """Return a short preview string suitable for error messages."""

    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip())
    else:
        text = str(value or "").strip()

    if not text:
        return ""

    return text[:max_len]

