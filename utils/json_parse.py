"""Helpers for parsing JSON responses from LLMs."""

from __future__ import annotations

import json
import re
from typing import Iterable


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _sanitize_json_text(text: str) -> str:
    """Normalize minor formatting issues commonly produced by LLMs."""

    replacements = {
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Remove dangling commas before closing braces/brackets.
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _balance_closing_delimiters(text: str) -> str:
    """Append closing delimiters when the model truncated the JSON string."""

    brace_balance = 0
    bracket_balance = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            brace_balance += 1
        elif ch == "}":
            brace_balance = max(brace_balance - 1, 0)
        elif ch == "[":
            bracket_balance += 1
        elif ch == "]":
            bracket_balance = max(bracket_balance - 1, 0)

    if brace_balance:
        text += "}" * brace_balance
    if bracket_balance:
        text += "]" * bracket_balance
    return text


def _unwrap_single_mapping(value: object) -> object:
    """Return nested mapping value when only one key is present."""

    seen: set[int] = set()
    current = value
    while isinstance(current, dict) and len(current) == 1:
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        (_, inner), = current.items()
        if not isinstance(inner, dict):
            break
        current = inner
    return current


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
        cleaned = _sanitize_json_text(cleaned)

        start = cleaned.find("{")
        alt_start = cleaned.find("[")
        if (start == -1 or (0 <= alt_start < start)) and alt_start != -1:
            start = alt_start
        if start == -1:
            return None

        candidate = cleaned[start:]
        candidate = candidate.strip()
        candidate = _sanitize_json_text(candidate)

        def _attempt_parse(text: str) -> dict | list | None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        obj: dict | list | None = _attempt_parse(candidate)

        if obj is None:
            closing_matches = list(re.finditer(r"[}\]]", candidate))
            for match in reversed(closing_matches):
                snippet = candidate[: match.end()]
                parsed = _attempt_parse(snippet)
                if parsed is not None:
                    obj = parsed
                    break

        if obj is None:
            balanced = _balance_closing_delimiters(candidate)
            obj = _attempt_parse(balanced)

        if obj is None:
            return None

        if required_keys and isinstance(obj, dict):
            unwrapped = _unwrap_single_mapping(obj)
            if isinstance(unwrapped, dict):
                obj = unwrapped
            if not isinstance(obj, dict) or not all(key in obj for key in required_keys):
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

