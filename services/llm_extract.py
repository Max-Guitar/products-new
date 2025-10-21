"""Lightweight helpers for extracting attribute hints before LLM calls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


INCH_PER_MM = 1 / 25.4


_SCALE_PATTERN = re.compile(
    r"(?i)\b(25\.5|24\.75|34|30)\s*(?:inch|in|\")\b|\b(\d{3,4})\s*mm\b"
)
_RADIUS_PATTERN = re.compile(
    r"(?i)\b(7\.25|9\.5|10|12|14|16|20)\s*(?:inch|in|\")\b|\b(\d{2,3})\s*mm\b"
)
_FRETS_PATTERN = re.compile(r"(?i)\b(20|21|22|24)\s*(frets?)\b")
_NUT_WIDTH_PATTERN = re.compile(
    r"(?i)\b(1\.65|1\.6875|1\.75)\s*(?:in|\")\b|\b(41\.?3|42|43)\s*mm\b"
)

_MATERIAL_KEYWORDS = {
    "mahogany",
    "maple",
    "spruce",
    "rosewood",
    "ebony",
    "alder",
    "ash",
}

_MATERIAL_PRIORITY: dict[str, Sequence[str]] = {
    "top_material": ("spruce", "maple"),
    "fretboard_material": ("rosewood", "ebony", "maple"),
    "neck_material": ("mahogany", "maple"),
    "body_material": ("mahogany", "alder", "ash", "maple"),
}

_NUMERIC_CODES = {
    "scale_mensur",
    "neck_radius",
    "amount_of_frets",
    "neck_nutwidth",
    "nut_width",
}


@dataclass
class RegexExtraction:
    """Container for regex extraction results and metadata."""

    values: dict[str, str]
    hits: list[str]


def _format_inches(value: float | int | str) -> str | None:
    """Normalize ``value`` to a string in inches with two decimals at most."""

    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
    else:
        numeric = float(value)

    rounded = round(numeric, 2)
    if math.isclose(rounded, int(round(rounded))):
        return f"{int(round(rounded))}\""
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return f"{text}\""


def _convert_mm_to_inches(value_mm: str) -> str | None:
    try:
        numeric = float(value_mm)
    except (TypeError, ValueError):
        return None
    inches = numeric * INCH_PER_MM
    return _format_inches(inches)


def _match_materials(text: str) -> dict[str, str]:
    matches: list[tuple[int, str]] = []
    lowered = text.lower()
    for keyword in _MATERIAL_KEYWORDS:
        for match in re.finditer(rf"\b{re.escape(keyword)}\b", lowered):
            matches.append((match.start(), keyword))
    matches.sort(key=lambda item: item[0])

    assigned: dict[str, str] = {}
    used_keywords: set[str] = set()

    for attr, priority in _MATERIAL_PRIORITY.items():
        for position, keyword in matches:
            if keyword not in priority:
                continue
            if attr in assigned:
                break
            if keyword in used_keywords and keyword != "maple":
                # Allow "maple" to fill multiple slots if nothing better appears.
                continue
            assigned[attr] = keyword
            used_keywords.add(keyword)
            break

    # Fallback pass for remaining matches (fill first unassigned attr if unique keyword)
    for position, keyword in matches:
        if keyword in used_keywords:
            continue
        for attr in ("body_material", "top_material", "neck_material", "fretboard_material"):
            if attr not in assigned:
                assigned[attr] = keyword
                used_keywords.add(keyword)
                break

    return assigned


def regex_extract(text: object) -> RegexExtraction:
    """Extract deterministic hints (scale, radius, materials…) from ``text``."""

    if not isinstance(text, str) or not text.strip():
        return RegexExtraction(values={}, hits=[])

    normalized_text = text.strip()
    values: dict[str, str] = {}
    hits: list[str] = []

    scale_match = _SCALE_PATTERN.search(normalized_text)
    if scale_match:
        inch_value = scale_match.group(1)
        mm_value = scale_match.group(2)
        if inch_value:
            formatted = _format_inches(inch_value)
        else:
            formatted = _convert_mm_to_inches(mm_value)
        if formatted:
            values["scale_mensur"] = formatted
            hits.append("scale_mensur")

    radius_match = _RADIUS_PATTERN.search(normalized_text)
    if radius_match:
        inch_value = radius_match.group(1)
        mm_value = radius_match.group(2)
        if inch_value:
            formatted = _format_inches(inch_value)
        else:
            formatted = _convert_mm_to_inches(mm_value)
        if formatted:
            values["neck_radius"] = formatted
            hits.append("neck_radius")

    frets_match = _FRETS_PATTERN.search(normalized_text)
    if frets_match:
        count = frets_match.group(1)
        if count:
            values["amount_of_frets"] = count
            hits.append("amount_of_frets")

    nut_match = _NUT_WIDTH_PATTERN.search(normalized_text)
    if nut_match:
        inch_value = nut_match.group(1)
        mm_value = nut_match.group(2)
        if inch_value:
            formatted = _format_inches(inch_value)
        else:
            formatted = _convert_mm_to_inches(mm_value)
        if formatted:
            values["neck_nutwidth"] = formatted
            values["nut_width"] = formatted
            hits.append("neck_nutwidth")

    material_matches = _match_materials(normalized_text)
    for code, material in material_matches.items():
        if material:
            values.setdefault(code, material)
            if code not in hits:
                hits.append(code)

    return RegexExtraction(values=values, hits=hits)


def build_retry_questions(codes: Sequence[str]) -> dict[str, str]:
    """Return focused retry questions for numeric specs."""

    questions: dict[str, str] = {}
    for code in codes or []:
        code_key = str(code)
        if code_key == "scale_mensur":
            questions[code_key] = 'Return just a number with inches (e.g., 25.5\") if present, else empty.'
        elif code_key == "neck_radius":
            questions[code_key] = 'Return just a number with inches (e.g., 9.5\") if present, else empty.'
        elif code_key in {"neck_nutwidth", "nut_width"}:
            questions[code_key] = 'Return just a number with inches (e.g., 1.69\") if present, else empty.'
    return questions


def shortlist_allowed_values(
    code: str,
    meta: Mapping[str, object] | None,
    *,
    current: Iterable[str] | None = None,
    hints: Iterable[str] | None = None,
    limit: int = 50,
) -> list[str]:
    """Return up to ``limit`` allowed option labels for prompts."""

    meta = meta or {}
    options = meta.get("options") if isinstance(meta, Mapping) else None
    labels: list[str] = []
    if isinstance(options, Sequence):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            label = option.get("label")
            if not isinstance(label, str):
                continue
            text = label.strip()
            if text and text not in labels:
                labels.append(text)

    def _extend(values: Iterable[str] | None) -> None:
        for item in values or []:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in labels:
                labels.insert(0, text)

    _extend(current)
    _extend(hints)

    if len(labels) > limit:
        labels = labels[:limit]

    return labels


def is_numeric_spec(code: str) -> bool:
    return code in _NUMERIC_CODES

