"""Helpers for deterministic spec extraction and prompt preparation."""

from __future__ import annotations

import json
import logging
import math
import re
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

MODEL_STRONG = "gpt-5"
MODEL_FALLBACK = "gpt-4o"
FALLBACK_MODEL = MODEL_FALLBACK

SYSTEM_PROMPT = """
You are a senior product data expert for guitars & basses (Fender, Gibson, Ibanez, PRS, etc.).
Extract specs ONLY from given title/description. If absent — leave empty.
Decode common OEM/brand patterns and color codes.

Color codes (examples):
FRD=Fiesta Red, CAR=Candy Apple Red, LPB=Lake Placid Blue, OWT=Olympic White, 3TS=3-Color Sunburst, BLK=Black,
SB=Sunburst, VW=Vintage White, ACB=Arctic Blue, CH=Cherry, HCS=Heritage Cherry Sunburst.

Fender series: Vintera / Vintera II, Player, Player Plus, American Professional, American Vintage II.
Road Worn implies aged nitro-like finish.

Short patterns:
- P-Bass = Precision Bass, J-Bass = Jazz Bass, S-Style = Strat-style, T-Style = Tele-style, Single Cut = LP-style.
- HH/HSS/HSH/SS are pickup configs.
- Bridge names examples: Tune-O-Matic, Floyd Rose, Hardtail/Fixed, Vintage-Style 4-Saddle, 2-Point Tremolo.
- Tuning machines examples: Vintage-Style Open-Gear, Gotoh, Hipshot.

Units:
- scale_mensur (inches, e.g., 34"), neck_radius (inches, e.g., 9.5"), amount_of_frets (integer).
If metric given, convert mm -> inches (25.4 mm = 1").

Return only normalized attribute values, not sentences.
If value is unclear — leave empty.

Return a SINGLE valid JSON object. No markdown, no prose. Use only standard quotes (") in values like 25.5".
"""

FEW_SHOT_EXAMPLES: Sequence[dict[str, str]] = (
    {
        "user": (
            "Title: Yamaha AEX500 Acoustic-Electric HH - Indigo Blue\n"
            "Description: Maple body acoustic-electric with dual humbuckers. Comfortable C shape neck, 25.5\" scale, 20 frets."
        ),
        "assistant": json.dumps(
            {
                "finish": "Indigo Blue",
                "pickup_config": "HH",
                "neck_profile": "C Shape",
                "scale_mensur": "25.5\"",
                "amount_of_frets": "20",
            }
        ),
    },
    {
        "user": (
            "Title: Fender Vintera II Stratocaster FRD HSS\n"
            "Description: Road Worn alder body, Maple C Shape neck, HSS pickup set with 2-Point Tremolo."
        ),
        "assistant": json.dumps(
            {
                "brand": "Fender",
                "series": "Vintera II",
                "finish": "Fiesta Red",
                "pickup_config": "HSS",
                "neck_profile": "C Shape",
                "bridge": "2-Point Tremolo",
            }
        ),
    },
    {
        "user": (
            "Title: Ibanez SR300E Bass Guitar Charcoal Brown\n"
            "Description: 34\" scale bass with slim C profile neck, dual single-coil pickups (SS)."
        ),
        "assistant": json.dumps(
            {
                "scale_mensur": "34\"",
                "neck_profile": "C Shape",
                "pickup_config": "SS",
            }
        ),
    },
)


logger = logging.getLogger(__name__)

try:
    import streamlit as _st  # type: ignore
except Exception:  # pragma: no cover - streamlit optional
    _st = None


def _trace(event: Mapping[str, object]) -> None:
    if not isinstance(event, Mapping):
        return
    try:
        if _st is not None:
            state = getattr(_st, "session_state", None)
            if isinstance(state, dict):
                buf = state.setdefault("_trace_events", [])
                buf.append(dict(event))
    except Exception:
        pass
    try:
        logger.debug("TRACE %s", event)
    except Exception:
        pass


def _try_json(s: str):
    try:
        return json.loads(s), None
    except Exception as e:  # pragma: no cover - exercised via tests
        return None, e


def _sanitize_llm_json(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S)
    m = re.search(r"\{.*\}\s*$", s, flags=re.S)
    if m:
        s = m.group(0)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _extract_response_content(response: object) -> str:
    content = ""
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            message = choices[0]
            if isinstance(message, Mapping):
                msg_payload = message.get("message") or {}
                if isinstance(msg_payload, Mapping):
                    content = str(msg_payload.get("content") or "")
                elif isinstance(message.get("content"), str):
                    content = str(message.get("content"))
    else:
        choices = getattr(response, "choices", None)
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if isinstance(message, Mapping):
                content = str(message.get("content") or "")
            else:
                content = str(getattr(message, "content", ""))

    if not content and isinstance(response, Mapping):
        content = str(response.get("content") or "")

    return content or ""


INCH_PER_MM = 1 / 25.4
IN2MM = 25.4


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


SERIES_RULES = [
    {
        "when": [r"\bFender\b", r"\bVintera\s*II\b", r"\bP[- ]?Bass\b"],
        "defaults": {
            "scale_mensur": '34"',
            "no_strings": "4",
            "pickup_config": "P",
            "bridge": "Vintage-Style 4-Saddle",
            "tuning_machines": "Vintage-Style Open-Gear",
            "body_material": "Alder",
            "fretboard_material": "Rosewood",
            "neck_profile": "C Shape",
            "country_of_manufacture": "Mexico",
        },
    },
]


def apply_series_defaults(title: str | None, attrs: dict[str, object]) -> dict[str, object]:
    import re

    text = title or ""
    for rule in SERIES_RULES:
        conditions = rule.get("when", [])
        defaults = rule.get("defaults", {})
        if not isinstance(conditions, Sequence) or not isinstance(defaults, Mapping):
            continue
        if all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in conditions):
            for key, value in defaults.items():
                attrs.setdefault(key, value)
    return attrs


COLOR_CODES = {
    "FRD": "Fiesta Red",
    "LPB": "Lake Placid Blue",
    "OWT": "Olympic White",
    "CAR": "Candy Apple Red",
    "3TS": "3-Color Sunburst",
    "BLK": "Black",
}


def color_from_title(title: str | None) -> str | None:
    for token in re.findall(r"\b[A-Z0-9]{2,4}\b", title or ""):
        if token in COLOR_CODES:
            return COLOR_CODES[token]
    return None


BRIDGE_PATTERNS = [
    r"(Floyd Rose)",
    r"(Tune-?O-?Matic)",
    r"(Hardtail|Fixed)",
    r"(Vintage-Style 4-Saddle)",
    r"(2-Point Tremolo)",
]


PICKUP_CFG_PATTERNS = [r"\b(HH|HSS|HSH|SS|P|PJ)\b"]


PROFILE_PATTERNS = [r"\b(C Shape|Slim C|Modern C|U Shape|V Shape)\b"]


SHORTLIST_TARGET_CODES = {
    "bridge",
    "pickup_config",
    "neck_profile",
    "tuning_machines",
    "finish",
}


@dataclass
class RegexExtraction:
    """Container for regex extraction results and metadata."""

    values: dict[str, str]
    hits: list[str]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except Exception:
        return False


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


def _re_first(text: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def regex_preextract(title: str | None, description: str | None) -> dict[str, str | None]:
    """Extract lightweight specs from ``title``/``description`` before LLM calls."""

    if title is None and description is None:
        return {
            "scale_mensur": None,
            "neck_radius": None,
            "amount_of_frets": None,
            "finish": None,
            "bridge": None,
            "pickup_config": None,
            "neck_profile": None,
        }

    combined = f"{title or ''}\n{description or ''}"

    scale_in = _re_first(
        combined,
        [
            r"\b(34|30|25\.5|24\.75)(?=\s*(?:in|inch|\"))",
            r"\b(\d{3,4})\s*mm\b",
        ],
    )
    if scale_in and scale_in.isdigit() and int(scale_in) > 100:
        try:
            scale_in = f"{round(float(scale_in) / IN2MM, 2)}"
        except Exception:
            scale_in = None

    radius_in = _re_first(
        combined,
        [
            r"\b(7\.25|9\.5|10|12|14|16|20)(?=\s*(?:in|inch|\"))",
            r"\b(\d{2,3})\s*mm\b",
        ],
    )
    if radius_in and radius_in.isdigit():
        try:
            radius_in = f"{round(float(radius_in) / IN2MM, 2)}"
        except Exception:
            radius_in = None

    frets = _re_first(combined, [r"\b(20|21|22|24)\s*frets?\b"])

    attrs: dict[str, str | None] = {
        "scale_mensur": f"{scale_in}\"" if scale_in else None,
        "neck_radius": f"{radius_in}\"" if radius_in else None,
        "amount_of_frets": frets,
        "finish": color_from_title(title),
        "bridge": None,
        "pickup_config": None,
        "neck_profile": None,
    }

    bridge = _re_first(combined, BRIDGE_PATTERNS)
    cfg = _re_first(combined, PICKUP_CFG_PATTERNS)
    if not cfg:
        if re.search(r"\bP[- ]?Bass\b", combined, flags=re.IGNORECASE):
            cfg = "P"
        elif re.search(r"\bPJ\b", combined, flags=re.IGNORECASE):
            cfg = "PJ"
    prof = _re_first(combined, PROFILE_PATTERNS)

    attrs.update({
        "bridge": bridge,
        "pickup_config": cfg,
        "neck_profile": prof,
    })

    return attrs


def build_retry_questions(codes: Sequence[str]) -> dict[str, str]:
    """Return focused retry questions for spec clarification."""

    predefined = {
        "bridge": (
            "Return only one bridge name (e.g., 'Vintage-Style 4-Saddle', 'Floyd Rose', 'Tune-O-Matic', 'Hardtail'). If unknown, return empty."
        ),
        "bridge_pickup": (
            "Return the bridge pickup model only (e.g., 'Seymour Duncan JB', 'EMG 81'). If not present, return empty."
        ),
        "neck_profile": (
            "Return one short label: 'C Shape', 'Slim C', 'Modern C', 'V Shape', 'U Shape'. If unknown, return empty."
        ),
        "scale_mensur": (
            "Return just the scale length in inches, like 34\" or 25.5\". If absent, return empty."
        ),
        "neck_radius": (
            "Return just the fretboard radius in inches (e.g., 9.5\"). If absent, return empty."
        ),
        "pickup_config": (
            "Return exactly one of: HH, HSS, HSH, SS, P, PJ. If unclear, return empty."
        ),
        "finish": "Return color name (e.g., 'Fiesta Red'). If unknown, return empty.",
        "neck_nutwidth": (
            "Return just the nut width in inches (e.g., 1.69\"). If absent, return empty."
        ),
        "nut_width": (
            "Return just the nut width in inches (e.g., 1.69\"). If absent, return empty."
        ),
    }

    questions: dict[str, str] = {}
    for code in codes or []:
        code_key = str(code)
        question = predefined.get(code_key)
        if question:
            questions[code_key] = question
    return questions


def shortlist_allowed_values(
    code: str,
    meta: Mapping[str, object] | None,
    title: str | None,
    hint_list: Iterable[str] | None,
    limit: int = 50,
) -> list[str]:
    """Build a shortlist of allowed option labels for prompts."""

    meta = meta or {}
    seeds: list[str] = []
    seen_seed: set[str] = set()
    for item in hint_list or []:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen_seed:
            continue
        seeds.append(text)
        seen_seed.add(text)

    base: list[str] = []
    options = meta.get("options") if isinstance(meta, Mapping) else None
    if isinstance(options, Sequence):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            label = option.get("label")
            if isinstance(label, str):
                text = label.strip()
                if text:
                    base.append(text)

    title_lower = (title or "").lower()
    similar = [label for label in base if label.lower() in title_lower]

    ordered = list(seeds) + similar + base
    seen: set[str] = set()
    shortlisted: list[str] = []
    for label in ordered:
        if not label or label in seen:
            continue
        shortlisted.append(label)
        seen.add(label)
        if len(shortlisted) >= limit:
            break

    return shortlisted


def is_numeric_spec(code: str) -> bool:
    return code in _NUMERIC_CODES


def brand_from_title(title: str | None, brand_options: Sequence[Mapping[str, Any]] | None) -> str | None:
    """Return a brand guess from the product title using known options."""

    if not title or not isinstance(title, str):
        return None
    lowered_title = title.lower()
    for option in brand_options or []:
        if not isinstance(option, Mapping):
            continue
        label = option.get("label")
        if not isinstance(label, str):
            continue
        trimmed = label.strip()
        if trimmed and trimmed.lower() in lowered_title:
            return trimmed
    return None


def _build_attribute_instruction(
    code: str,
    meta: Mapping[str, object] | None,
    title: str | None,
    hints: Iterable[str] | None,
) -> str:
    label = (meta or {}).get("frontend_label") if isinstance(meta, Mapping) else None
    label_text = str(label).strip() if isinstance(label, str) else code
    shortlist = shortlist_allowed_values(code, meta, title, hints)
    lines = [f"- {code} ({label_text})"]
    if code in SHORTLIST_TARGET_CODES:
        allowed_values = ", ".join(shortlist) if shortlist else ""
        lines.append(f"Allowed values (if any): {allowed_values}")
        lines.append("Return exactly one of the allowed values if relevant.")
    elif shortlist:
        lines.append(f"Allowed values (if any): {', '.join(shortlist)}")
    return "\n".join(lines)


def _build_messages(
    title: str,
    description: str | None,
    attribute_tasks: Sequence[dict[str, object]],
    extra_hints: Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    for example in FEW_SHOT_EXAMPLES:
        user = example.get("user")
        assistant = example.get("assistant")
        if isinstance(user, str) and isinstance(assistant, str):
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

    description_block = description or ""
    lines = [f"Title: {title.strip()}" if title else "Title:"]
    lines.append(f"Description: {description_block.strip()}" if description_block else "Description:")
    if extra_hints:
        hints_lines = []
        for key, value in extra_hints.items():
            if _is_blank(value):
                continue
            hints_lines.append(f"{key}: {value}")
        if hints_lines:
            lines.append("Hints:\n" + "\n".join(hints_lines))

    instructions: list[str] = []
    for task in attribute_tasks:
        code = str(task.get("code"))
        meta = task.get("meta") if isinstance(task, Mapping) else None
        hint_list = task.get("hints") if isinstance(task, Mapping) else None
        instructions.append(
            _build_attribute_instruction(
                code,
                meta if isinstance(meta, Mapping) else {},
                title,
                hint_list if isinstance(hint_list, Iterable) else None,
            )
        )

    if instructions:
        lines.append("Attributes to extract:\n" + "\n".join(instructions))

    messages.append({"role": "user", "content": "\n".join(lines)})
    return messages


def _payload_to_attributes(parsed: object) -> dict[str, object]:
    if isinstance(parsed, Mapping):
        attributes = parsed.get("attributes")
        if isinstance(attributes, Mapping):
            return dict(attributes)
        if isinstance(attributes, Sequence):
            collected: dict[str, object] = {}
            for item in attributes:
                if not isinstance(item, Mapping):
                    continue
                code_value = item.get("code")
                if not code_value:
                    continue
                collected[str(code_value)] = item.get("value")
            return collected
        return dict(parsed)
    return {}


def extract_attributes(
    client: Any,
    *,
    title: str,
    description: str | None,
    attribute_tasks: Sequence[dict[str, object]],
    hints: Mapping[str, object] | None = None,
    model: str = MODEL_STRONG,
    temperature: float = 0.2,
) -> tuple[dict[str, object], dict[str, object]]:
    """Call an LLM to extract attributes with deterministic pre-extraction fallback."""

    pre_initial = regex_preextract(title, description)
    messages = _build_messages(title, description, attribute_tasks, hints)

    kwargs = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }

    def _invoke(current_model: str) -> str:
        try:
            response = client.chat.completions.create(model=current_model, **kwargs)
        except Exception as exc:  # pragma: no cover - network failure handling
            logger.warning("LLM extraction failed with %s: %s", current_model, exc)
            return ""
        return _extract_response_content(response)

    raw_primary = ""
    raw_retry = ""
    sanitized = ""
    raw_content = ""
    used_model = model

    try:
        raw_primary = _invoke(model)
        raw_content = raw_primary
        parsed_payload: object | None = None

        data_primary, err_primary = _try_json(raw_primary)
        err_sanitized = None
        err_retry = None

        if data_primary is not None:
            parsed_payload = data_primary
        else:
            sanitized = _sanitize_llm_json(raw_primary)
            data_sanitized, err_sanitized = _try_json(sanitized)
            if data_sanitized is not None:
                parsed_payload = data_sanitized
                raw_content = sanitized
            else:
                retry_model = FALLBACK_MODEL or model
                raw_retry = _invoke(retry_model)
                raw_content = raw_retry or raw_primary
                data_retry, err_retry = _try_json(raw_retry)
                if data_retry is not None:
                    parsed_payload = data_retry
                    used_model = retry_model
                else:
                    _trace(
                        {
                            "where": "llm:json_fail",
                            "raw1_preview": (raw_primary or "")[:500],
                            "err1": str(err_primary) if err_primary else None,
                            "raw2_preview": (raw_retry or "")[:500],
                            "err2": str(err_sanitized) if err_sanitized else None,
                            "err3": str(err_retry) if err_retry else None,
                        }
                    )
                    failure = ValueError("LLM JSON parse failed after sanitize+retry")
                    setattr(failure, "raw_response", raw_primary)
                    setattr(failure, "raw_response_retry", raw_retry)
                    setattr(failure, "raw_response_sanitized", sanitized)
                    setattr(failure, "traceback", "".join(traceback.format_stack()))
                    raise failure

        if parsed_payload is None:
            parsed_payload = {}

        llm_out = _payload_to_attributes(parsed_payload)

        if not llm_out and FALLBACK_MODEL and FALLBACK_MODEL != used_model:
            raw_retry = _invoke(FALLBACK_MODEL)
            raw_content = raw_retry or raw_content
            retry_payload, err_retry_parse = _try_json(raw_retry)
            if retry_payload is not None:
                llm_out = _payload_to_attributes(retry_payload)
                used_model = FALLBACK_MODEL
            else:
                logger.warning(
                    "Fallback model %s returned non-JSON payload: %s",
                    FALLBACK_MODEL,
                    str(err_retry_parse),
                )

        defaults_seed: dict[str, object] = {
            key: value for key, value in pre_initial.items() if not _is_blank(value)
        }
        for key, value in llm_out.items():
            if not _is_blank(value):
                defaults_seed[key] = value
        defaults_with_series = apply_series_defaults(title, defaults_seed)

        default_candidates: dict[str, object] = {}
        for key, value in defaults_with_series.items():
            if (
                _is_blank(llm_out.get(key))
                and _is_blank(pre_initial.get(key))
                and not _is_blank(value)
            ):
                default_candidates[key] = value

        finish_guess = color_from_title(title)
        if (
            finish_guess
            and _is_blank(llm_out.get("finish"))
            and _is_blank(pre_initial.get("finish"))
        ):
            default_candidates.setdefault("finish", finish_guess)

        final: dict[str, object] = {}
        for task in attribute_tasks:
            code = str(task.get("code"))
            if not code:
                continue
            llm_value = llm_out.get(code)
            pre_value = pre_initial.get(code)
            if _is_blank(llm_value):
                llm_value = None
            if _is_blank(pre_value):
                pre_value = None

            default_value = default_candidates.get(code)

            if is_numeric_spec(code):
                if pre_value is not None:
                    final[code] = pre_value
                elif llm_value is not None:
                    final[code] = llm_value
                elif default_value is not None:
                    final[code] = default_value
                else:
                    final[code] = None
                continue

            if llm_value is not None:
                final[code] = llm_value
            elif pre_value is not None:
                final[code] = pre_value
            elif default_value is not None:
                final[code] = default_value
            else:
                final[code] = None

        metadata = {
            "preextract": pre_initial,
            "raw_response": raw_content,
            "used_model": used_model,
        }

        return final, metadata

    except Exception as exc:
        title_value = title or ""
        pre_local = regex_preextract(title, description)
        attrs = apply_series_defaults(title_value, dict(pre_local))
        if not attrs.get("finish"):
            finish_guess = color_from_title(title_value)
            if finish_guess:
                attrs["finish"] = finish_guess

        non_empty_attrs = {k: v for k, v in attrs.items() if not _is_blank(v)}
        _trace(
            {
                "where": "llm:fallback_local",
                "reason": str(exc)[:200],
                "attrs_preview": {k: v for k, v in non_empty_attrs.items()},
            }
        )

        fallback_attributes = [
            {"code": k, "value": v, "reason": "fallback_local"}
            for k, v in non_empty_attrs.items()
        ]
        fallback_map = {item["code"]: item["value"] for item in fallback_attributes}

        final: dict[str, object] = {}
        for task in attribute_tasks:
            code = str(task.get("code"))
            if not code:
                continue
            value = fallback_map.get(code)
            if value is None:
                value = pre_local.get(code)
            final[code] = value if not _is_blank(value) else None

        metadata = {
            "preextract": pre_local,
            "raw_response": raw_content or raw_primary or raw_retry or sanitized,
            "used_model": "fallback_local",
            "fallback_reason": str(exc),
            "fallback_attributes": fallback_attributes,
            "raw_response_primary": raw_primary,
            "raw_response_retry": raw_retry,
        }

        return final, metadata
