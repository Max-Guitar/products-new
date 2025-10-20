"""AI-assisted attribute autofill helpers based on the Colab reference notebook."""

from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import openai
import pandas as pd
import requests
import streamlit as st

from utils.http import MAGENTO_REST_PATH_CANDIDATES, build_magento_headers


ALWAYS_ATTRS: Set[str] = {"brand", "country_of_manufacture", "short_description"}

ELECTRIC_SET_IDS: Set[int] = {12, 16}

SET_ATTRS: Mapping[str, Set[str]] = {
    "Accessories": {
        "condition",
        "accessory_type",
        "strings",
        "cases_covers",
        "cables",
        "merchandise",
        "parts",
    },
    "Acoustic guitar": {
        "condition",
        "series",
        "acoustic_guitar_style",
        "acoustic_body_shape",
        "body_material",
        "top_material",
        "neck_profile",
        "electro_acoustic",
        "acoustic_cutaway",
        "no_strings",
        "orientation",
        "vintage",
        "kids_size",
        "finish",
        "bridge",
        "controls",
        "neck_material",
        "neck_radius",
        "tuning_machines",
        "fretboard_material",
        "scale_mensur",
        "amount_of_frets",
        "acoustic_pickup",
        "cases_covers",
    },
    "Amps": {
        "amp_style",
        "condition",
        "type",
        "speaker_configuration",
        "built_in_fx",
        "cover_included",
        "footswitch_included",
        "vintage",
    },
    "Bass Guitar": {
        "condition",
        "guitarstylemultiplechoice",
        "series",
        "model",
        "acoustic_bass",
        "pickup_config",
        "body_material",
        "top_material",
        "neck_profile",
        "no_strings",
        "orientation",
        "vintage",
        "kids_size",
        "finish",
        "bridge",
        "controls",
        "bridge_pickup",
        "neck_material",
        "neck_radius",
        "middle_pickup",
        "neck_pickup",
        "neck_nutwidth",
        "tuning_machines",
        "fretboard_material",
        "scale_mensur",
        "amount_of_frets",
        "cases_covers",
    },
    "Default": {"name", "price"},
    "Effects": {
        "condition",
        "effect_type",
        "vintage",
        "controls",
        "power",
        "power_polarity",
        "battery",
    },
    "Electric guitar": {
        "condition",
        "guitarstylemultiplechoice",
        "series",
        "model",
        "semi_hollow_body",
        "body_material",
        "top_material",
        "bridge_type",
        "pickup_config",
        "neck_profile",
        "no_strings",
        "orientation",
        "vintage",
        "kids_size",
        "finish",
        "bridge",
        "bridge_pickup",
        "middle_pickup",
        "neck_pickup",
        "controls",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "tuning_machines",
        "fretboard_material",
        "scale_mensur",
        "amount_of_frets",
        "cases_covers",
    },
}

SPEC_FORCE_CODES: Set[str] = {
    "bridge",
    "bridge_type",
    "bridge_pickup",
    "middle_pickup",
    "neck_pickup",
    "pickup_config",
    "scale_mensur",
    "neck_radius",
    "neck_nutwidth",
    "fretboard_material",
    "neck_material",
    "tuning_machines",
    "controls",
    "finish",
    "top_material",
    "body_material",
}

MEASUREMENT_CODES: Set[str] = {"scale_mensur", "neck_radius", "neck_nutwidth"}

SPEC_CODE_QUESTIONS: Mapping[str, str] = {
    "bridge": "Extract the bridge hardware name if mentioned; otherwise leave it empty.",
    "bridge_type": "Extract the bridge type (e.g. tremolo, hardtail) if present; otherwise leave empty.",
    "bridge_pickup": "List the bridge pickup model if described; otherwise leave it empty.",
    "middle_pickup": "List the middle pickup model if described; otherwise leave it empty.",
    "neck_pickup": "List the neck pickup model if described; otherwise leave it empty.",
    "pickup_config": "Extract the pickup configuration code (HH, HSS, etc.) if present; otherwise leave empty.",
    "scale_mensur": "Extract the scale length (e.g. 25.5\") if present; otherwise leave empty.",
    "neck_radius": "Extract the fretboard radius (e.g. 9.5\") if present; otherwise leave empty.",
    "neck_nutwidth": "Extract the nut width in mm if mentioned; otherwise leave empty.",
    "fretboard_material": "Extract the fretboard material if present; otherwise leave empty.",
    "neck_material": "Extract the neck material if present; otherwise leave empty.",
    "tuning_machines": "Extract the tuning machines type/brand if present; otherwise leave empty.",
    "controls": "Extract the controls layout (e.g. volume/tone) if described; otherwise leave empty.",
    "finish": "Extract the finish description if present; otherwise leave empty.",
    "top_material": "Extract the top wood/material if present; otherwise leave empty.",
}


def _normalize_pickup_config(value: str) -> str:
    cleaned = re.sub(r"[^HPJMS]", "", value.upper())
    replacements = {
        "HSS": "HSS",
        "HSH": "HSH",
    }
    if not cleaned:
        cleaned = value.upper()
    if cleaned in replacements:
        return replacements[cleaned]
    return cleaned


def _format_scale_match(match: re.Match) -> str:
    if match.groupdict().get("int"):
        inches = float(match.group("int")) + 0.5
        value = f"{inches:.2f}".rstrip("0").rstrip(".")
        return f"{value}\""
    value = match.group("val")
    unit = match.group("unit") or ""
    if unit.strip().lower().startswith("mm"):
        try:
            inches = float(value) / 25.4
            rounded = f"{inches:.2f}".rstrip("0").rstrip(".")
            return f"{rounded}\""
        except (TypeError, ValueError):
            return f"{value}mm"
    trimmed = str(value).rstrip("\"")
    return f"{trimmed}\""


def _format_radius_match(match: re.Match) -> str:
    value = match.group("val")
    unit = match.group("unit") or ""
    if unit.strip().lower().startswith("mm"):
        try:
            inches = float(value) / 25.4
            rounded = f"{inches:.2f}".rstrip("0").rstrip(".")
            return f"{rounded}\""
        except (TypeError, ValueError):
            return f"{value}mm"
    trimmed = str(value).rstrip("\"")
    return f"{trimmed}\""


def _format_nutwidth_match(match: re.Match) -> str:
    value = match.group("val")
    unit = match.group("unit") or ""
    if unit.strip().lower().startswith("mm"):
        return f"{value}mm"
    try:
        mm_value = float(value) * 25.4
        return f"{mm_value:.1f}mm"
    except (TypeError, ValueError):
        return f"{value}\""


REGEX_DETECTION_PATTERNS: Mapping[str, dict] = {
    "scale": {
        "attribute": "scale_mensur",
        "pattern": (
            re.compile(
                r"(?:scale (?:length|mensur)[^\d]{0,6})?(?P<val>\d+(?:\.\d+)?)(?P<unit>\s*(?:\"|inch(?:es)?|in))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm)(?:\b|[^a-z])",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<int>\d+)\s*(?:1/2|½)(?P<unit>\s*(?:\"|inch(?:es)?|in))",
                re.IGNORECASE,
            ),
        ),
        "formatter": lambda match: _format_scale_match(match),
    },
    "radius": {
        "attribute": "neck_radius",
        "pattern": (
            re.compile(r"(?P<val>7\\.25|9\\.5|10|12|14|16)(?P<unit>\s*(?:\"|inch(?:es)?|in)?)", re.IGNORECASE),
            re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm)(?:\b|[^a-z])", re.IGNORECASE),
        ),
        "formatter": lambda match: _format_radius_match(match),
    },
    "nut": {
        "attribute": "neck_nutwidth",
        "pattern": (
            re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mm)(?:\b|[^a-z])", re.IGNORECASE),
            re.compile(r"(?P<val>1\.65|1\.6875|1\.75)(?P<unit>\s*(?:\"|inch(?:es)?|in))", re.IGNORECASE),
        ),
        "formatter": lambda match: _format_nutwidth_match(match),
    },
    "pickup_config": {
        "attribute": "pickup_config",
        "pattern": re.compile(r"\b(H\s*/?\s*H|H\s*SS|S\s*SS|H\s*S|P\s*/?\s*J|MM)\b", re.IGNORECASE),
        "formatter": lambda match: _normalize_pickup_config(match.group(0)),
    },
    "pickup_models": {
        "attribute": None,
        "pattern": (
            re.compile(
                r"(Seymour Duncan [A-Za-z0-9 \-]+|DiMarzio [A-Za-z0-9 \-]+|Fender [A-Za-z0-9 \-]+|EMG [A-Z0-9\-]+|Fishman [A-Za-z0-9 \-]+|Bare Knuckle [A-Za-z0-9 \-]+)",
                re.IGNORECASE,
            ),
        ),
        "formatter": lambda match: match.group(1),
    },
    "bridges": {
        "attribute": "bridge",
        "pattern": (
            re.compile(
                r"(Floyd Rose|Tune-o-matic|Hardtail|6-saddle|2-point|Bigsby|Hipshot [A-Za-z0-9\-]+|Gotoh [A-Za-z0-9\-]+|Babicz [A-Za-z0-9\-]+)",
                re.IGNORECASE,
            ),
        ),
        "formatter": lambda match: match.group(1),
    },
    "materials": {
        "attribute": None,
        "pattern": re.compile(r"\b(Maple|Rosewood|Ebony|Alder|Ash|Mahogany)\b", re.IGNORECASE),
        "formatter": lambda match: match.group(1).title(),
    },
}

MATERIAL_ALIAS_MAP: Mapping[str, str] = {"rosewood": "Indian Rosewood"}

def _load_brand_lexicon() -> dict[str, object]:
    path = Path(__file__).with_name("brand_lexicon.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


BRAND_LEXICON: Mapping[str, object] = _load_brand_lexicon()


def _load_series_lexicon() -> dict[str, object]:
    path = Path(__file__).with_name("series_lexicon.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


SERIES_LEXICON: Mapping[str, object] = _load_series_lexicon()

AI_RULES_TEXT = (
    "Ты помощник по каталогизации гитар. Используй предоставленный контекст, "
    "чтобы заполнить атрибуты товара и всегда проверяй, что значение подтверждено "
    "исходным текстом. Отвечай строго валидным JSON-объектом без пояснений: "
    "{\"attributes\": [...]}\n"
    "Каждый элемент массива обязан соответствовать схеме:\n"
    "{\"code\": <string>, \"value\": <string|number|boolean|array<string>>, "
    "\"reason\": <string>, \"evidence\": <string>}\n"
    "Поле evidence обязательно и должно содержать дословный фрагмент источника длиной 5–80 символов.\n"
    "Если данных нет, ставь value как пустую строку или [], а в reason и evidence объясняй, что "
    "информация отсутствует.\n"
    "Используй hints.regex_hits, hints.brand_lexicon и hints.series_lexicon только как подсказки; "
    "подтверждай их прямыми цитатами.\n"
    "Нормализуй стили: Telecaster→T-Style, Stratocaster→S-Style, Les Paul→Single cut, SG→Double cut; "
    "для T-Style добавляй Single cut, для S-Style — Double cut.\n"
    "Мультиселекты возвращай массивом уникальных строк, булевы — true/false. Для категорий используй текстовые лейблы.\n"
    "- For bass guitars, the default number of strings (no_strings) is never 6.\n"
    "  Most basses have 4 or 5 strings, 6 is rare.\n"
    "  Always check product name, model, or description for a digit 4/5/6 near the word \"string(s)\" or \"bass\".\n"
    "  If no explicit count found and it’s a bass guitar, prefer 4.\n"
    "Пример 1:\n"
    "Вход → name: \"Fender Player Stratocaster\", hints.regex_hits.scale: 25.5\"\n"
    "Ответ → {\"attributes\":[{\"code\":\"scale_mensur\",\"value\":\"25.5\"\",\"reason\":\"Стандартная мензура для Player Stratocaster\",\"evidence\":\"scale length 25.5\"\"}]}\n"
    "Пример 2:\n"
    "Вход → description содержит \"Mahogany body\"\n"
    "Ответ → {\"attributes\":[{\"code\":\"body_material\",\"value\":\"Mahogany\",\"reason\":\"В описании указано дерево корпуса\",\"evidence\":\"Mahogany body\"}]}\n"
)

AI_RULES_HASH = hashlib.sha256(AI_RULES_TEXT.encode("utf-8")).hexdigest()

STYLE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("telecaster", "T-Style"),
    ("stratocaster", "S-Style"),
    ("les paul", "Single cut"),
    ("sg", "Double cut"),
)

STYLE_CASCADE: Mapping[str, str] = {
    "T-Style": "Single cut",
    "S-Style": "Double cut",
}


@dataclass
class AiConversation:
    """Stateful conversation wrapper stored in ``st.session_state``."""

    conversation_id: Optional[str] = None
    rules_hash: Optional[str] = None
    system_messages: list[dict[str, str]] = field(default_factory=list)
    model: Optional[str] = None


def get_ai_conversation(model: str | None = None) -> AiConversation:
    """Return a global AI conversation, resetting when rules/model change."""

    stored = st.session_state.get("ai_conv")
    if isinstance(stored, AiConversation):
        conv = stored
    elif isinstance(stored, dict):
        conv = AiConversation(
            conversation_id=stored.get("conversation_id"),
            rules_hash=stored.get("rules_hash"),
            system_messages=list(stored.get("system_messages") or []),
            model=stored.get("model"),
        )
    else:
        conv = AiConversation()

    model_changed = bool(model) and conv.model not in (None, model)
    rules_changed = conv.rules_hash != AI_RULES_HASH
    if model_changed or rules_changed:
        conv = AiConversation(
            conversation_id=None,
            rules_hash=AI_RULES_HASH,
            system_messages=[{"role": "system", "content": AI_RULES_TEXT}],
            model=model,
        )
    else:
        conv.rules_hash = AI_RULES_HASH
        if model is not None:
            conv.model = model
        if not conv.system_messages:
            conv.system_messages = [{"role": "system", "content": AI_RULES_TEXT}]
        if conv.model is None:
            conv.model = model

    st.session_state["ai_conv"] = conv
    return conv


def derive_styles_from_texts(texts: Iterable[object] | None) -> list[str]:
    """Detect guitar styles from textual hints (model/series/product name)."""

    if not texts:
        return []

    styles: list[str] = []
    seen: Set[str] = set()
    for text in texts:
        if text is None:
            continue
        lowered = str(text).casefold()
        if not lowered.strip():
            continue
        for keyword, style in STYLE_KEYWORDS:
            if keyword == "sg":
                if re.search(r"\bsg\b", lowered):
                    if style not in seen:
                        seen.add(style)
                        styles.append(style)
                continue
            if keyword in lowered and style not in seen:
                seen.add(style)
                styles.append(style)
    return styles


def _ensure_list_of_strings(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if item is None or (isinstance(item, float) and pd.isna(item)):
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _custom_attributes_map(product: Mapping[str, object] | None) -> dict[str, object]:
    mapping: dict[str, object] = {}
    if not isinstance(product, Mapping):
        return mapping
    custom_attrs = product.get("custom_attributes")
    if not isinstance(custom_attrs, Iterable):
        return mapping
    for attr in custom_attrs:
        if not isinstance(attr, Mapping):
            continue
        code = attr.get("attribute_code")
        if not isinstance(code, str) or not code:
            continue
        mapping[code] = attr.get("value")
    return mapping


def _collect_product_texts(
    product: Mapping[str, object] | None,
    custom_attrs: Mapping[str, object],
) -> list[str]:
    texts: list[str] = []
    candidates: list[object] = []
    if isinstance(product, Mapping):
        candidates.extend(
            [
                product.get("name"),
                product.get("sku"),
                product.get("meta_title"),
            ]
        )
    candidates.extend(
        [
            custom_attrs.get("short_description"),
            custom_attrs.get("description"),
            custom_attrs.get("meta_title"),
            custom_attrs.get("meta_keyword"),
            custom_attrs.get("meta_description"),
        ]
    )
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        text = _strip_html(candidate)
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        texts.append(text)
    return texts


def _build_evidence_snippet(text: str, start: int, end: int, pad: int = 30) -> str:
    left = max(start - pad, 0)
    right = min(end + pad, len(text))
    snippet = text[left:right].strip()
    return snippet[:120]


def _detect_regex_hits(texts: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    hits: dict[str, list[dict[str, str]]] = {}
    for raw_text in texts:
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        for key, spec in REGEX_DETECTION_PATTERNS.items():
            pattern = spec.get("pattern")
            if isinstance(pattern, re.Pattern):
                patterns = [pattern]
            elif isinstance(pattern, (list, tuple, set)):
                patterns = [pat for pat in pattern if isinstance(pat, re.Pattern)]
            else:
                patterns = []
            if not patterns:
                continue
            for pattern_obj in patterns:
                for match in pattern_obj.finditer(raw_text):
                    value = match.group(0)
                    formatter = spec.get("formatter")
                    if callable(formatter):
                        try:
                            value = formatter(match)
                        except Exception:
                            value = match.group(0)
                    evidence = _build_evidence_snippet(raw_text, match.start(), match.end())
                    entry: dict[str, str] = {"value": str(value), "evidence": evidence}
                    attribute = spec.get("attribute")
                    if isinstance(attribute, str) and attribute:
                        entry["attribute"] = attribute
                    hits.setdefault(key, []).append(entry)
    for key, entries in hits.items():
        deduped: list[dict[str, str]] = []
        seen_values: Set[str] = set()
        for entry in entries:
            value = entry.get("value")
            if not isinstance(value, str):
                continue
            normalized = value.casefold()
            if normalized in seen_values:
                continue
            seen_values.add(normalized)
            deduped.append(entry)
        hits[key] = deduped
    return hits


def _match_brand_lexicon(
    brand_hint: Optional[str],
    texts: Iterable[str],
) -> tuple[dict[str, object], dict[str, object]]:
    if not brand_hint:
        return {}, {}
    brand_canonical: Optional[str] = None
    brand_payload: Mapping[str, object] | None = None
    for candidate, payload in BRAND_LEXICON.items():
        if not isinstance(candidate, str):
            continue
        if candidate.casefold() == str(brand_hint).casefold():
            brand_canonical = candidate
            if isinstance(payload, Mapping):
                brand_payload = payload
            break
    if brand_payload is None:
        return {}, {}
    normalized_texts = [str(text).casefold() for text in texts if isinstance(text, str)]
    series_hits: list[str] = []
    if isinstance(brand_payload.get("series"), Iterable):
        for series in brand_payload.get("series", []):
            if not isinstance(series, str) or not series.strip():
                continue
            lowered = series.casefold()
            if any(lowered in text for text in normalized_texts):
                series_hits.append(series)
    attribute_candidates: dict[str, list[str]] = {}
    model_hits: list[dict[str, object]] = []
    models_payload = brand_payload.get("models")
    if isinstance(models_payload, Mapping):
        for model_name, attrs in models_payload.items():
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            lowered = model_name.casefold()
            if not any(lowered in text for text in normalized_texts):
                continue
            attributes_map: dict[str, object] = {}
            if isinstance(attrs, Mapping):
                for attr_code, attr_value in attrs.items():
                    if attr_code == "pickups" and isinstance(attr_value, Mapping):
                        for pickup_code, pickup_value in attr_value.items():
                            if not isinstance(pickup_code, str):
                                continue
                            values_list = _ensure_list_of_strings(pickup_value)
                            if values_list:
                                attribute_candidates.setdefault(pickup_code, []).extend(
                                    values_list
                                )
                                attributes_map[pickup_code] = values_list
                    elif isinstance(attr_code, str):
                        values_list = _ensure_list_of_strings(attr_value)
                        if values_list:
                            attribute_candidates.setdefault(attr_code, []).extend(
                                values_list
                            )
                            attributes_map[attr_code] = values_list
            if attributes_map:
                model_hits.append(
                    {
                        "model": model_name,
                        "attributes": attributes_map,
                    }
                )
    for attr_code, values in list(attribute_candidates.items()):
        deduped: list[str] = []
        seen: Set[str] = set()
        for value in values:
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value)
        attribute_candidates[attr_code] = deduped
    if not attribute_candidates and not series_hits:
        return {}, {}
    hint_payload: dict[str, object] = {
        "brand": brand_canonical or brand_hint,
        "series": series_hits,
        "attributes": attribute_candidates,
    }
    hits_payload: dict[str, object] = {
        "brand": brand_canonical or brand_hint,
        "series": series_hits,
        "models": model_hits,
    }
    return hint_payload, hits_payload


def _match_series_lexicon(
    attribute_set_name: Optional[str],
    brand_hint: Optional[str],
    texts: Iterable[str],
) -> tuple[dict[str, object], dict[str, object]]:
    if not attribute_set_name:
        return {}, {}
    payload = SERIES_LEXICON.get(str(attribute_set_name))
    if not isinstance(payload, Mapping):
        return {}, {}
    normalized_texts = [str(text).casefold() for text in texts if isinstance(text, str)]
    brand_candidates = {str(brand_hint).casefold()} if brand_hint else set()
    matched_series: list[str] = []
    attribute_candidates: dict[str, list[str]] = {}
    hits: list[dict[str, object]] = []
    for brand_name, series_entries in payload.items():
        if not isinstance(brand_name, str):
            continue
        if brand_candidates and brand_name.casefold() not in brand_candidates:
            continue
        entries_iter = series_entries if isinstance(series_entries, list) else []
        for entry in entries_iter:
            if not isinstance(entry, Mapping):
                continue
            series_name = entry.get("series")
            keywords = entry.get("keywords")
            attrs = entry.get("attributes")
            if not isinstance(series_name, str) or not series_name.strip():
                continue
            if keywords:
                keyword_matches = False
                for keyword in keywords:
                    if not isinstance(keyword, str) or not keyword.strip():
                        continue
                    lowered = keyword.casefold()
                    if any(lowered in text for text in normalized_texts):
                        keyword_matches = True
                        break
                if not keyword_matches:
                    continue
            matched_series.append(series_name)
            if isinstance(attrs, Mapping):
                attributes_map: dict[str, list[str]] = {}
                for attr_code, attr_values in attrs.items():
                    values_list = _ensure_list_of_strings(attr_values)
                    if not values_list:
                        continue
                    attribute_candidates.setdefault(attr_code, []).extend(values_list)
                    attributes_map[attr_code] = values_list
                hits.append({"brand": brand_name, "series": series_name, "attributes": attributes_map})
            else:
                hits.append({"brand": brand_name, "series": series_name})
    if not matched_series and not attribute_candidates:
        return {}, {}
    for attr_code, values in list(attribute_candidates.items()):
        deduped: list[str] = []
        seen: Set[str] = set()
        for value in values:
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value)
        attribute_candidates[attr_code] = deduped
    hint_payload: dict[str, object] = {
        "series": matched_series,
        "attributes": attribute_candidates,
    }
    if brand_hint:
        hint_payload["brand"] = brand_hint
    hits_payload: dict[str, object] = {"matches": hits}
    return hint_payload, hits_payload


def _normalize_measurement_string(value: str) -> str:
    normalized = (
        value.replace("”", '\"')
        .replace("″", '\"')
        .replace("“", '"')
        .replace("’", "'")
        .replace("½", " 1/2")
    )
    normalized = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?:inch|inches)",
        lambda m: f"{m.group('num')}\"",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?P<int>\d+)\s*(?:1/2)\s*(?:\"|in|inch|inches)",
        lambda m: f"{float(m.group('int')) + 0.5}\"",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*\"",
        lambda m: f"{m.group('num')}\"",
        normalized,
    )
    normalized = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*mm",
        lambda m: f"{m.group('num')}mm",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _apply_material_alias(value: str) -> str:
    alias = MATERIAL_ALIAS_MAP.get(value.casefold())
    if alias:
        return alias
    return value


def _normalize_value_for_code(code: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    lowered_code = code.casefold()
    if lowered_code == "pickup_config":
        return _normalize_pickup_config(text)
    if lowered_code in {"fretboard_material", "neck_material", "top_material", "body_material"}:
        aliased = _apply_material_alias(text.title())
        return aliased
    if lowered_code not in MEASUREMENT_CODES:
        return text
    normalized = _normalize_measurement_string(text)
    mm_match = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*mm\b", normalized, re.IGNORECASE)
    if lowered_code in {"scale_mensur", "neck_radius"}:
        if mm_match:
            try:
                inches = float(mm_match.group("val")) / 25.4
                value_str = f"{inches:.2f}".rstrip("0").rstrip(".")
                return f"{value_str}\""
            except (TypeError, ValueError):
                return f"{mm_match.group('val')}mm"
        inch_match = re.search(
            r"(?P<val>\d+(?:\.\d+)?)\s*(?:\"|inch|inches|in)\b",
            normalized,
            re.IGNORECASE,
        )
        if inch_match:
            value_str = f"{float(inch_match.group('val')):.2f}".rstrip("0").rstrip(".")
            return f"{value_str}\""
        return normalized
    if lowered_code == "neck_nutwidth":
        if mm_match:
            try:
                mm_value = float(mm_match.group("val"))
            except (TypeError, ValueError):
                return f"{mm_match.group('val')}mm"
            formatted = f"{mm_value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}mm"
        inch_match = re.search(
            r"(?P<val>\d+(?:\.\d+)?)\s*(?:\"|inch|inches|in)\b",
            normalized,
            re.IGNORECASE,
        )
        if inch_match:
            try:
                mm_value = float(inch_match.group("val")) * 25.4
                formatted = f"{mm_value:.1f}".rstrip("0").rstrip(".")
                return f"{formatted}mm"
            except (TypeError, ValueError):
                return normalized
        return normalized
    return normalized


def _map_to_meta_option(
    session: requests.Session,
    api_base: str,
    code: str,
    value: str,
) -> tuple[str, Optional[str]]:
    meta = get_attribute_meta(session, api_base, code) or {}
    options = meta.get("options")
    if not isinstance(options, list) or not options:
        return value, None

    def _normalize(text: object) -> str:
        return re.sub(r"\s+", " ", str(text or "").casefold()).strip()

    normalized_value = _normalize(value)
    if not normalized_value:
        return value, None

    best_label: Optional[str] = None
    best_score: tuple[int, int] | None = None
    for option in options:
        if not isinstance(option, Mapping):
            continue
        label = option.get("label")
        opt_value = option.get("value")
        normalized_label = _normalize(label)
        normalized_opt_value = _normalize(opt_value)
        if normalized_value == normalized_label or normalized_value == normalized_opt_value:
            best_label = str(label or opt_value or value).strip()
            best_score = (0, len(normalized_label))
            break
        if normalized_label and normalized_label in normalized_value:
            score = (1, -len(normalized_label))
            if best_score is None or score < best_score:
                best_score = score
                best_label = str(label or value).strip()
        elif normalized_value and normalized_value in normalized_label:
            score = (2, len(normalized_value))
            if best_score is None or score < best_score:
                best_score = score
                best_label = str(label or value).strip()
        elif normalized_value and normalized_value in normalized_opt_value:
            score = (3, len(normalized_value))
            if best_score is None or score < best_score:
                best_score = score
                best_label = str(label or value).strip()
    if best_label:
        return best_label, best_label
    return value, None


def _postprocess_suggestions(
    suggestions_map: dict[str, dict[str, object]],
    session: requests.Session,
    api_base: str,
) -> list[dict[str, object]]:
    adjustments: list[dict[str, object]] = []
    for code, payload in suggestions_map.items():
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("value")
        if value in (None, ""):
            continue
        updated_value = value
        change_reason: Optional[str] = None
        if isinstance(value, str):
            normalized = _normalize_value_for_code(code, value)
            if code in MEASUREMENT_CODES:
                normalized = _normalize_measurement_string(normalized)
            mapped_value, mapped_reason = _map_to_meta_option(session, api_base, code, normalized)
            updated_value = mapped_value
            change_reason = mapped_reason
            if mapped_value != normalized:
                change_reason = mapped_reason or "meta_mapping"
            elif normalized != value:
                change_reason = "normalized"
        elif isinstance(value, (list, tuple, set)):
            normalized_items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    item_text = str(item)
                else:
                    item_text = item
                item_normalized = _normalize_value_for_code(code, item_text)
                if code in MEASUREMENT_CODES:
                    item_normalized = _normalize_measurement_string(item_normalized)
                mapped_value, mapped_reason = _map_to_meta_option(
                    session, api_base, code, item_normalized
                )
                normalized_items.append(mapped_value)
                if mapped_reason:
                    change_reason = mapped_reason
                elif item_normalized != item_text:
                    change_reason = "normalized"
            deduped_items: list[str] = []
            seen: Set[str] = set()
            for item in normalized_items:
                normalized_key = item.casefold()
                if normalized_key in seen:
                    continue
                seen.add(normalized_key)
                deduped_items.append(item)
            updated_value = deduped_items
        if updated_value != value:
            payload["value"] = updated_value
            adjustments.append(
                {
                    "code": code,
                    "from": value,
                    "to": updated_value,
                    "reason": change_reason or "normalized",
                }
            )
    return adjustments


def _merge_unique(*iterables: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: Set[str] = set()
    for iterable in iterables:
        for item in iterable or []:
            if not isinstance(item, str):
                candidate = str(item)
            else:
                candidate = item
            normalized = candidate.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def normalize_category_label(label: str | None, meta: Mapping[str, object] | None) -> str | None:
    if not label:
        return None
    s = str(label).strip()
    if not s:
        return None

    meta = meta or {}
    canon = meta.get("labels_to_values") if isinstance(meta, Mapping) else None
    if not isinstance(canon, Mapping):
        canon = {}

    if s in canon:
        return s

    low = s.casefold()
    rev_low = st.session_state.setdefault(
        "cat_lowmap", {str(k).casefold(): str(k) for k in canon.keys()}
    )
    for key in canon.keys():
        key_cf = str(key).casefold()
        if key_cf not in rev_low:
            rev_low[key_cf] = str(key)
    if low in rev_low:
        return rev_low[low]
    if low.endswith("s") and low[:-1] in rev_low:
        return rev_low[low[:-1]]
    plural = f"{low}s"
    if plural in rev_low:
        return rev_low[plural]
    return None


def _normalize_category_candidate(
    candidate: object,
    categories_meta: Mapping[str, object] | None,
) -> str | None:
    """Normalize a category candidate coming from text or numeric identifiers."""

    if candidate is None:
        return None

    categories_meta = categories_meta if isinstance(categories_meta, Mapping) else {}
    values_to_labels = categories_meta.get("values_to_labels")
    if not isinstance(values_to_labels, Mapping):
        values_to_labels = {}

    text = str(candidate).strip()
    if not text:
        return None
    if text.isdigit():
        label = values_to_labels.get(text)
        if isinstance(label, str) and label.strip():
            normalized = normalize_category_label(label, categories_meta)
            return normalized or label.strip()
        return None
    return normalize_category_label(text, categories_meta)


def _guess_brand_from_name(
    product_name: object,
    categories_meta: Mapping[str, object] | None,
) -> str | None:
    """Heuristically detect a brand hint from the product name using known brands."""

    if not isinstance(product_name, str):
        return None

    name_text = product_name.strip()
    if not name_text:
        return None

    categories_meta = categories_meta if isinstance(categories_meta, Mapping) else {}
    labels_to_values = categories_meta.get("labels_to_values")
    if not isinstance(labels_to_values, Mapping) or not labels_to_values:
        return None

    seen: Set[str] = set()
    sorted_labels = sorted(
        (
            str(label).strip()
            for label in labels_to_values.keys()
            if str(label or "").strip()
        ),
        key=lambda value: (-len(value), value.casefold()),
    )

    for label in sorted_labels:
        if label in seen:
            continue
        seen.add(label)
        pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)
        if pattern.search(name_text):
            return label

    return None


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return all(_is_blank(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return all(_is_blank(item) for item in value)
    return False


def _truncate_preview(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    snippet = str(text).strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit] + "…"


def _sanitize_hints(hints: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(hints, Mapping):
        return {}
    sanitized: dict[str, object] = {}
    for key, value in hints.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (list, tuple, set)):
            normalized = _ensure_list_of_strings(list(value))
            if normalized:
                sanitized[key] = normalized
            continue
        if _is_blank(value):
            continue
        if isinstance(value, (str, bool, int, float)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def _normalize_boolean(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1", "y"}:
            return True
        if lowered in {"false", "no", "0", "n"}:
            return False
    return None


def _extract_response_text(response: object) -> str:
    if response is None:
        return ""

    if hasattr(response, "output_text"):
        output_text = getattr(response, "output_text")
        if output_text:
            return str(output_text)

    text_parts: list[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))
                elif isinstance(part, Mapping):
                    raw = part.get("text")
                    if raw:
                        text_parts.append(str(raw))
    if text_parts:
        return "\n".join(text_parts).strip()

    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(response, attr):
            try:
                data = getattr(response, attr)()
            except TypeError:
                continue
            if isinstance(data, Mapping):
                maybe_text = data.get("output_text")
                if maybe_text:
                    return str(maybe_text)
                output_data = data.get("output")
                if isinstance(output_data, list):
                    collected: list[str] = []
                    for entry in output_data:
                        content = entry.get("content") if isinstance(entry, Mapping) else None
                        if isinstance(content, list):
                            for part in content:
                                text = part.get("text") if isinstance(part, Mapping) else None
                                if text:
                                    collected.append(str(text))
                    if collected:
                        return "\n".join(collected).strip()
    if isinstance(response, Mapping):
        maybe_text = response.get("output_text")
        if maybe_text:
            return str(maybe_text)
    return ""


class HTMLResponseError(RuntimeError):
    """Raised when Magento returns an HTML page instead of JSON."""

    def __init__(self, base_url: str):
        super().__init__(
            "❌ HTML вместо JSON — проверь base URL или WAF/Cloudflare; "
            f"попробован путь: {base_url}"
        )
        self.base_url = base_url


def _magento_headers(session: requests.Session) -> dict[str, str]:
    """Build default Magento headers using the session auth token if available."""

    return build_magento_headers(session=session)


def _looks_like_json(text: str) -> bool:
    try:
        json.loads(text)
    except Exception:
        return False
    return True


def probe_api_base(session: requests.Session, origin: str) -> str:
    """Discover a working Magento REST prefix for ``origin``."""

    origin = origin.rstrip("/")
    non_json_paths: list[str] = []
    for path in MAGENTO_REST_PATH_CANDIDATES:
        base = f"{origin}{path}"
        try:
            resp = session.get(
                f"{base}/store/storeViews",
                headers=_magento_headers(session),
                timeout=10,
            )
        except requests.RequestException:
            continue
        content_type = resp.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            non_json_paths.append(base)
            continue
        if resp.status_code in {200, 401, 403}:
            return base
    if non_json_paths:
        raise HTMLResponseError(non_json_paths[-1])
    raise RuntimeError("❌ Не удалось определить корректный REST-префикс.")


def _build_url(api_base: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{api_base.rstrip('/')}{path}"


def get_product_by_sku(session: requests.Session, api_base: str, sku: str) -> dict:
    url = _build_url(api_base, f"/products/{quote(sku, safe='')}")
    resp = session.get(url, headers=_magento_headers(session), timeout=20)
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError(f"❌ HTML вместо JSON — проверь REST базу: {api_base}")
    if resp.status_code == 404:
        raise RuntimeError(f"❌ SKU={sku} не найден.")
    if not resp.ok:
        raise RuntimeError(f"Magento API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def get_attribute_sets_map(session: requests.Session, api_base: str) -> Dict[int, str]:
    urls = (
        _build_url(
            api_base,
            "/eav/attribute-sets/list?searchCriteria[currentPage]=1&searchCriteria[pageSize]=200",
        ),
        _build_url(
            api_base,
            "/products/attribute-sets/sets/list?searchCriteria[currentPage]=1&searchCriteria[pageSize]=200",
        ),
    )
    for url in urls:
        try:
            resp = session.get(url, headers=_magento_headers(session), timeout=20)
        except requests.RequestException:
            continue
        content_type = resp.headers.get("Content-Type", "").lower()
        if resp.ok and ("json" in content_type or _looks_like_json(resp.text)):
            data = resp.json()
            items = data.get("items", data) or []
            return {
                (item.get("attribute_set_id") or item.get("id")): item.get("attribute_set_name")
                or item.get("name")
                for item in items
            }
    raise RuntimeError("❌ Не удалось получить список attribute sets.")


_meta_cache: MutableMapping[str, dict] = {}
_CATEGORIES_CACHE: Optional[List[str]] = None


def get_attribute_meta(session: requests.Session, api_base: str, code: str) -> dict:
    if code in _meta_cache:
        return _meta_cache[code]
    url = _build_url(api_base, f"/products/attributes/{quote(code, safe='')}")
    try:
        resp = session.get(url, headers=_magento_headers(session), timeout=15)
    except requests.RequestException:
        _meta_cache[code] = {}
        return _meta_cache[code]
    content_type = resp.headers.get("Content-Type", "").lower()
    if resp.ok and ("json" in content_type or _looks_like_json(resp.text)):
        _meta_cache[code] = resp.json()
    else:
        _meta_cache[code] = {}
    return _meta_cache[code]


def list_categories(session: requests.Session, api_base: str) -> List[Dict[str, str]]:
    """Return Magento categories as a flat, alphabetically sorted list."""

    url = _build_url(api_base, "/categories")
    try:
        resp = session.get(url, headers=_magento_headers(session), timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError("❌ Не удалось загрузить категории.") from exc

    content_type = resp.headers.get("Content-Type", "").lower()
    if not resp.ok or ("json" not in content_type and not _looks_like_json(resp.text)):
        raise RuntimeError(
            f"Magento API error {resp.status_code}: {resp.text[:300]}"
        )

    payload = resp.json() or {}
    items: List[Dict[str, str]] = []

    def _walk(node: Optional[Dict[str, object]]):
        if not isinstance(node, dict):
            return
        cat_id = node.get("id")
        name = node.get("name")
        if cat_id not in (None, "") and isinstance(name, str) and name.strip():
            items.append({"id": str(cat_id), "name": name.strip()})
        for child in node.get("children_data", []) or []:
            _walk(child)

    _walk(payload)

    def _sort_key(item: Dict[str, str]):
        return (item.get("name", "").lower(), item.get("id", ""))

    return sorted(items, key=_sort_key)


def option_label_for(session: requests.Session, api_base: str, code: str, raw_value) -> Tuple[Optional[str], Optional[str]]:
    meta = get_attribute_meta(session, api_base, code) or {}
    frontend_input = (meta.get("frontend_input") or "").lower()
    options = meta.get("options") or []
    if frontend_input in {"select", "multiselect"} and raw_value not in (None, "", []):
        values = [str(val).strip() for val in str(raw_value).split(",") if str(val).strip()]
        id_to_label = {
            str(opt.get("value")): opt.get("label") for opt in options if "value" in opt
        }
        labels = [id_to_label.get(val, val) for val in values]
        return ", ".join([label for label in labels if label]), frontend_input
    return None, frontend_input or None


def compute_allowed_attrs(
    attr_set_id: Optional[int],
    sets: Mapping[object, Set[str]],
    id_to_name: Mapping[int, str],
    always: Iterable[str],
) -> Set[str]:
    allowed = set(always)
    if attr_set_id in sets:
        allowed |= set(sets[attr_set_id])
    name = id_to_name.get(attr_set_id) if attr_set_id is not None else None
    if isinstance(name, str) and name in sets:
        allowed |= set(sets[name])
    if "Default" in sets:
        allowed |= set(sets["Default"])
    return allowed


def collect_attributes_table(
    product: Mapping[str, object],
    allowed: Sequence[str],
    session: requests.Session,
    api_base: str,
) -> pd.DataFrame:
    data: Dict[str, Dict[str, Optional[str]]] = {
        code: {"raw": None, "label": None, "type": None} for code in allowed
    }
    for attr in product.get("custom_attributes", []) or []:
        code = attr.get("attribute_code")
        value = attr.get("value")
        if code in data:
            meta = get_attribute_meta(session, api_base, code)
            label, input_type = option_label_for(session, api_base, code, value)
            data[code].update(
                {
                    "raw": value,
                    "label": label,
                    "type": input_type or meta.get("frontend_input"),
                }
            )
    for field in ("sku", "name", "price", "weight"):
        if field in data:
            value = product.get(field)
            meta = get_attribute_meta(session, api_base, field)
            label, input_type = option_label_for(session, api_base, field, value)
            data[field].update(
                {
                    "raw": value,
                    "label": label,
                    "type": input_type or meta.get("frontend_input"),
                }
            )
    df = pd.DataFrame(data).T.rename(columns={"raw": "raw_value"})
    return df


def _strip_html(value: Optional[object]) -> str:
    try:
        return re.sub(r"<[^>]+>", "", str(value or "")).strip()
    except Exception:
        return ""


def _openai_complete(
    conv: AiConversation | None,
    user_msg: str,
    api_key: str,
    model: str = "gpt-5-mini",
    timeout: int = 60,
) -> tuple[dict, str]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured.")

    if isinstance(conv, AiConversation) and conv.system_messages:
        system_messages = list(conv.system_messages)
    else:
        system_messages = [{"role": "system", "content": AI_RULES_TEXT}]

    messages_payload: list[dict[str, str]] = []
    for message in system_messages:
        role = str(message.get("role") or "system")
        content = message.get("content")
        if content is None:
            continue
        messages_payload.append({"role": role, "content": str(content)})
    messages_payload.append({"role": "user", "content": user_msg})

    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - client init safety
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    kwargs = {
        "model": model,
        "input": messages_payload,
        "temperature": 0,
        "timeout": timeout,
    }
    if isinstance(conv, AiConversation) and conv.conversation_id:
        kwargs["conversation"] = {"id": conv.conversation_id}

    try:
        response = client.responses.create(**kwargs)
    except TypeError:
        # Fallback for environments without the Responses API.
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages_payload,
                temperature=0,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - network error handling
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        try:
            content = chat_response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI response missing message content") from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI response was not valid JSON") from exc

        if isinstance(conv, AiConversation):
            conv.conversation_id = None
            conv.rules_hash = AI_RULES_HASH
            conv.model = model
            conv.system_messages = system_messages

        return parsed, content
    except Exception as exc:  # pragma: no cover - network error handling
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    raw_content = _extract_response_text(response)
    if not raw_content:
        raise RuntimeError("OpenAI response missing message content")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI response was not valid JSON") from exc

    if isinstance(conv, AiConversation):
        conversation_info = getattr(response, "conversation", None)
        new_id = getattr(conversation_info, "id", None)
        if new_id is None and isinstance(conversation_info, Mapping):
            new_id = conversation_info.get("id")
        if new_id:
            conv.conversation_id = new_id
        conv.rules_hash = AI_RULES_HASH
        conv.model = model
        conv.system_messages = system_messages

    return parsed, raw_content


def enrich_ai_suggestions(
    ai_df: pd.DataFrame,
    hints: Mapping[str, object] | None,
    categories_meta: Mapping[str, object] | None,
    attribute_set_name: str | None,
    set_id: object | None,
    product: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Post-process AI suggestions with deterministic hints and cascades."""

    if not isinstance(ai_df, pd.DataFrame):
        ai_df = pd.DataFrame()

    hints = hints or {}
    normalized_hints = _sanitize_hints(hints)
    categories_meta = categories_meta if isinstance(categories_meta, Mapping) else {}

    records = ai_df.to_dict(orient="records")
    order: list[str] = []
    by_code: dict[str, dict[str, object]] = {}

    for record in records:
        code = record.get("code")
        if not code:
            continue
        code_key = str(code)
        order.append(code_key)
        cleaned: dict[str, object] = {"code": code_key}
        for key, value in record.items():
            if key == "code":
                continue
            if isinstance(value, float) and pd.isna(value):
                value = None
            cleaned[key] = value
        by_code[code_key] = cleaned

    set_name_cf = str(attribute_set_name or "").casefold()
    bass_hint_flag = bool(normalized_hints.get("bass_hint"))
    product_name_text = ""
    if isinstance(product, Mapping):
        product_name_text = str(product.get("name", ""))
    is_bass_item = "bass" in set_name_cf or bass_hint_flag
    auto_no_strings_value: str | None = None
    if is_bass_item:
        entry = by_code.get("no_strings")
        existing_value = entry.get("value") if isinstance(entry, Mapping) else None
        existing_value_str = ""
        if isinstance(existing_value, str):
            existing_value_str = existing_value.strip()
        elif isinstance(existing_value, (int, float)):
            existing_value_str = str(existing_value)
        elif isinstance(existing_value, (list, tuple, set)):
            cleaned_items = [
                str(item).strip()
                for item in existing_value
                if not _is_blank(item)
            ]
            if cleaned_items:
                existing_value_str = cleaned_items[0]
        should_override = _is_blank(existing_value) or existing_value_str == "6"
        if should_override:
            tokens = [token.strip() for token in product_name_text.split()] if product_name_text else []
            has_five = "5" in product_name_text
            has_v_token = any(token.upper() == "V" for token in tokens)
            suggested_value = "5" if has_five or has_v_token else "4"
            entry = by_code.setdefault("no_strings", {"code": "no_strings"})
            entry["value"] = suggested_value
            entry.pop("evidence", None)
            entry["reason"] = "enriched_bass_default"
            if "no_strings" not in order:
                order.append("no_strings")
            auto_no_strings_value = suggested_value

    # Categories enrichment with brand/set hints
    categories_meta = categories_meta if isinstance(categories_meta, Mapping) else {}
    labels_to_values_map = categories_meta.get("labels_to_values")
    if not isinstance(labels_to_values_map, Mapping):
        labels_to_values_map = {}

    existing_entry = by_code.get("categories")
    existing_categories_raw = _ensure_list_of_strings(
        (existing_entry or {}).get("value")
    )
    existing_categories_canon: list[str] = []
    existing_category_ids: list[str] = []
    unmatched_existing: list[str] = []
    for item in existing_categories_raw:
        normalized_label = _normalize_category_candidate(item, categories_meta)
        if normalized_label:
            if normalized_label not in existing_categories_canon:
                existing_categories_canon.append(normalized_label)
            if labels_to_values_map:
                resolved_value = labels_to_values_map.get(normalized_label)
                if resolved_value is not None:
                    resolved_str = str(resolved_value).strip()
                    if resolved_str and resolved_str not in existing_category_ids:
                        existing_category_ids.append(resolved_str)
        else:
            text = str(item).strip()
            if text and text not in unmatched_existing:
                unmatched_existing.append(text)

    brand_hint_values = _ensure_list_of_strings(normalized_hints.get("brand_hint"))
    set_hint_values = _ensure_list_of_strings(normalized_hints.get("set_hint"))
    brand_attr_values = _ensure_list_of_strings((by_code.get("brand") or {}).get("value"))
    attribute_set_candidates: list[str] = []
    if attribute_set_name and not _is_blank(attribute_set_name):
        attribute_set_candidates.append(str(attribute_set_name).strip())

    category_candidates = _merge_unique(
        brand_hint_values,
        brand_attr_values,
        set_hint_values,
        attribute_set_candidates,
    )

    normalized_categories: list[str] = list(existing_categories_canon)
    resolved_category_ids: list[str] = list(existing_category_ids)
    ignored_candidates: list[str] = []
    canonical_seen: Set[str] = set(existing_categories_canon)
    resolved_seen: Set[str] = set(existing_category_ids)

    for candidate in category_candidates:
        normalized_label = _normalize_category_candidate(candidate, categories_meta)
        if normalized_label:
            if labels_to_values_map:
                resolved_value = labels_to_values_map.get(normalized_label)
                if resolved_value is None:
                    candidate_text = str(candidate).strip()
                    if candidate_text and candidate_text not in ignored_candidates:
                        ignored_candidates.append(candidate_text)
                    continue
                resolved_str = str(resolved_value).strip()
                if resolved_str and resolved_str not in resolved_seen:
                    resolved_seen.add(resolved_str)
                    resolved_category_ids.append(resolved_str)
            if normalized_label not in canonical_seen:
                canonical_seen.add(normalized_label)
                normalized_categories.append(normalized_label)
        else:
            candidate_text = str(candidate).strip()
            if candidate_text and candidate_text not in ignored_candidates:
                ignored_candidates.append(candidate_text)

    normalized_categories_added = [
        label for label in normalized_categories if label not in existing_categories_canon
    ]

    if normalized_categories:
        category_entry = by_code.setdefault(
            "categories",
            {"code": "categories", "reason": "enriched_from_hints"},
        )
        category_entry["value"] = normalized_categories
        if resolved_category_ids:
            category_entry["resolved_category_ids"] = list(resolved_category_ids)
        if _is_blank(category_entry.get("reason")):
            category_entry["reason"] = "enriched_from_hints"
        if normalized_categories_added:
            category_entry["normalized_categories_added"] = list(normalized_categories_added)
        ignored_tokens = _merge_unique(
            _ensure_list_of_strings(category_entry.get("ignored_by_ai_filter")),
            unmatched_existing,
            ignored_candidates,
        )
        if ignored_tokens:
            category_entry["ignored_by_ai_filter"] = ignored_tokens
            category_entry["unmatched_labels"] = list(ignored_tokens)
        elif unmatched_existing:
            category_entry["unmatched_labels"] = list(unmatched_existing)
        if "categories" not in order:
            order.append("categories")
    else:
        combined_ignored = _merge_unique(unmatched_existing, ignored_candidates)
        if combined_ignored:
            category_entry = by_code.setdefault(
                "categories",
                {"code": "categories", "reason": "enriched_from_hints"},
            )
            category_entry["ignored_by_ai_filter"] = combined_ignored
            category_entry["unmatched_labels"] = list(combined_ignored)
            if "categories" not in order:
                order.append("categories")

    # Style enrichment using hints and detected model/series
    existing_styles = _ensure_list_of_strings(
        (by_code.get("guitarstylemultiplechoice") or {}).get("value")
    )
    style_hint_values = _ensure_list_of_strings(normalized_hints.get("style_hint"))
    model_series_candidates = _ensure_list_of_strings(
        normalized_hints.get("model_series_candidates")
    )
    candidate_texts: list[str] = list(model_series_candidates)
    for attr_code in ("model", "series"):
        entry = by_code.get(attr_code)
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("value")
        if isinstance(value, (list, tuple, set)):
            candidate_texts.extend([str(item) for item in value if not _is_blank(item)])
        elif not _is_blank(value):
            candidate_texts.append(str(value))

    # set_name_cf already computed above
    try:
        set_id_int = int(set_id) if set_id not in (None, "") else None
    except (TypeError, ValueError):
        set_id_int = None
    is_electric = set_name_cf in {"electric guitar", "electric guitars"} or (
        set_id_int in ELECTRIC_SET_IDS if set_id_int is not None else False
    )

    derived_styles = (
        derive_styles_from_texts(candidate_texts) if is_electric else []
    )
    merged_styles = _merge_unique(existing_styles, style_hint_values, derived_styles)
    if merged_styles and is_electric:
        cascade_styles = [
            STYLE_CASCADE.get(style)
            for style in merged_styles
            if STYLE_CASCADE.get(style)
        ]
        merged_styles = _merge_unique(merged_styles, cascade_styles)
    if merged_styles and not is_bass_item:
        style_entry = by_code.setdefault(
            "guitarstylemultiplechoice",
            {"code": "guitarstylemultiplechoice", "reason": "enriched_from_hints"},
        )
        style_entry["value"] = merged_styles
        if _is_blank(style_entry.get("reason")):
            style_entry["reason"] = "enriched_from_hints"
        if "guitarstylemultiplechoice" not in order:
            order.append("guitarstylemultiplechoice")

    # Normalize multi-select payloads to lists of strings
    for code_key, payload in by_code.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        if isinstance(value, (list, tuple, set)):
            payload["value"] = _ensure_list_of_strings(value)

    enriched_records: list[dict[str, object]] = []
    seen_codes: Set[str] = set()
    for code_key in order:
        if code_key in seen_codes:
            continue
        payload = by_code.get(code_key)
        if isinstance(payload, dict):
            enriched_records.append(payload)
            seen_codes.add(code_key)
    for code_key, payload in by_code.items():
        if code_key in seen_codes or not isinstance(payload, dict):
            continue
        enriched_records.append(payload)

    result_df = pd.DataFrame(enriched_records)
    original_attrs = dict(getattr(ai_df, "attrs", {}))
    meta_payload: dict[str, object] = {}
    if isinstance(original_attrs.get("meta"), Mapping):
        meta_payload = dict(original_attrs.get("meta"))
    brand_hint_meta: str | None = None
    for hint_value in brand_hint_values:
        normalized = _normalize_category_candidate(hint_value, categories_meta)
        if normalized:
            brand_hint_meta = normalized
            break
    if brand_hint_meta is None and brand_hint_values:
        brand_hint_meta = brand_hint_values[0]
    meta_payload["brand_hint"] = brand_hint_meta
    meta_payload["bass_hint"] = bool(is_bass_item)
    if auto_no_strings_value is not None:
        meta_payload["no_strings_auto"] = auto_no_strings_value
    meta_payload["normalized_categories_added"] = normalized_categories_added
    original_attrs["meta"] = meta_payload
    result_df.attrs = original_attrs
    return result_df


def infer_missing(
    product: Mapping[str, object],
    df_full: pd.DataFrame,
    session: requests.Session,
    api_base: str,
    allowed: Sequence[str],
    api_key: str,
    *,
    conv: AiConversation | None = None,
    hints: Mapping[str, object] | None = None,
    model: str = "gpt-5-mini",
) -> pd.DataFrame:
    allowed_set: Set[str] = {str(item) for item in allowed if isinstance(item, str)}
    missing: List[dict] = []
    known: Dict[str, str] = {}
    current_attributes: Dict[str, object] = {}
    bool_codes: Set[str] = set()
    missing_codes: Set[str] = set()
    force_codes: Set[str] = {
        "categories",
        "guitarstylemultiplechoice",
        "no_strings",
    } | set(
        SPEC_FORCE_CODES
    )
    postprocess_adjustments: list[dict[str, object]] = []
    global _CATEGORIES_CACHE

    for raw_code, row in df_full.iterrows():
        code = str(raw_code)
        row_type = str(row.get("type") or "").lower()
        label = row.get("label")
        raw_value = row.get("raw_value")
        value_for_payload = label if not _is_blank(label) else raw_value

        if code == "categories":
            normalized_value = _ensure_list_of_strings(value_for_payload)
            current_attributes[code] = normalized_value
            if code in allowed_set and _is_blank(normalized_value):
                if _CATEGORIES_CACHE is None:
                    cats = list_categories(session, api_base)
                    if cats:
                        _CATEGORIES_CACHE = [
                            cat.get("name")
                            for cat in cats
                            if isinstance(cat, dict)
                            and isinstance(cat.get("name"), str)
                            and cat.get("name").strip()
                        ]
                    else:
                        _CATEGORIES_CACHE = []
                missing.append(
                    {
                        "code": "categories",
                        "type": "multiselect",
                        "options": list(_CATEGORIES_CACHE or []),
                    }
                )
                missing_codes.add(code)
            continue

        if code in allowed_set and row_type == "boolean":
            bool_codes.add(code)

        if row_type == "multiselect":
            normalized_value = _ensure_list_of_strings(value_for_payload)
        elif row_type == "boolean":
            normalized_value = _normalize_boolean(value_for_payload)
        else:
            normalized_value = (
                str(value_for_payload).strip()
                if isinstance(value_for_payload, str)
                else value_for_payload
            )
        current_attributes[code] = normalized_value

        current_value = label if not _is_blank(label) else raw_value
        if code in allowed_set and _is_blank(current_value):
            meta = get_attribute_meta(session, api_base, code)
            missing.append(
                {
                    "code": code,
                    "type": meta.get("frontend_input"),
                    "options": [
                        opt.get("label")
                        for opt in meta.get("options", [])
                        if opt.get("label") not in (None, "")
                    ],
                }
            )
            missing_codes.add(code)
        elif code in allowed_set and not _is_blank(current_value):
            known[code] = str(current_value)

    ai_codes = sorted(
        {
            code
            for code in force_codes
            if code in allowed_set
        }
        | {code for code in bool_codes if code in allowed_set}
        | {code for code in missing_codes if code in allowed_set}
    )

    custom_attrs_map = _custom_attributes_map(product)
    short_desc = _strip_html(custom_attrs_map.get("short_description"))
    long_desc = _strip_html(custom_attrs_map.get("description"))
    meta_title = _strip_html(custom_attrs_map.get("meta_title"))
    if not meta_title:
        meta_title = _strip_html((product or {}).get("meta_title"))
    vendor_sku = _strip_html(custom_attrs_map.get("vendor_sku"))
    mpn_value = _strip_html(custom_attrs_map.get("mpn"))
    ean_value = _strip_html(custom_attrs_map.get("ean"))
    upc_value = _strip_html(custom_attrs_map.get("upc"))
    product_texts = _collect_product_texts(product, custom_attrs_map)
    regex_hits = _detect_regex_hits(product_texts)
    if not isinstance(regex_hits, dict):
        regex_hits = {}
    full_text_parts: list[str] = [
        str(text)
        for text in product_texts
        if isinstance(text, str) and text.strip()
    ]
    if full_text_parts:
        full_text = " \n".join(full_text_parts)
        m = re.search(r"(\d)\s*(?:string|strings)", full_text, re.IGNORECASE)
        if m:
            regex_hits["no_strings"] = m.group(1)
    attributes_raw_payload = product.get("custom_attributes")
    if isinstance(attributes_raw_payload, list):
        attributes_raw = attributes_raw_payload
    elif isinstance(attributes_raw_payload, Iterable):
        attributes_raw = list(attributes_raw_payload)
    else:
        attributes_raw = []

    if isinstance(hints, Mapping):
        hints_payload = dict(hints)
    else:
        hints_payload = {}

    attribute_set_hint = (
        hints_payload.get("attribute_set_name")
        or hints_payload.get("set_hint")
        or ""
    )
    product_name_text = ""
    if isinstance(product, Mapping):
        product_name_text = str(product.get("name", ""))
    is_bass_product = "bass" in str(attribute_set_hint).lower() or "bass" in product_name_text.lower()
    if is_bass_product:
        hints_payload.setdefault("bass_hint", True)

    sanitized_hints = _sanitize_hints(hints_payload)
    attribute_set_name = sanitized_hints.get("attribute_set_name") or sanitized_hints.get(
        "set_hint"
    )

    brand_current = current_attributes.get("brand")
    categories_meta_for_hints = st.session_state.get("categories_meta")
    if not isinstance(categories_meta_for_hints, Mapping):
        categories_meta_for_hints = {}
    if _is_blank(sanitized_hints.get("brand_hint")) and _is_blank(brand_current):
        guessed_brand = _guess_brand_from_name(
            (product or {}).get("name") if isinstance(product, Mapping) else None,
            categories_meta_for_hints,
        )
        if guessed_brand:
            sanitized_hints["brand_hint"] = guessed_brand

    if regex_hits:
        sanitized_hints["regex_hits"] = regex_hits

    brand_hint_values = _ensure_list_of_strings(sanitized_hints.get("brand_hint"))
    brand_for_lexicon: Optional[str] = brand_hint_values[0] if brand_hint_values else None
    if not brand_for_lexicon and isinstance(brand_current, str) and brand_current.strip():
        brand_for_lexicon = str(brand_current).strip()
    brand_lexicon_hint, brand_lexicon_hits = _match_brand_lexicon(
        brand_for_lexicon,
        product_texts,
    )
    if brand_lexicon_hint:
        sanitized_hints["brand_lexicon"] = brand_lexicon_hint

    brand_for_series: Optional[str] = None
    if isinstance(brand_lexicon_hint, Mapping):
        brand_for_series = str(brand_lexicon_hint.get("brand") or "").strip() or None
    if not brand_for_series:
        brand_for_series = brand_for_lexicon
    series_lexicon_hint, series_lexicon_hits = _match_series_lexicon(
        attribute_set_name,
        brand_for_series,
        product_texts,
    )
    if series_lexicon_hint:
        sanitized_hints["series_lexicon"] = series_lexicon_hint

    description_payload = long_desc or short_desc

    user_payload = {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "short_description": short_desc,
        "long_description": long_desc,
        "description": description_payload,
        "meta_title": meta_title,
        "vendor_sku": vendor_sku,
        "mpn": mpn_value,
        "ean": ean_value,
        "upc": upc_value,
        "attributes_raw": attributes_raw,
        "attributes_raw_map": custom_attrs_map,
        "attribute_set_name": attribute_set_name,
        "current_attributes": current_attributes,
        "ai_codes": ai_codes,
        "hints": sanitized_hints,
        "known_values": known,
        "missing": missing,
        "regex_hits": regex_hits,
        "brand_lexicon": brand_lexicon_hint,
        "series_lexicon": series_lexicon_hint,
    }

    payload_json = json.dumps(user_payload, ensure_ascii=False)
    completion, raw_content = _openai_complete(conv, payload_json, api_key, model=model)
    attributes = completion.get("attributes") if isinstance(completion, Mapping) else []
    ai_df = pd.DataFrame(attributes or [])

    def _df_to_code_map(df: pd.DataFrame) -> dict[str, dict[str, object]]:
        mapping: dict[str, dict[str, object]] = {}
        if not isinstance(df, pd.DataFrame):
            return mapping
        for _, row in df.iterrows():
            code_value = row.get("code")
            if not code_value:
                continue
            code_key = str(code_value)
            payload: dict[str, object] = {"code": code_key}
            for key, value in row.items():
                if key == "code":
                    continue
                if isinstance(value, float) and pd.isna(value):
                    value = None
                payload[key] = value
            mapping[code_key] = payload
        return mapping

    suggestions_map = _df_to_code_map(ai_df)
    if suggestions_map:
        postprocess_adjustments.extend(
            _postprocess_suggestions(suggestions_map, session, api_base)
        )
    order: list[str] = list(ai_codes)
    for code in suggestions_map.keys():
        if code not in order:
            order.append(code)

    raw_previews: list[str] = [_truncate_preview(raw_content)]
    remaining_all = [
        code for code in ai_codes if _is_blank(suggestions_map.get(code, {}).get("value"))
    ]
    spec_missing = [code for code in SPEC_FORCE_CODES if code in remaining_all]
    remaining_before_retry = spec_missing or remaining_all
    retried = False
    retry_questions: dict[str, str] = {}
    retry_context: dict[str, object] | None = None

    if remaining_before_retry:
        context_already_found = {
            code: entry.get("value")
            for code, entry in suggestions_map.items()
            if not _is_blank(entry.get("value"))
        }
        brand_hint_values = _ensure_list_of_strings(sanitized_hints.get("brand_hint"))
        if brand_hint_values and _is_blank(context_already_found.get("brand")):
            context_already_found["brand"] = brand_hint_values[0]
        if regex_hits:
            for regex_key, entries in regex_hits.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    attr = entry.get("attribute")
                    val = entry.get("value")
                    if (
                        isinstance(attr, str)
                        and attr not in context_already_found
                        and not _is_blank(val)
                    ):
                        context_already_found[attr] = val
            context_already_found["regex_hits"] = regex_hits
        if brand_lexicon_hint:
            context_already_found["brand_lexicon"] = brand_lexicon_hint
        if series_lexicon_hint:
            context_already_found["series_lexicon"] = series_lexicon_hint
        retry_context = context_already_found
        retry_questions = {
            code: SPEC_CODE_QUESTIONS.get(
                code,
                f"Extract the value for {code} if present; otherwise leave empty.",
            )
            for code in remaining_before_retry
        }
        retry_payload = dict(user_payload)
        retry_payload["context_already_found"] = context_already_found
        retry_payload["remaining_codes"] = remaining_before_retry
        if retry_questions:
            retry_payload["questions"] = retry_questions
        retry_json = json.dumps(retry_payload, ensure_ascii=False)
        completion_retry, raw_retry = _openai_complete(conv, retry_json, api_key, model=model)
        retry_attributes = (
            completion_retry.get("attributes")
            if isinstance(completion_retry, Mapping)
            else []
        )
        retry_df = pd.DataFrame(retry_attributes or [])
        retry_map = _df_to_code_map(retry_df)
        raw_previews.append(_truncate_preview(raw_retry))

        for code, payload in retry_map.items():
            if not isinstance(payload, Mapping):
                continue
            value = payload.get("value")
            if _is_blank(value):
                continue
            if code in suggestions_map and not _is_blank(suggestions_map[code].get("value")):
                continue
            suggestions_map[code] = payload
            if code not in order:
                order.append(code)
        retried = True
        if suggestions_map:
            postprocess_adjustments.extend(
                _postprocess_suggestions(suggestions_map, session, api_base)
            )

    final_rows: list[dict[str, object]] = []
    for code in order:
        payload = suggestions_map.get(code)
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("value")
        if _is_blank(value):
            continue
        row_payload = dict(payload)
        row_payload["code"] = code
        final_rows.append(row_payload)

    if final_rows:
        result_df = pd.DataFrame(final_rows)
    else:
        result_df = pd.DataFrame(columns=["code", "value", "reason"])

    evidence_log: dict[str, str] = {}
    for row in final_rows:
        if not isinstance(row, Mapping):
            continue
        code = row.get("code")
        evidence = row.get("evidence")
        if isinstance(code, str) and isinstance(evidence, str) and evidence.strip():
            evidence_log[code] = evidence

    previews_for_log: list[str] = []
    if raw_previews:
        if raw_previews[0]:
            previews_for_log.append(f"initial: {raw_previews[0]}")
        if retried and len(raw_previews) > 1 and raw_previews[1]:
            previews_for_log.append(f"retry: {raw_previews[1]}")

    metadata = {
        "rules_hash": getattr(conv, "rules_hash", AI_RULES_HASH)
        if isinstance(conv, AiConversation)
        else AI_RULES_HASH,
        "used_hints": sanitized_hints,
        "retried": retried,
        "remaining_before_retry": remaining_before_retry,
        "remaining_all_codes": remaining_all,
        "spec_missing": spec_missing,
        "retry_questions": retry_questions,
        "retry_context": retry_context,
        "regex_hits": regex_hits,
        "brand_lexicon_hits": brand_lexicon_hits,
        "series_lexicon_hits": series_lexicon_hits,
        "lexicon_hits": {
            "brand": brand_lexicon_hits,
            "series": series_lexicon_hits,
        },
        "postprocess_adjustments": postprocess_adjustments,
        "raw_response_preview": "\n".join(previews_for_log),
        "evidence": evidence_log,
    }
    result_df.attrs["meta"] = metadata
    return result_df


def _first_sentence(text: Optional[object]) -> str:
    text = _strip_html(text)
    match = re.search(r"([^.?!]*[.?!])", text)
    if match:
        text = match.group(1).strip()
    return text[:200] + ("…" if len(text) > 200 else "")


def build_attributes_display_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """Prepare a human-friendly table with attribute values for Streamlit."""

    if df_full.empty:
        return pd.DataFrame(columns=["Attribute", "Value"])

    values: List[Tuple[str, str]] = []
    for code, row in df_full.iterrows():
        value = row.get("label")
        if value is None or (isinstance(value, str) and not value.strip()):
            value = row.get("raw_value")

        if value is None or (isinstance(value, str) and not value.strip()):
            display_value = "-"
        else:
            first_sentence = _first_sentence(value)
            display_value = first_sentence or str(value)
            if not str(display_value).strip():
                display_value = "-"

        values.append((str(code), str(display_value)))

    df_display = pd.DataFrame(values, columns=["Attribute", "Value"])
    return df_display
