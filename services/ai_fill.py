"""AI-assisted attribute autofill helpers based on the Colab reference notebook."""

from __future__ import annotations

import json
import os
import re
import hashlib
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

AI_RULES_TEXT = (
    "Ты помощник по каталогизации гитар. Используй входные данные, чтобы заполнить "
    "пропущенные атрибуты товара. Всегда отвечай JSON вида "
    "{\"attributes\":[{\"code\":...,\"value\":...,\"reason\":...}]}. Правила:\n"
    "- Бренд и название набора атрибутов помогают подобрать категории по совпадению "
    "названий; если нет уверенности — оставляй поле пустым.\n"
    "- Ищи упоминания серии и модели в названии, используй их в подсказках.\n"
    "- Соотносить модели Telecaster→T-Style, Stratocaster→S-Style, Les Paul→Single cut, "
    "SG→Double cut.\n"
    "- При наличии T-Style добавляй также Single cut; при наличии S-Style добавляй "
    "Double cut.\n"
    "- Значения мультиселектов представляй как массив строк, без дублей.\n"
    "- Булевы значения задавай как true/false.\n"
    "- Для категорий используй текстовые лейблы.\n"
    "- Если данных недостаточно — возвращай пустое значение."
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

    # Categories enrichment with brand/set hints
    existing_categories_raw = _ensure_list_of_strings(
        (by_code.get("categories") or {}).get("value")
    )
    existing_categories_canon: list[str] = []
    for item in existing_categories_raw:
        normalized_label = normalize_category_label(item, categories_meta)
        if normalized_label and normalized_label not in existing_categories_canon:
            existing_categories_canon.append(normalized_label)

    brand_hint_values = _ensure_list_of_strings(normalized_hints.get("brand_hint"))
    set_hint_values = _ensure_list_of_strings(normalized_hints.get("set_hint"))
    category_hints: list[str] = []
    category_hints.extend(brand_hint_values)
    category_hints.extend(set_hint_values)
    if attribute_set_name and not _is_blank(attribute_set_name):
        category_hints.append(str(attribute_set_name).strip())

    merged_categories = _merge_unique(existing_categories_raw, category_hints)
    normalized_categories: list[str] = []
    unmatched_labels: list[str] = []
    for label in merged_categories:
        normalized_label = normalize_category_label(label, categories_meta)
        if normalized_label:
            if normalized_label not in normalized_categories:
                normalized_categories.append(normalized_label)
        else:
            unmatched_labels.append(label)

    normalized_categories_added = [
        label for label in normalized_categories if label not in existing_categories_canon
    ]

    if normalized_categories:
        category_entry = by_code.setdefault(
            "categories",
            {"code": "categories", "reason": "enriched_from_hints"},
        )
        category_entry["value"] = normalized_categories
        if _is_blank(category_entry.get("reason")):
            category_entry["reason"] = "enriched_from_hints"
        if unmatched_labels:
            category_entry["unmatched_labels"] = _ensure_list_of_strings(unmatched_labels)
        if normalized_categories_added:
            category_entry["normalized_categories_added"] = list(normalized_categories_added)
        if "categories" not in order:
            order.append("categories")
    elif unmatched_labels:
        category_entry = by_code.setdefault(
            "categories",
            {"code": "categories", "reason": "enriched_from_hints"},
        )
        category_entry["unmatched_labels"] = _ensure_list_of_strings(unmatched_labels)
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

    set_name_cf = str(attribute_set_name or "").casefold()
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
    if merged_styles:
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
        normalized = normalize_category_label(hint_value, categories_meta)
        if normalized:
            brand_hint_meta = normalized
            break
    if brand_hint_meta is None and brand_hint_values:
        brand_hint_meta = brand_hint_values[0]
    meta_payload["brand_hint"] = brand_hint_meta
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
    force_codes: Set[str] = {"categories", "guitarstylemultiplechoice"}
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

    short_desc = ""
    for attr in product.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == "short_description":
            short_desc = _strip_html(attr.get("value"))
            break

    sanitized_hints = _sanitize_hints(hints)
    attribute_set_name = sanitized_hints.get("attribute_set_name") or sanitized_hints.get(
        "set_hint"
    )

    user_payload = {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "description": short_desc,
        "attribute_set_name": attribute_set_name,
        "current_attributes": current_attributes,
        "ai_codes": ai_codes,
        "hints": sanitized_hints,
        "known_values": known,
        "missing": missing,
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
    order: list[str] = list(ai_codes)
    for code in suggestions_map.keys():
        if code not in order:
            order.append(code)

    raw_previews: list[str] = [_truncate_preview(raw_content)]
    remaining_before_retry = [
        code for code in ai_codes if _is_blank(suggestions_map.get(code, {}).get("value"))
    ]
    retried = False

    if remaining_before_retry:
        context_already_found = {
            code: entry.get("value")
            for code, entry in suggestions_map.items()
            if not _is_blank(entry.get("value"))
        }
        retry_payload = dict(user_payload)
        retry_payload["context_already_found"] = context_already_found
        retry_payload["remaining_codes"] = remaining_before_retry
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
        "raw_response_preview": "\n".join(previews_for_log),
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
