"""AI-assisted attribute autofill helpers based on the Colab reference notebook."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import openai
import pandas as pd
import requests
import streamlit as st

from utils.http import MAGENTO_REST_PATH_CANDIDATES, build_magento_headers


ALWAYS_ATTRS: Set[str] = {"brand", "country_of_manufacture", "short_description"}

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
    system_msg: str,
    user_msg: str,
    api_key: str,
    model: str = "gpt-5-mini",
    timeout: int = 60,
) -> dict:
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured.")

    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            request_timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - network error handling
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI response missing message content") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI response was not valid JSON") from exc


def infer_missing(
    product: Mapping[str, object],
    df_full: pd.DataFrame,
    session: requests.Session,
    api_base: str,
    allowed: Sequence[str],
    api_key: str,
    model: str = "gpt-5-mini",
) -> pd.DataFrame:
    missing: List[dict] = []
    known: Dict[str, str] = {}
    for code, row in df_full.iterrows():
        current_value = row["label"] or row["raw_value"]
        if code in allowed and (current_value is None or str(current_value).strip() == ""):
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
        else:
            known[code] = str(current_value)
    short_desc = ""
    for attr in product.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == "short_description":
            short_desc = _strip_html(attr.get("value"))
            break
    user_payload = {
        "product": {
            "sku": product.get("sku"),
            "name": product.get("name"),
            "description": short_desc,
        },
        "known_values": known,
        "missing": missing,
    }
    system_message = (
        "Ты помощник по каталогизации гитар. Используй данные о товаре, чтобы заполнить "
        "пустые поля. Ответ JSON {\"attributes\":[{code,value,reason}]}."
    )
    completion = _openai_complete(
        system_message, json.dumps(user_payload, ensure_ascii=False), api_key, model=model
    )
    return pd.DataFrame(completion.get("attributes", []))


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
