"""AI-assisted attribute autofill helpers based on the Colab reference notebook."""

from __future__ import annotations

import html
import json
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


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


def _looks_like_json(text: str) -> bool:
    try:
        json.loads(text)
    except Exception:
        return False
    return True


def probe_api_base(session: requests.Session, origin: str) -> str:
    """Discover a working Magento REST prefix for ``origin``."""

    origin = origin.rstrip("/")
    for path in ("/rest/V1", "/rest/default/V1", "/rest/all/V1", "/index.php/rest/V1"):
        base = f"{origin}{path}"
        try:
            resp = session.get(f"{base}/store/storeViews", timeout=10)
        except requests.RequestException:
            continue
        content_type = resp.headers.get("Content-Type", "").lower()
        if resp.status_code in {200, 401, 403} and (
            "json" in content_type or _looks_like_json(resp.text)
        ):
            return base
    raise RuntimeError("❌ Не удалось определить корректный REST-префикс.")


def _build_url(api_base: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{api_base.rstrip('/')}{path}"


def get_product_by_sku(session: requests.Session, api_base: str, sku: str) -> dict:
    url = _build_url(api_base, f"/products/{quote(sku, safe='')}")
    resp = session.get(url, timeout=20)
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
            resp = session.get(url, timeout=20)
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
        resp = session.get(url, timeout=15)
    except requests.RequestException:
        _meta_cache[code] = {}
        return _meta_cache[code]
    content_type = resp.headers.get("Content-Type", "").lower()
    if resp.ok and ("json" in content_type or _looks_like_json(resp.text)):
        _meta_cache[code] = resp.json()
    else:
        _meta_cache[code] = {}
    return _meta_cache[code]


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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return json.loads(data["choices"][0]["message"]["content"])


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


def show_scrollable_ai(df_full: pd.DataFrame, df_suggest: pd.DataFrame) -> None:
    ai = {r["code"]: r["value"] for _, r in df_suggest.iterrows()}
    flattened = {
        c: (df_full.loc[c, "label"] or df_full.loc[c, "raw_value"])
        for c in df_full.index
    }

    flattened.update(ai)
    columns = list(flattened.keys())

    ths = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    tds: List[str] = []
    for code in columns:
        val = _first_sentence(flattened.get(code))
        is_ai = code in ai
        mark = " 🤖" if is_ai else ""
        style = "background:#fff7cc;" if is_ai else ""
        tds.append(f"<td style='{style}'>{html.escape(val)}{mark}</td>")

    html_table = f"""
    <div style="overflow-x:auto;border:1px solid #ddd;border-radius:6px;">
    <table style="border-collapse:collapse;font-family:monospace;">
      <thead><tr>{ths}</tr></thead>
      <tbody><tr>{''.join(tds)}</tr></tbody>
    </table>
    </div>
    """
    st.markdown(html_table, unsafe_allow_html=True)
