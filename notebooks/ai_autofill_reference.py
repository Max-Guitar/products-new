# ============================================================
# MAGENTO ATTRIBUTES → AI AUTOFILL (GPT-5-mini) → PREVIEW
# ============================================================

# ---------- CONFIG ----------
MAGENTO_BASE_URL = "https://max.guitars"     # без /rest
MAGENTO_ADMIN_TOKEN = "YOUR_MAGENTO_ADMIN_TOKEN"
SKU = "ART-25648"

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"       # 🔑 ключ OpenAI
OPENAI_MODEL   = "gpt-5-mini"

ALWAYS_ATTRS = {"brand", "country_of_manufacture", "short_description"}

SET_ATTRS = {
    "Accessories": {"condition", "accessory_type", "strings", "cases_covers", "cables", "merchandise", "parts"},
    "Acoustic guitar": {"condition", "series", "acoustic_guitar_style", "acoustic_body_shape",
                        "body_material", "top_material", "neck_profile", "electro_acoustic",
                        "acoustic_cutaway", "no_strings", "orientation", "vintage", "kids_size",
                        "finish", "bridge", "controls", "neck_material", "neck_radius",
                        "tuning_machines", "fretboard_material", "scale_mensur",
                        "amount_of_frets", "acoustic_pickup", "cases_covers"},
    "Amps": {"amp_style", "condition", "type", "speaker_configuration",
             "built_in_fx", "cover_included", "footswitch_included", "vintage"},
    "Bass Guitar": {"condition", "guitarstylemultiplechoice", "series", "model", "acoustic_bass",
                    "pickup_config", "body_material", "top_material", "neck_profile", "no_strings",
                    "orientation", "vintage", "kids_size",
                    "finish", "bridge", "controls", "bridge_pickup", "neck_material",
                    "neck_radius", "middle_pickup", "neck_pickup", "neck_nutwidth",
                    "tuning_machines", "fretboard_material", "scale_mensur",
                    "amount_of_frets", "cases_covers"},
    "Default": {"name", "price"},
    "Effects": {"condition", "effect_type", "vintage", "controls", "power", "power_polarity", "battery"},
    "Electric guitar": {"condition", "guitarstylemultiplechoice", "series", "model", "semi_hollow_body",
                        "body_material", "top_material", "bridge_type", "pickup_config", "neck_profile",
                        "no_strings", "orientation", "vintage", "kids_size",
                        "finish", "bridge", "bridge_pickup", "middle_pickup", "neck_pickup",
                        "controls", "neck_material", "neck_radius", "neck_nutwidth",
                        "tuning_machines", "fretboard_material", "scale_mensur",
                        "amount_of_frets", "cases_covers"}
}

# ============================================================
#                 IMPLEMENTATION (HELPERS + RUN)
# ============================================================

import json, re, html, urllib.parse, requests, pandas as pd
from IPython.display import display, HTML

def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MagentoAIHelper/1.0"
    }

def _looks_like_json(text: str) -> bool:
    try: json.loads(text); return True
    except Exception: return False

def probe_api_base(origin: str, token: str) -> str:
    for path in ["/rest/V1", "/rest/default/V1", "/rest/all/V1", "/index.php/rest/V1"]:
        base = origin.rstrip("/") + path
        try:
            r = requests.get(f"{base}/store/storeViews", headers=_headers(token), timeout=10)
            if r.status_code in (200,401,403) and ("json" in r.headers.get("Content-Type","").lower() or _looks_like_json(r.text)):
                return base
        except Exception: pass
    raise RuntimeError("❌ Не удалось определить корректный REST-префикс.")

def get_product_by_sku(api_base, token, sku):
    url = f"{api_base}/products/{urllib.parse.quote(sku, safe='')}"
    r = requests.get(url, headers=_headers(token), timeout=20)
    if "text/html" in r.headers.get("Content-Type","").lower():
        raise RuntimeError(f"❌ HTML вместо JSON — проверь REST базу: {api_base}")
    if r.status_code == 404: raise RuntimeError(f"❌ SKU={sku} не найден.")
    if not r.ok: raise RuntimeError(f"Magento API error {r.status_code}: {r.text[:300]}")
    return r.json()

def get_attribute_sets_map(api_base, token):
    urls = [f"{api_base}/eav/attribute-sets/list?searchCriteria[currentPage]=1&searchCriteria[pageSize]=200",
            f"{api_base}/products/attribute-sets/sets/list?searchCriteria[currentPage]=1&searchCriteria[pageSize]=200"]
    for u in urls:
        r = requests.get(u, headers=_headers(token), timeout=20)
        if r.ok and ("json" in r.headers.get("Content-Type","").lower() or _looks_like_json(r.text)):
            items = r.json().get("items", r.json())
            return {i.get("attribute_set_id") or i.get("id"): i.get("attribute_set_name") or i.get("name") for i in items}
    raise RuntimeError("❌ Не удалось получить список attribute sets.")

_meta_cache = {}
def get_attribute_meta(api_base, token, code):
    if code in _meta_cache: return _meta_cache[code]
    url = f"{api_base}/products/attributes/{urllib.parse.quote(code, safe='')}"
    r = requests.get(url, headers=_headers(token), timeout=15)
    if r.ok and ("json" in r.headers.get("Content-Type","").lower() or _looks_like_json(r.text)):
        _meta_cache[code] = r.json(); return _meta_cache[code]
    _meta_cache[code] = {}; return {}

def option_label_for(code, raw_value):
    meta = _meta_cache.get(code) or {}
    inp = (meta.get("frontend_input") or "").lower()
    opts = meta.get("options") or []
    if inp in ("select","multiselect") and raw_value not in (None,"",[]):
        vals = [str(v).strip() for v in str(raw_value).split(",") if v.strip()]
        id2label = {str(o.get("value")): o.get("label") for o in opts if "value" in o}
        labels = [id2label.get(v,v) for v in vals]
        return ", ".join([l for l in labels if l]), inp
    return None, inp or None

def compute_allowed_attrs(attr_set_id, sets, id2name, always):
    a = set(always)
    if attr_set_id in sets: a |= sets[attr_set_id]
    name = id2name.get(attr_set_id)
    if isinstance(name,str) and name in sets: a |= sets[name]
    if "Default" in sets: a |= sets["Default"]
    return a

def collect_attributes_table(product, allowed, api_base, token):
    data = {c: {"raw":None,"label":None,"type":None} for c in allowed}
    for it in product.get("custom_attributes", []):
        c,v = it.get("attribute_code"), it.get("value")
        if c in data:
            meta = get_attribute_meta(api_base, token, c)
            lab,typ = option_label_for(c,v)
            data[c].update({"raw":v,"label":lab,"type":typ or meta.get("frontend_input")})
    for f in ["sku","name","price","weight"]:
        if f in allowed:
            v = product.get(f)
            meta = get_attribute_meta(api_base, token, f)
            lab,typ = option_label_for(f,v)
            data[f].update({"raw":v,"label":lab,"type":typ or meta.get("frontend_input")})
    df = pd.DataFrame(data).T.rename(columns={"raw":"raw_value"})
    return df

# ---------- OpenAI inference ----------
def _strip_html(s): return re.sub(r"<[^>]+>","",s or "")
def _openai_complete(system_msg, user_msg):
    r = requests.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {OPENAI_API_KEY}","Content-Type":"application/json"},
        json={"model":OPENAI_MODEL,"temperature":0,
              "response_format":{"type":"json_object"},
              "messages":[{"role":"system","content":system_msg},{"role":"user","content":user_msg}]},
        timeout=60)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])

def infer_missing(product, df_full, api_base, token, allowed):
    miss, ctx = [], {}
    for code,row in df_full.iterrows():
        val = row["label"] or row["raw_value"]
        if code in allowed and (val is None or str(val).strip()==""):
            meta = get_attribute_meta(api_base, token, code)
            miss.append({"code":code,"type":meta.get("frontend_input"),
                         "options":[o["label"] for o in meta.get("options",[]) if o.get("label")]})
        else: ctx[code]=str(val)
    desc = _strip_html(next((a["value"] for a in product["custom_attributes"] if a["attribute_code"]=="short_description"),""))
    msg = {"product":{"sku":product["sku"],"name":product["name"],"description":desc},
           "known_values":ctx,"missing":miss}
    sys = "Ты помощник по каталогизации гитар. Используй данные о товаре, чтобы заполнить пустые поля. Ответ JSON {attributes:[{code,value,reason}]}."
    out = _openai_complete(sys,json.dumps(msg,ensure_ascii=False))
    return pd.DataFrame(out.get("attributes",[]))

# ---------- Scrollable summary ----------
def _one_sentence(s):
    s = _strip_html(s or "").strip()
    m = re.search(r"([^.?!]*[.?!])", s)
    s = (m.group(1) if m else s).strip()
    return s[:200]+("…" if len(s)>200 else "")

def show_scrollable_ai(df_full, df_suggest):
    ai = {r["code"]:r["value"] for _,r in df_suggest.iterrows()}
    flat = {c:(df_full.loc[c,"label"] or df_full.loc[c,"raw_value"]) for c in df_full.index}
    flat.update(ai)
    cols=list(flat.keys())
    ths="".join(f"<th>{c}</th>" for c in cols)
    tds=[]
    for c in cols:
        val=_one_sentence(flat[c])
        mark=" 🤖" if c in ai else ""
        style="background:#fff7cc;" if c in ai else ""
        tds.append(f"<td style='{style}'>{html.escape(str(val))}{mark}</td>")
    html_table=f"""
    <div style="overflow-x:auto;border:1px solid #ddd;border-radius:6px;">
    <table style="border-collapse:collapse;font-family:monospace;">
      <thead><tr>{ths}</tr></thead><tbody><tr>{''.join(tds)}</tr></tbody>
    </table></div>"""
    display(HTML(html_table))

# ============================================================
#                         RUN
# ============================================================

try:
    api_base = probe_api_base(MAGENTO_BASE_URL, MAGENTO_ADMIN_TOKEN)
    product  = get_product_by_sku(api_base, MAGENTO_ADMIN_TOKEN, SKU)
    id2name  = get_attribute_sets_map(api_base, MAGENTO_ADMIN_TOKEN)
    set_id   = product["attribute_set_id"]
    set_name = id2name.get(set_id,"Unknown")
    allowed  = compute_allowed_attrs(set_id, SET_ATTRS, id2name, ALWAYS_ATTRS)

    print(f"API base: {api_base}\nSKU: {SKU} | Set: {set_name} ({set_id})")

    df_full = collect_attributes_table(product, allowed, api_base, MAGENTO_ADMIN_TOKEN)
    df_suggest = infer_missing(product, df_full, api_base, MAGENTO_ADMIN_TOKEN, allowed)
    print(f"\nAI предложил {len(df_suggest)} значений, модель {OPENAI_MODEL}:\n")
    show_scrollable_ai(df_full, df_suggest)

except Exception as e:
    print("Ошибка:", e)
