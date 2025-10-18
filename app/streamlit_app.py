from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import quote
import re

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

st.set_page_config(
    page_title="🤖 Peter v.1.0 (AI Content Manager)",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 Peter v.1.0 (AI Content Manager)")

st.markdown(
    """
<style>
.ag-theme-balham .ag-cell,
.ag-theme-balham .ag-header-cell {
  border-right: 1px solid #eee !important;
}
.ag-theme-balham .ag-header {
  border-bottom: 1px solid #e6e6e6 !important;
}
.ag-theme-balham .ag-root-wrapper,
.ag-theme-balham .ag-root-wrapper-body,
.ag-theme-balham .ag-center-cols-viewport {
  overflow: visible !important;
}
.ag-popup, .ag-popup-editor, .ag-menu, .ag-select-list {
  z-index: 9999 !important;
}
.block-container, .stApp {
  overflow: visible !important;
}
</style>
""",
    unsafe_allow_html=True,
)

from connectors.magento.attributes import AttributeMetaCache
from connectors.magento.categories import ensure_categories_meta, get_categories_tree
from connectors.magento.client import get_default_products
from services.ai_fill import (
    ALWAYS_ATTRS,
    SET_ATTRS,
    HTMLResponseError,
    collect_attributes_table,
    compute_allowed_attrs,
    get_attribute_sets_map,
    get_product_by_sku,
    probe_api_base,
)
from services.inventory import (
    get_attr_set_id,
    get_backorders_parallel,
    detect_default_source_code,
    get_source_items,
    iter_products_by_attr_set,
    list_attribute_sets,
)
from services.normalizers import normalize_for_magento
from utils.http import build_magento_headers, get_session


_DEF_ATTR_SET_NAME = "Default"
_ALLOWED_TYPES = {"simple", "configurable"}


_ATTRIBUTE_SET_ICONS = {
    "Accessories": "🧩",
    "Acoustic guitar": "🎻",
    "Amps": "🎚️",
    "Bass Guitar": "🎸",
    "Default": "🧾",
    "Effects": "🎛️",
    "Electric guitar": "🎸",
}
_DEFAULT_ATTRIBUTE_ICON = "🧩"


EMPTY_ID = ""
EMPTY_LABEL = "(none)"


TagCellRenderer = JsCode(
    """
class TagCellRenderer {
  init(params) {
    const v2l = (params.colDef.cellRendererParams && params.colDef.cellRendererParams.v2l) || {};
    const toLabel = (x) => v2l[String(x)] || String(x || "");

    this.eGui = document.createElement('div');
    this.eGui.style.display = 'flex';
    this.eGui.style.flexWrap = 'nowrap';
    this.eGui.style.alignItems = 'center';
    this.eGui.style.gap = '6px';
    this.eGui.style.overflow = 'hidden';
    this.eGui.style.whiteSpace = 'nowrap';
    this.eGui.style.textOverflow = 'ellipsis';

    const values = Array.isArray(params.value) ? params.value : (params.value ? [params.value] : []);
    const labels = values.map(toLabel);

    labels.forEach(lbl => {
      const pill = document.createElement('span');
      pill.textContent = lbl;
      pill.style.display = 'inline-flex';
      pill.style.alignItems = 'center';
      pill.style.padding = '2px 8px';
      pill.style.border = '1px solid #DDD';
      pill.style.borderRadius = '999px';
      pill.style.background = '#f7f7f9';
      pill.style.fontSize = '12px';
      pill.style.whiteSpace = 'nowrap';
      this.eGui.appendChild(pill);
    });
  }
  getGui(){ return this.eGui; }
  refresh(){ return false; }
}
"""
)


TagMultiEditor = JsCode(
    """
class TagMultiEditor {
  init(params) {
    this.params = params;
    const labels = (params && params.values) || [];
    this.v2l = (params && params.v2l) || {};
    this.l2v = (params && params.l2v) || {};
    const toId  = (s) => this.l2v.hasOwnProperty(String(s)) ? this.l2v[String(s)] : String(s);
    const toLbl = (id) => this.v2l.hasOwnProperty(String(id)) ? this.v2l[String(id)] : String(id);

    this.selected = Array.isArray(params.value) ? params.value.map(v => String(v)) :
                    (params.value ? [String(params.value)] : []);

    this.root = document.createElement('div');
    this.root.style.padding = '8px';
    this.root.style.minWidth = '360px';
    this.root.style.maxWidth = '560px';

    this.tags = document.createElement('div');
    this.tags.style.display = 'flex';
    this.tags.style.flexWrap = 'nowrap';
    this.tags.style.gap = '6px';
    this.tags.style.whiteSpace = 'nowrap';
    this.tags.style.overflow = 'hidden';
    this.tags.style.textOverflow = 'ellipsis';
    this.tags.style.marginBottom = '8px';
    this.root.appendChild(this.tags);

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.gap = '8px';

    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.placeholder = 'Type to search…';
    this.input.style.flex = '1';
    this.input.style.padding = '6px 8px';
    this.input.style.border = '1px solid #DDD';
    this.input.style.borderRadius = '6px';

    this.addBtn = document.createElement('button');
    this.addBtn.textContent = 'Add';
    this.addBtn.style.padding = '6px 10px';

    row.appendChild(this.input);
    row.appendChild(this.addBtn);
    this.root.appendChild(row);

    this.sugg = document.createElement('div');
    this.sugg.style.border = '1px solid #EEE';
    this.sugg.style.marginTop = '6px';
    this.sugg.style.maxHeight = '0px';
    this.root.appendChild(this.sugg);

    const renderTags = () => {
      this.tags.innerHTML = '';
      this.selected.forEach(id => {
        const pill = document.createElement('span');
        pill.textContent = toLbl(id);
        pill.style.padding = '2px 8px';
        pill.style.border = '1px solid #DDD';
        pill.style.borderRadius = '999px';
        pill.style.background = '#f7f7f9';
        pill.style.fontSize = '12px';
        pill.style.display = 'inline-flex';
        pill.style.alignItems = 'center';
        pill.style.gap = '6px';
        pill.style.whiteSpace = 'nowrap';

        const x = document.createElement('span');
        x.textContent = '×';
        x.style.cursor = 'pointer';
        x.onclick = () => { this.selected = this.selected.filter(v => v !== id); renderTags(); };
        pill.appendChild(x);
        this.tags.appendChild(pill);
      });
    };

    const addLabel = (lbl) => {
      const s = String(lbl || '').trim();
      if (!s) return;
      const id = toId(s);
      if (!this.selected.includes(String(id))) {
        this.selected.push(String(id));
        renderTags();
      }
    };

    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { addLabel(this.input.value); this.input.value=''; }
      if (e.key === 'Backspace' && !this.input.value) { this.selected.pop(); renderTags(); }
    });
    this.addBtn.onclick = () => { addLabel(this.input.value); this.input.value=''; };

    renderTags();
  }
  getGui(){ return this.root; }
  afterGuiAttached(){ if (this.input) this.input.focus(); }
  getValue(){ return this.selected.slice(); }
  destroy(){}
  isPopup(){ return true; }
}
"""
)


TagOneLineRenderer = JsCode(
    """
class TagOneLineRenderer {
  init(p){
    const map = (p.colDef.cellRendererParams && p.colDef.cellRendererParams.v2l) || {};
    this.eGui = document.createElement('div');
    this.eGui.style.display='flex';
    this.eGui.style.flexWrap='nowrap';
    this.eGui.style.alignItems='center';
    this.eGui.style.gap='6px';
    this.eGui.style.overflow='hidden';
    this.eGui.style.whiteSpace='nowrap';
    this.eGui.style.textOverflow='ellipsis';
    const values = Array.isArray(p.value) ? p.value : (p.value ? [p.value] : []);
    values.map(v=>map[String(v)]||String(v||"")).forEach(lbl=>{
      const pill=document.createElement('span');
      pill.textContent=lbl;
      pill.style.padding='2px 8px';
      pill.style.border='1px solid #DDD';
      pill.style.borderRadius='999px';
      pill.style.background='#f7f7f9';
      pill.style.fontSize='12px';
      pill.style.whiteSpace='nowrap';
      this.eGui.appendChild(pill);
    });
  }
  getGui(){return this.eGui}
  refresh(){return false}
}
"""
)


TreeMultiEditor = JsCode(
    """
class TreeMultiEditor {
  init(p){
    this.p=p;
    const tree = (p && p.tree) || [];
    const v2l  = (p && p.v2l)  || {};
    const sel = Array.isArray(p.value) ? p.value.map(String) : (p.value?[String(p.value)]:[]);
    this.selected = new Set(sel);

    this.root = document.createElement('div');
    this.root.style.minWidth='540px';
    this.root.style.maxWidth='740px';
    this.root.style.maxHeight='70vh';
    this.root.style.overflow='auto';
    this.root.style.padding='8px';

    const sWrap=document.createElement('div');
    sWrap.style.display='flex';
    sWrap.style.marginBottom='8px';
    const input=document.createElement('input');
    input.type='text'; input.placeholder='Search categories...';
    input.style.flex='1'; input.style.padding='6px 8px';
    input.style.border='1px solid #DDD'; input.style.borderRadius='6px';
    sWrap.appendChild(input); this.root.appendChild(sWrap);

    this.treeBox=document.createElement('div'); this.root.appendChild(this.treeBox);

    const renderNode=(node, level=0)=>{
      const row=document.createElement('div');
      row.style.display='flex'; row.style.alignItems='center';
      row.style.gap='8px'; row.style.padding='4px 4px 4px '+(12*level)+'px';
      const cb=document.createElement('input'); cb.type='checkbox';
      const id=String(node.id||""); const name=String(node.name||"");
      cb.checked=this.selected.has(id);
      cb.onchange=()=>{ if(cb.checked) this.selected.add(id); else this.selected.delete(id); };
      const lbl=document.createElement('span'); lbl.textContent=name;
      row.appendChild(cb); row.appendChild(lbl);
      return {row, cb};
    };

    const renderTree=(nodes, box, level=0)=>{
      (nodes||[]).forEach(nd=>{
        const {row}=renderNode(nd, level);
        box.appendChild(row);
        const ch=nd.children_data||[];
        if(ch.length){
          renderTree(ch, box, level+1);
        }
      });
    };

    const fullTree = (this.p.treeData || []).length ? this.p.treeData : tree;
    const build = (filter='')=>{
      this.treeBox.innerHTML='';
      if(!filter){
        renderTree(fullTree, this.treeBox, 0);
      }else{
        const q=filter.toLowerCase();
        Object.keys(v2l).forEach(id=>{
          const path=v2l[id].toLowerCase();
          if(path.includes(q)){
            const row=document.createElement('div');
            row.style.display='flex'; row.style.alignItems='center';
            row.style.gap='8px'; row.style.padding='4px';
            const cb=document.createElement('input'); cb.type='checkbox';
            cb.checked=this.selected.has(id);
            cb.onchange=()=>{ if(cb.checked) this.selected.add(id); else this.selected.delete(id); };
            const lbl=document.createElement('span'); lbl.textContent=v2l[id];
            row.appendChild(cb); row.appendChild(lbl);
            this.treeBox.appendChild(row);
          }
        });
      }
    };

    input.addEventListener('input', ()=>build(input.value));
    build('');
  }
  getGui(){ return this.root }
  afterGuiAttached(){}
  getValue(){ return Array.from(this.selected) }
  isPopup(){ return true }
  destroy(){}
}
"""
)


SearchableSelectEditor = JsCode(
    """
class SearchableSelectEditor {
  init(p){
    this.p=p;
    const labels=(p && p.values)||[];
    this.l2v=p.l2v||{};
    const toId=(s)=>this.l2v.hasOwnProperty(String(s))?this.l2v[String(s)]:String(s);
    const cur=p.value!=null?String(p.value):"";

    this.root=document.createElement('div');
    this.root.style.minWidth='280px'; this.root.style.maxHeight='50vh';
    this.root.style.overflow='auto'; this.root.style.padding='8px';

    const input=document.createElement('input');
    input.type='text'; input.placeholder='Search...';
    input.style.width='100%'; input.style.padding='6px 8px';
    input.style.border='1px solid #DDD'; input.style.borderRadius='6px';
    this.root.appendChild(input);

    const box=document.createElement('div');
    box.style.marginTop='6px'; this.root.appendChild(box);

    const render=(q='')=>{
      box.innerHTML='';
      const pool=labels.filter(l=>l.toLowerCase().includes(q.toLowerCase()));
      pool.forEach(lbl=>{
        const row=document.createElement('div');
        row.textContent=lbl; row.style.padding='6px 8px'; row.style.cursor='pointer';
        row.onmouseenter=()=>row.style.background='#f0f4ff';
        row.onmouseleave=()=>row.style.background='white';
        row.onclick=()=>{ this.value=toId(lbl); };
        box.appendChild(row);
      });
    };
    input.addEventListener('input', ()=>render(input.value));
    render('');
    this.value=cur;
  }
  getGui(){return this.root}
  afterGuiAttached(){}
  getValue(){return this.value}
  isPopup(){return true}
  destroy(){}
}
"""
)


HtmlEditor = JsCode(
    """
class HtmlEditor {
  init(p){
    this.p=p;
    const val = (p.value==null) ? "" : String(p.value);
    this.root=document.createElement('div');
    this.root.style.minWidth='500px'; this.root.style.maxWidth='800px';
    this.root.style.maxHeight='70vh'; this.root.style.overflow='auto';
    const box=document.createElement('div');
    box.contentEditable='true';
    box.style.minHeight='140px';
    box.style.padding='8px';
    box.style.border='1px solid #DDD';
    box.style.borderRadius='6px';
    box.innerHTML=val;
    this.root.appendChild(box);
    this.box=box;
  }
  getGui(){return this.root}
  afterGuiAttached(){ this.box.focus(); }
  getValue(){ return this.box.innerHTML; }
  isPopup(){ return true }
  destroy(){}
}
"""
)


BASE_ORDER = [
    "sku",
    "name",
    "price",
    "condition",
    "country_of_manufacture",
    "brand",
]

ALIASES = {
    "product type (attribute set)": "attribute_set_id",
    "attribute set": "attribute_set_id",
    "cases covers": "cases_covers",
    "short description": "short_description",
    "guitar style": "guitarstylemultiplechoice",
    "series": "series",
    "model": "model",
    "amp type": "amp_type",
    "effect type": "effect_type",
    "accessory type": "accessory_type",
    "power (watt)": "power_watt",
    "built in fx": "built_in_fx",
    "footswitch included": "footswitch_included",
    "cover included": "cover_included",
    "power polarity": "power_polarity",
}


def norm_name(s: str) -> str:
    k = (s or "").strip().lower()
    return ALIASES.get(k, k).replace(" ", "_")


ORDER_BY_SET_NAME = {
    "acoustic guitar": [
        "series",
        "acoustic_guitar_style",
        "acoustic_body_shape",
        "body_material",
        "top_material",
        "finish",
        "bridge",
        "controls",
        "acoustic_pickup",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "acoustic_cutaway",
        "electro_acoustic",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],
    "electric guitar": [
        "series",
        "model",
        "guitarstylemultiplechoice",
        "body_material",
        "top_material",
        "finish",
        "bridge_type",
        "bridge",
        "pickup_config",
        "bridge_pickup",
        "middle_pickup",
        "neck_pickup",
        "controls",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "semi_hollow_body",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],
    "electric guitars": [
        "series",
        "model",
        "guitarstylemultiplechoice",
        "body_material",
        "top_material",
        "finish",
        "bridge_type",
        "bridge",
        "pickup_config",
        "bridge_pickup",
        "middle_pickup",
        "neck_pickup",
        "controls",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "semi_hollow_body",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],
    "bass guitar": [
        "series",
        "model",
        "body_material",
        "top_material",
        "finish",
        "bridge",
        "pickup_config",
        "bridge_pickup",
        "middle_pickup",
        "neck_pickup",
        "controls",
        "neck_profile",
        "neck_material",
        "neck_radius",
        "neck_nutwidth",
        "fretboard_material",
        "tuning_machines",
        "scale_mensur",
        "amount_of_frets",
        "no_strings",
        "orientation",
        "acoustic_bass",
        "kids_size",
        "vintage",
        "cases_covers",
        "categories",
        "short_description",
    ],
    "amps": [
        "series",
        "model",
        "amp_type",
        "built_in_fx",
        "cover_included",
        "footswitch_included",
        "battery",
        "power",
        "power_watt",
        "power_polarity",
        "controls",
        "categories",
        "short_description",
    ],
    "effects": [
        "series",
        "model",
        "brand",
        "effect_type",
        "controls",
        "battery",
        "power",
        "power_polarity",
        "vintage",
        "categories",
        "short_description",
    ],
    "accessories": [
        "accessory_type",
        "type",
        "cables",
        "strings",
        "cases_covers",
        "merchandise",
        "parts",
        "categories",
        "short_description",
    ],
}

ORDER_BY_SET_NAME["acoustic guitars"] = ORDER_BY_SET_NAME["acoustic guitar"]
ORDER_BY_SET_NAME["bass guitars"] = ORDER_BY_SET_NAME["bass guitar"]
ORDER_BY_SET_NAME["amplifiers"] = ORDER_BY_SET_NAME["amps"]


def build_column_order_for_set(
    df_cols: list[str], set_name: str | None
) -> list[str]:
    normalized_to_actual: dict[str, str] = {}
    ordered_actual: list[str] = []
    for col in df_cols:
        if col == "attribute_set_id":
            continue
        normalized = norm_name(col)
        if normalized not in normalized_to_actual:
            normalized_to_actual[normalized] = col
            ordered_actual.append(col)

    name = (set_name or "").strip().lower()

    base_candidates = [norm_name(c) for c in BASE_ORDER]
    base = [
        normalized_to_actual[norm]
        for norm in base_candidates
        if norm in normalized_to_actual
    ]

    specific_list = ORDER_BY_SET_NAME.get(name, [])
    specific_candidates = [norm_name(c) for c in specific_list]
    specific = [
        normalized_to_actual[norm]
        for norm in specific_candidates
        if norm in normalized_to_actual
    ]

    seen: set[str] = set()
    ordered: list[str] = []
    for seq in (base, specific):
        for actual in seq:
            if actual not in seen:
                seen.add(actual)
                ordered.append(actual)

    for actual in ordered_actual:
        if actual not in seen:
            seen.add(actual)
            ordered.append(actual)

    return ordered


ID_RX = re.compile(r"#(\d+)\)?$")


DEBUG = False


def _get_custom_attr_value(item: dict, code: str, default=None):
    for attr in item.get("custom_attributes", []) or []:
        if attr.get("attribute_code") == code:
            return attr.get("value", default)
    return default


def _attr_label(meta: dict, code: str) -> str:
    for key in ("default_frontend_label", "frontend_label", "store_label"):
        label = meta.get(key)
        if isinstance(label, str) and label.strip():
            return label.strip()
    if code == "sku":
        return "SKU"
    if code == "name":
        return "Name"
    return code.replace("_", " ").title()


def _meta_options(meta: dict) -> list:
    options = []
    for opt in meta.get("options", []) or []:
        if not isinstance(opt, dict):
            continue
        label = opt.get("label")
        if not isinstance(label, str) or not label.strip():
            label = str(opt.get("value", "")).strip()
        else:
            label = label.strip()
        if label:
            options.append(label)
    # preserve order but deduplicate while keeping first occurrence
    seen = set()
    unique = []
    for item in options:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _meta_options_list(meta: dict) -> list[str]:
    return [
        str(opt.get("label", "")).strip()
        for opt in (meta.get("options") or [])
        if isinstance(opt, dict)
        and str(opt.get("label", "")).strip()
    ]


def _v2l(meta: dict) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in (meta.get("values_to_labels") or {}).items()
    }


def _l2v(meta: dict) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in (meta.get("options_map") or {}).items()
    }


def _inject_empty_option(meta: dict):
    opts = meta.setdefault("options", [])
    has_empty = any(
        str(o.get("value", "")) == "" for o in opts if isinstance(o, dict)
    )
    if not has_empty:
        opts.insert(0, {"label": EMPTY_LABEL, "value": EMPTY_ID})
    v2l = meta.setdefault("values_to_labels", {})
    l2v = meta.setdefault("options_map", {})
    v2l[str(EMPTY_ID)] = EMPTY_LABEL
    l2v[str(EMPTY_LABEL)] = str(EMPTY_ID)


def _coerce_multiselect_cell(v, l2v: dict[str, str]) -> list[str]:
    if v is None or v == "" or v == []:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for el in v:
            if isinstance(el, dict):
                val = (
                    el.get("value")
                    or el.get("id")
                    or el.get("category_id")
                )
                if val is None:
                    lbl = str(el.get("label", "")).strip()
                    if lbl:
                        val = l2v.get(lbl) or lbl
                if val is not None:
                    out.append(str(val))
            else:
                s = str(el).strip()
                out.append(l2v.get(s, s))
        return out
    s = str(v).strip()
    if "," in s:
        return [
            l2v.get(x.strip(), x.strip())
            for x in s.split(",")
            if x.strip()
        ]
    return [l2v.get(s, s)]


def _attr_set_icon(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return _DEFAULT_ATTRIBUTE_ICON
    return _ATTRIBUTE_SET_ICONS.get(name.strip(), _DEFAULT_ATTRIBUTE_ICON)


def _format_attr_set_title(name: str) -> str:
    clean_name = name.strip() if isinstance(name, str) and name.strip() else "—"
    icon = _attr_set_icon(clean_name)
    return f"{icon} {clean_name}"


def _split_multiselect_input(values) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        iterable = [str(item).strip() for item in values]
    elif isinstance(values, str):
        iterable = [item.strip() for item in values.split(",")]
    elif values in (None, ""):
        iterable = []
    elif isinstance(values, float) and pd.isna(values):
        iterable = []
    else:
        iterable = [str(values).strip()]

    return [item for item in iterable if item]


def _format_multiselect_display_value(values) -> str:
    parts = _split_multiselect_input(values)
    if not parts:
        return ""
    return ", ".join(parts)


def _pin_sku_name(order: Iterable[str], df_cols: Iterable[str]) -> list[str]:
    if not isinstance(order, list):
        try:
            order = list(order or [])  # type: ignore[arg-type]
        except TypeError:
            order = []
    if not isinstance(df_cols, list):
        try:
            df_cols = list(df_cols or [])  # type: ignore[arg-type]
        except TypeError:
            df_cols = []
    lead = [c for c in ("sku", "name") if c in df_cols]
    rest = [c for c in order if c not in lead and c in df_cols]
    missing = [c for c in df_cols if c not in lead and c not in rest]
    return lead + rest + missing


def _parse_category_token(token: object) -> int | None:
    if token is None:
        return None
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, float):
        if math.isnan(token):
            return None
        try:
            return int(token)
        except (TypeError, ValueError):
            return None
    text = str(token).strip()
    if not text:
        return None
    match = ID_RX.search(text)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _cat_to_ids(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[int] = []
        for item in value:
            cid = _parse_category_token(item)
            if cid is not None and cid not in result:
                result.append(cid)
        return result
    cid = _parse_category_token(value)
    return [cid] if cid is not None else []


def _ids_to_labels(ids: list[int], values_to_labels: dict[str, str]) -> list[str]:
    labels: list[str] = []
    for cid in ids or []:
        key = str(cid)
        label = values_to_labels.get(key)
        if label and label not in labels:
            labels.append(label)
    return labels


def _labels_to_ids(labels: list[str], options_map: dict[str, int]) -> list[int]:
    if not labels:
        return []
    ids: list[int] = []
    for label in labels:
        if label is None:
            continue
        text = str(label).strip()
        if not text:
            continue
        if text in options_map:
            cid = options_map[text]
        else:
            lower = text.casefold()
            cid = options_map.get(lower)
            if cid is None:
                try:
                    cid = int(text)
                except (TypeError, ValueError):
                    continue
        if isinstance(cid, float) and math.isnan(cid):
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        if cid_int not in ids:
            ids.append(cid_int)
    return ids


def _categories_value_to_ids(value: object, options_map: dict[str, int]) -> list[int]:
    ids = _cat_to_ids(value)
    if ids:
        return ids
    labels: list[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                labels.append(text)
    else:
        labels = _split_multiselect_input(value)
    labels = [label for label in labels if label]
    if not labels:
        return []
    return _labels_to_ids(labels, options_map)


def _convert_df_for_storage(
    df: pd.DataFrame,
    label_to_id: dict[str, str],
    multiselect_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    storage = df.copy(deep=True)

    columns_to_process = set(multiselect_columns or [])
    for column in columns_to_process:
        if column in storage.columns and column != "categories":
            storage[column] = storage[column].apply(_split_multiselect_input)

    if "categories" in storage.columns:
        options_map: dict[str, int] = {}
        for key, value in (label_to_id or {}).items():
            try:
                options_map[key] = int(value)
                options_map[key.casefold()] = int(value)
            except (AttributeError, TypeError, ValueError):
                continue

        storage["categories"] = storage["categories"].apply(
            lambda labels: _labels_to_ids(
                list(labels) if isinstance(labels, (list, tuple, set)) else _split_multiselect_input(labels),
                options_map,
            )
        )
    return storage


def _value_for_editor(attr_row: dict, meta: dict):
    frontend_input = (meta.get("frontend_input") or "").lower()
    raw = attr_row.get("raw_value") if attr_row else None
    label = attr_row.get("label") if attr_row else None
    value_to_label = meta.get("values_to_labels") or {}

    def _lookup(value):
        if value is None:
            return None
        candidates = []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            candidates.append(stripped)
            candidates.append(stripped.lower())
            try:
                candidates.append(int(stripped))
            except (TypeError, ValueError):
                pass
        else:
            candidates.append(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidates.append(int(value))
                candidates.append(str(int(value)))
        for candidate in candidates:
            if candidate in value_to_label:
                mapped = value_to_label[candidate]
                if isinstance(mapped, str):
                    return mapped
                return str(mapped)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    if frontend_input == "boolean":
        if isinstance(raw, bool):
            return raw
        if raw in (None, ""):
            return False
        raw_str = str(raw).strip().lower()
        return raw_str in {"1", "true", "yes", "y", "on"}

    if frontend_input == "multiselect":
        selected = []
        values: list[str] = []
        if isinstance(raw, (list, tuple, set)):
            values = [str(item) for item in raw]
        elif raw not in (None, ""):
            values = [item.strip() for item in str(raw).split(",")]
        elif label:
            values = [item.strip() for item in str(label).split(",")]

        for value in values:
            if not value:
                continue
            mapped = _lookup(value)
            if mapped is None:
                mapped = str(value)
            selected.append(mapped)
        return selected

    if frontend_input == "select":
        mapped = _lookup(raw)
        if mapped:
            return mapped
        if label:
            return str(label)
        if raw not in (None, ""):
            return str(raw)
        return ""

    value = label if label not in (None, "") else raw
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_editor_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _sanitize_for_storage(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set)):
        sanitized = [str(item).strip() for item in value if str(item).strip()]
        return sanitized
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _build_column_config(column_order, column_meta):
    config = {}
    for column in column_order:
        if column == "sku":
            config[column] = st.column_config.TextColumn("SKU", disabled=True)
            continue
        if column == "name":
            config[column] = st.column_config.TextColumn("Name", disabled=True)
            continue
        if column == "attribute set":
            config[column] = st.column_config.TextColumn(
                "Attribute Set", disabled=True
            )
            continue
        meta = column_meta.get(column, {})
        label = _attr_label(meta, column)
        frontend_input = (meta.get("frontend_input") or "").lower()
        options = _meta_options(meta)
        if frontend_input == "boolean":
            config[column] = st.column_config.CheckboxColumn(label)
        elif frontend_input == "select" and options:
            config[column] = st.column_config.SelectboxColumn(
                label=label,
                options=options,
                required=False,
            )
        elif frontend_input == "multiselect":
            help_parts = ["Comma-separated values"]
            if options:
                preview = ", ".join(options[:5])
                help_parts.append(f"Options include: {preview}")
            config[column] = st.column_config.Column(
                label=label,
                help=". ".join(help_parts),
            )
        else:
            config[column] = st.column_config.TextColumn(label)
    return config


def _build_column_config_for_step1_like(step: str) -> tuple[dict, list[str]]:
    if step == "step1":
        config = st.session_state.get("step1_column_config_cache", {})
        disabled = st.session_state.get("step1_disabled_cols_cache", [])
        return config, disabled
    if step == "step2":
        return {}, []
    return {}, []


def _reset_step2_state():
    step2_state = st.session_state.get("step2")
    if step2_state is None:
        return
    if not isinstance(step2_state, dict):
        st.session_state["step2"] = {}
        return

    for key in (
        "wide",
        "wide_orig",
        "wide_synced",
        "dfs",
        "original",
        "staged",
        "wide_meta",
        "current_set_id",
    ):
        step2_state.pop(key, None)


def _ensure_step2_state() -> dict:
    state = st.session_state.setdefault("step2", {})
    state.setdefault("dfs", {})
    state.setdefault("original", {})
    state.setdefault("column_config", {})
    state.setdefault("disabled", {})
    state.setdefault("row_meta", {})
    state.setdefault("set_names", {})
    state.setdefault("wide", {})
    state.setdefault("wide_orig", {})
    state.setdefault("wide_meta", {})
    state.setdefault("wide_colcfg", {})
    state.setdefault("wide_synced", {})
    state.setdefault("staged", {})
    return state


def _build_value_column_config(row_meta_map: dict[str, dict]):
    if not row_meta_map:
        return st.column_config.Column("Value")

    frontend_inputs = {
        (meta.get("frontend_input") or meta.get("backend_type") or "text").lower()
        for meta in row_meta_map.values()
    }

    frontend_inputs.discard("")

    def _options_union() -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()
        for meta in row_meta_map.values():
            for option in meta.get("options", []) or []:
                if isinstance(option, str):
                    label = option.strip()
                else:
                    label = ""
                if not label or label in seen:
                    continue
                seen.add(label)
                collected.append(label)
        return collected

    if len(frontend_inputs) == 1:
        input_type = frontend_inputs.pop()
        options = _options_union()
        if input_type == "select":
            return st.column_config.SelectboxColumn("Value", options=options)
        if input_type == "multiselect":
            return st.column_config.MultiselectColumn("Value", options=options)
        if input_type == "boolean":
            return st.column_config.CheckboxColumn("Value")
        if input_type in {"int", "integer", "decimal", "price", "float"}:
            return st.column_config.NumberColumn("Value")
        return st.column_config.TextColumn("Value")

    return st.column_config.Column("Value")


def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert attribute rows to a wide table with one row per SKU."""

    if df_long.empty:
        return pd.DataFrame(columns=["sku", "name"]).astype(object)

    base = df_long[["sku", "name"]].drop_duplicates().set_index("sku")
    pv = df_long.pivot_table(
        index="sku",
        columns="attribute_code",
        values="value",
        aggfunc=lambda x: x.iloc[-1],
    )
    out = base.join(pv, how="left").reset_index()
    for column in out.columns:
        out[column] = out[column].astype(object)
    return out


def apply_wide_edits_to_long(
    df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> pd.DataFrame:
    """Propagate edits from a wide table back to the long representation."""

    if not isinstance(df_long, pd.DataFrame) or df_long.empty:
        return df_long.copy(deep=True) if isinstance(df_long, pd.DataFrame) else df_long

    if not isinstance(df_wide, pd.DataFrame) or df_wide.empty:
        return df_long.copy(deep=True)

    if "sku" not in df_wide.columns:
        return df_long.copy(deep=True)

    wide_indexed = df_wide.set_index("sku")
    updated = df_long.copy(deep=True)

    for row_idx, row in updated.iterrows():
        sku = row.get("sku")
        code = row.get("attribute_code")
        if not sku or not code:
            continue
        if sku not in wide_indexed.index:
            continue
        if code not in wide_indexed.columns:
            continue
        updated.at[row_idx, "value"] = wide_indexed.at[sku, code]

    return updated


def _ensure_wide_meta_options(
    meta_map: dict[str, dict], sample_df: pd.DataFrame | None
) -> None:
    if not isinstance(meta_map, dict):
        return
    for meta in meta_map.values():
        if not isinstance(meta, dict):
            continue
        options = meta.get("options")
        if isinstance(options, list):
            continue
        meta["options"] = []


def _refresh_wide_meta_from_cache(
    meta_cache: AttributeMetaCache | None,
    meta_map: dict[str, dict],
    attr_codes: list[str],
    wide_df: pd.DataFrame | None,
) -> None:
    if not isinstance(meta_map, dict) or not attr_codes:
        return

    if not isinstance(meta_cache, AttributeMetaCache):
        return

    unique_codes: list[str] = []
    for raw in attr_codes:
        if not isinstance(raw, str):
            continue
        code = raw.strip()
        if not code or code in unique_codes:
            continue
        unique_codes.append(code)

    target_codes = [code for code in unique_codes if code != "categories"]
    if target_codes:
        meta_cache.build_and_set_static_for(target_codes, store_id=0)

    for code in unique_codes:
        cached_meta = meta_cache.get(code)
        if isinstance(cached_meta, dict):
            meta_map[code] = cached_meta


def _coerce_for_ui(df: pd.DataFrame, meta_map: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for code, meta in (meta_map or {}).items():
        t = (meta.get("frontend_input") or meta.get("backend_type") or "text").lower()
        if code not in out.columns:
            continue
        if t == "boolean":
            out[code] = out[code].apply(
                lambda v: str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
                if v not in (None, "")
                else False
            )
        elif t == "multiselect":
            if code == "categories":
                values_to_labels = meta.get("values_to_labels") or {}
                raw_map = meta.get("options_map") or {}
                options_map: dict[str, int] = {}
                if isinstance(raw_map, dict):
                    for key, value in raw_map.items():
                        if not isinstance(key, str):
                            continue
                        options_map[key] = value
                        options_map[key.casefold()] = value
                        try:
                            options_map[str(int(value))] = int(value)
                        except (TypeError, ValueError):
                            continue
                out[code] = out[code].apply(
                    lambda v: _ids_to_labels(
                        _categories_value_to_ids(v, options_map),
                        values_to_labels,
                    )
                )
            else:
                out[code] = out[code].apply(
                    lambda v: v
                    if isinstance(v, (list, tuple, set))
                    else ([] if v in (None, "") else [str(v).strip()])
                )
        else:
            out[code] = out[code].astype(object)
    return out


def _to_list_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            if isinstance(item, float) and math.isnan(item):
                continue
            normalized.append(str(item))
        return normalized
    return [str(value)]


def _ensure_list_of_strings(value: object) -> list[str]:
    return _to_list_str(value)


def _ensure_config_options(
    column: str, cfg_obj, options: list[str]
):  # pragma: no cover - UI helper
    if not options:
        return cfg_obj
    if hasattr(cfg_obj, "_config") and isinstance(cfg_obj._config, dict):
        cfg_obj._config["options"] = options
        return cfg_obj
    try:
        cfg_obj.options = options
        return cfg_obj
    except Exception:
        label = (
            getattr(cfg_obj, "label", None)
            or getattr(cfg_obj, "_label", None)
            or column
        )
        if isinstance(cfg_obj, st.column_config.SelectboxColumn):
            return st.column_config.SelectboxColumn(label, options=options)
        if isinstance(cfg_obj, st.column_config.MultiselectColumn):
            return st.column_config.MultiselectColumn(label, options=options)
    return cfg_obj


def _coerce_column_for_config(
    df: pd.DataFrame, column: str, cfg_obj
) -> tuple[pd.DataFrame, object]:  # pragma: no cover - UI helper
    if isinstance(cfg_obj, st.column_config.CheckboxColumn):
        df[column] = df[column].fillna(False).astype(bool)
        return df, cfg_obj
    if isinstance(cfg_obj, st.column_config.SelectboxColumn):
        df[column] = df[column].astype("string")
        options = getattr(cfg_obj, "options", None) or []
        str_options = [str(opt) for opt in options]
        return df, _ensure_config_options(column, cfg_obj, str_options)
    if isinstance(cfg_obj, st.column_config.MultiselectColumn):
        df[column] = df[column].apply(_ensure_list_of_strings)
        options = getattr(cfg_obj, "options", None) or []
        str_options = [str(opt) for opt in options]
        return df, _ensure_config_options(column, cfg_obj, str_options)
    return df, cfg_obj


def _probe_editor_groups(
    df: pd.DataFrame,
    columns: list[str],
    column_config: dict[str, object],
    disabled: list[str],
    key_prefix: str,
):  # pragma: no cover - UI helper
    def _try(columns_chunk: list[str], suffix: str) -> bool:
        sub_df = df[columns_chunk]
        sub_cfg = {
            col: column_config[col]
            for col in columns_chunk
            if col in column_config
        }
        sub_disabled = [col for col in disabled if col in columns_chunk]
        try:
            st.data_editor(
                sub_df,
                column_config=sub_cfg,
                disabled=sub_disabled,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key=f"{key_prefix}::{suffix}",
            )
        except Exception as exc:
            st.warning(
                f"Editor failed for columns {columns_chunk}: {exc}")
            return False
        return True

    def _split(columns_chunk: list[str], suffix: str) -> set[str]:
        if not columns_chunk:
            return set()
        if _try(columns_chunk, suffix):
            return set()
        if len(columns_chunk) == 1:
            return {columns_chunk[0]}
        mid = len(columns_chunk) // 2
        left = _split(columns_chunk[:mid], suffix + "L")
        right = _split(columns_chunk[mid:], suffix + "R")
        return left | right

    return _split(columns, "root")


def _apply_categories_fallback(meta_map: dict[str, dict]) -> None:
    if not isinstance(meta_map, dict):
        return
    if not st.session_state.get("step2_categories_failed"):
        return
    meta = meta_map.get("categories")
    if isinstance(meta, dict) and meta.get("options"):
        return
    fallback = meta_map.setdefault("categories", {})
    fallback["frontend_input"] = "text"
    fallback.pop("options", None)
    fallback.pop("options_map", None)
    fallback.pop("values_to_labels", None)


def colcfg_from_meta(
    code: str, meta: dict | None, sample_df: pd.DataFrame | None = None
):
    meta = meta or {}
    column_label = _attr_label(meta, code)
    ftype_raw = meta.get("frontend_input") or meta.get("backend_type") or "text"
    ftype = str(ftype_raw).lower()
    labels = [
        opt.get("label")
        for opt in (meta.get("options") or [])
        if isinstance(opt, dict) and opt.get("label")
    ]

    if ftype == "multiselect" and labels:
        return st.column_config.MultiselectColumn(column_label, options=labels)
    if ftype == "boolean":
        return st.column_config.CheckboxColumn(column_label)
    if ftype in {"select", "dropdown"} and labels:
        return st.column_config.SelectboxColumn(column_label, options=labels)
    if ftype in {"int", "integer", "decimal", "price", "float"}:
        return st.column_config.NumberColumn(column_label)
    return st.column_config.TextColumn(column_label)


def build_wide_colcfg(
    wide_meta: dict[str, dict], sample_df: pd.DataFrame | None = None
):
    cfg = {
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "name": st.column_config.TextColumn("Name", disabled=False),
    }

    for code, original_meta in list(wide_meta.items()):
        meta = original_meta or {}
        cfg[code] = colcfg_from_meta(code, meta)

    if "guitarstylemultiplechoice" in cfg:
        guitar_cfg = cfg["guitarstylemultiplechoice"]
        if hasattr(guitar_cfg, "label"):
            guitar_cfg.label = "Guitar style"
        elif hasattr(guitar_cfg, "_label"):
            guitar_cfg._label = "Guitar style"

    return cfg


def build_attributes_df(
    df_changed: pd.DataFrame,
    session,
    api_base: str,
    attribute_sets: dict,
    attr_sets_map: dict,
    meta_cache: AttributeMetaCache,
):
    core_codes = ["brand", "condition"]
    rows_by_set: dict[int, list[dict]] = defaultdict(list)
    row_meta_by_set: dict[int, dict[str, dict]] = defaultdict(dict)
    set_names: dict[int, str] = {}

    cats_meta = meta_cache.get("categories") if isinstance(meta_cache, AttributeMetaCache) else {}
    if not isinstance(cats_meta, dict):
        cats_meta = {}
    cat_values_to_labels: dict[str, str] = cats_meta.get("values_to_labels") or {}
    raw_options_map: dict[str, int] = cats_meta.get("options_map") or {}
    cat_options_map: dict[str, int] = {}
    for key, value in raw_options_map.items():
        if not isinstance(key, str):
            continue
        cat_options_map[key] = value
        cat_options_map[key.casefold()] = value
        try:
            cat_options_map[str(int(value))] = int(value)
        except (TypeError, ValueError):
            continue
    cat_labels = [
        opt.get("label")
        for opt in (cats_meta.get("options") or [])
        if isinstance(opt, dict) and opt.get("label")
    ]

    for _, row in df_changed.iterrows():
        sku_value = str(row.get("sku", "")).strip()
        if not sku_value:
            continue

        name_value = row.get("name")
        attr_set_value = row.get("attribute set")
        try:
            with st.spinner(f"🔄 {sku_value}: loading attributes…"):
                product = get_product_by_sku(session, api_base, sku_value)
        except Exception as exc:  # pragma: no cover - UI interaction
            st.warning(f"{sku_value}: failed to fetch attributes ({exc})")
            continue

        attr_set_id = None
        if pd.notna(attr_set_value):
            attr_set_id = attribute_sets.get(attr_set_value)
        if attr_set_id is None:
            attr_set_id = product.get("attribute_set_id")

        if attr_set_id is None:
            continue

        try:
            attr_set_id_int = int(attr_set_id)
        except (TypeError, ValueError):
            continue

        set_names[attr_set_id_int] = attr_sets_map.get(attr_set_id_int, str(attr_set_id_int))

        allowed = compute_allowed_attrs(
            attr_set_id_int,
            SET_ATTRS,
            attr_sets_map or {},
            ALWAYS_ATTRS,
        )

        editor_codes: list[str] = []
        for code in core_codes:
            if code not in editor_codes:
                editor_codes.append(code)
        for code in sorted(allowed):
            if code not in editor_codes:
                editor_codes.append(code)

        df_full = collect_attributes_table(
            product,
            editor_codes,
            session,
            api_base,
        )
        attr_rows = df_full.to_dict(orient="index") if not df_full.empty else {}

        loadable_codes = [
            code
            for code in editor_codes
            if code not in {"sku", "name"}
        ]
        meta_cache.load(loadable_codes)

        base_name = (
            str(name_value).strip()
            if isinstance(name_value, str) and name_value.strip()
            else str(product.get("name", ""))
        )

        for code in editor_codes:
            if code in {"sku", "name", "attribute set"}:
                continue
            meta = meta_cache.get(code) or {}
            value = _value_for_editor(attr_rows.get(code, {}), meta)
            frontend_input = (
                meta.get("frontend_input")
                or meta.get("backend_type")
                or "text"
            ).lower()

            if frontend_input in {"int", "integer"}:
                try:
                    value = int(str(value).strip()) if str(value).strip() else None
                except (TypeError, ValueError):
                    value = str(value).strip() if isinstance(value, str) else value
            elif frontend_input in {"decimal", "price", "float"}:
                try:
                    value = float(str(value).strip()) if str(value).strip() else None
                except (TypeError, ValueError):
                    value = str(value).strip() if isinstance(value, str) else value
            elif frontend_input == "boolean":
                value = bool(value)
            elif frontend_input == "multiselect":
                if not isinstance(value, list):
                    value = _split_multiselect_input(value)
            elif value is None:
                value = ""

            row_id = f"{sku_value}|{code}"
            row_meta_by_set[attr_set_id_int][row_id] = {
                "attribute_code": code,
                "frontend_input": frontend_input,
                "backend_type": (meta.get("backend_type") or "").lower(),
                "options": _meta_options(meta),
                "valid_examples": meta.get("valid_examples") or [],
            }
            rows_by_set[attr_set_id_int].append(
                {
                    "__row_id": row_id,
                    "sku": sku_value,
                    "name": base_name,
                    "attribute_code": code,
                    "value": value,
                }
            )

        category_links = (
            (product.get("extension_attributes") or {}).get("category_links") or []
        )
        categories = [
            str(link.get("category_id", "")).strip()
            for link in category_links
            if str(link.get("category_id", "")).strip()
        ]
        categories_ids = _cat_to_ids(categories)
        if not categories_ids and categories:
            categories_ids = _labels_to_ids(categories, cat_options_map)
        category_labels = _ids_to_labels(categories_ids, cat_values_to_labels)
        row_id = f"{sku_value}|categories"
        row_meta_by_set[attr_set_id_int][row_id] = {
            "attribute_code": "categories",
            "frontend_input": "multiselect",
            "backend_type": "text",
            "options": cat_labels,
            "valid_examples": category_labels[:5],
        }
        rows_by_set[attr_set_id_int].append(
            {
                "__row_id": row_id,
                "sku": sku_value,
                "name": base_name,
                "attribute_code": "categories",
                "value": category_labels,
            }
        )

    if not rows_by_set:
        return {}, {}, {}, meta_cache, {}, {}

    dfs: dict[int, pd.DataFrame] = {}
    column_configs: dict[int, dict] = {}
    disabled: dict[int, list[str]] = {}

    for set_id, rows in rows_by_set.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df.sort_values(["sku", "attribute_code"]).reset_index(drop=True)
        df = df.set_index("__row_id", drop=True)
        df = df[["sku", "name", "attribute_code", "value"]]
        dfs[set_id] = df
        column_configs[set_id] = {
            "sku": st.column_config.TextColumn("SKU", disabled=True),
            "name": st.column_config.TextColumn("Name", disabled=True),
            "attribute_code": st.column_config.TextColumn(
                "Attribute Code", disabled=True
            ),
            "value": _build_value_column_config(row_meta_by_set.get(set_id, {})),
        }
        disabled[set_id] = ["sku", "name", "attribute_code"]

    return dfs, column_configs, disabled, meta_cache, row_meta_by_set, set_names

def _is_blank_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return all(_is_blank_value(item) for item in value)
    return False


def _normalize_seq_for_compare(value: object) -> tuple:
    if not isinstance(value, (list, tuple, set)):
        return ()
    normalized: list[object] = []
    for item in value:
        if _is_blank_value(item):
            continue
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue
            normalized.append(stripped)
        else:
            normalized.append(item)
    if not normalized:
        return ()
    try:
        return tuple(sorted(normalized, key=lambda x: str(x)))
    except Exception:  # pragma: no cover - safety fallback
        return tuple(normalized)


def _values_equal(left: object, right: object) -> bool:
    if _is_blank_value(left) and _is_blank_value(right):
        return True
    if isinstance(left, float) and pd.isna(left):
        left = None
    if isinstance(right, float) and pd.isna(right):
        right = None
    if isinstance(left, str):
        left = left.strip()
    if isinstance(right, str):
        right = right.strip()
    if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
        return _normalize_seq_for_compare(left) == _normalize_seq_for_compare(right)
    return left == right


def _df_differs(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        return isinstance(a, pd.DataFrame) != isinstance(b, pd.DataFrame)

    a2 = a.copy(deep=True)
    b2 = b.copy(deep=True)
    cols = list(dict.fromkeys(list(a2.columns) + list(b2.columns)))
    a2 = a2.reindex(columns=cols).astype(object)
    b2 = b2.reindex(columns=cols).astype(object)

    if len(a2) != len(b2):
        return True

    for col in cols:
        la = list(a2[col]) if col in a2 else [None] * len(a2)
        lb = list(b2[col]) if col in b2 else [None] * len(b2)
        for left, right in zip(la, lb):
            if not _values_equal(left, right):
                return True

    return False


def apply_product_update(session, api_base: str, sku: str, attributes: dict):
    if not attributes:
        return

    attrs = dict(attributes or {})

    attr_set_id = attrs.pop("attribute_set_id", None)
    if isinstance(attr_set_id, float) and pd.isna(attr_set_id):
        attr_set_id = None
    if isinstance(attr_set_id, str):
        cleaned = attr_set_id.strip()
        attr_set_id = cleaned or None

    payload: dict[str, object] = {"sku": sku}
    if attr_set_id is not None:
        try:
            payload["attribute_set_id"] = int(attr_set_id)
        except (TypeError, ValueError):
            payload["attribute_set_id"] = attr_set_id

    custom_attributes = []
    extension_attributes: dict[str, object] = {}

    for code, value in attrs.items():
        if code == "categories":
            category_links = [
                {"position": idx, "category_id": str(cat_id)}
                for idx, cat_id in enumerate(value or [])
            ]
            extension_attributes["category_links"] = category_links
        else:
            custom_attributes.append({"attribute_code": code, "value": value})

    if custom_attributes:
        payload["custom_attributes"] = custom_attributes
    if extension_attributes:
        payload["extension_attributes"] = extension_attributes

    url = f"{api_base.rstrip('/')}/products/{quote(sku, safe='')}"
    resp = session.put(
        url,
        json={"product": payload},
        headers=build_magento_headers(session=session),
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Magento update failed for {sku}: {resp.status_code} {resp.text[:200]}"
        )


def save_step2_to_magento():
    step2_state = st.session_state.get("step2")
    if not isinstance(step2_state, dict):
        st.info("No changes to save.")
        return

    session = st.session_state.get("mg_session")
    base_url = st.session_state.get("mg_base_url")
    if not session or not base_url:
        st.error("Magento session is not initialized")
        return

    staged_map = (step2_state.get("staged") or {}).copy()
    if staged_map:
        for set_id, draft in staged_map.items():
            if not isinstance(draft, pd.DataFrame):
                continue
            draft_wide = draft.copy(deep=True)
            step2_state["wide"][set_id] = draft_wide
            step2_state["wide_orig"][set_id] = draft_wide.copy(deep=True)
            existing_long = step2_state["dfs"].get(set_id)
            updated_long = apply_wide_edits_to_long(existing_long, draft_wide)
            if isinstance(updated_long, pd.DataFrame):
                if "value" in updated_long.columns:
                    updated_long["value"] = updated_long["value"].astype(object)
                step2_state["dfs"][set_id] = updated_long
                step2_state["original"][set_id] = updated_long.copy(deep=True)
            else:
                step2_state["dfs"][set_id] = draft_wide.copy(deep=True)
                step2_state["original"][set_id] = draft_wide.copy(deep=True)
        step2_state["staged"].clear()

    wide_map: dict[int, pd.DataFrame] = step2_state.get("wide", {})
    wide_meta_map: dict[int, dict[str, dict]] = step2_state.get("wide_meta", {})
    baseline_map: dict[int, pd.DataFrame] = step2_state.get("wide_synced", {})
    meta_cache = step2_state.get("meta_cache")
    if meta_cache is None:
        meta_cache = step2_state.get("meta")

    if isinstance(meta_cache, AttributeMetaCache):
        meta_cache.build_and_set_static_for(["country_of_manufacture"], store_id=0)

    if not wide_map:
        st.info("No changes to save.")
        return

    payload_by_sku: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []

    for set_id, wide_df in wide_map.items():
        if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
            continue

        attr_cols = [
            col
            for col in wide_df.columns
            if col not in ("sku", "name", "attribute_set_id")
        ]
        if not attr_cols:
            continue

        meta_for_set = wide_meta_map.get(set_id, {})
        baseline_df = baseline_map.get(set_id)
        baseline_idx = (
            baseline_df.set_index("sku")
            if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty
            else None
        )
        current_idx = wide_df.set_index("sku")

        for sku, row in current_idx.iterrows():
            sku_value = str(sku).strip() if isinstance(sku, str) else str(sku)
            if not sku_value:
                continue

            attr_set_value = row.get("attribute_set_id", set_id)
            if isinstance(attr_set_value, float) and pd.isna(attr_set_value):
                attr_set_value = set_id
            if attr_set_value in (None, ""):
                attr_set_value = set_id
            try:
                attr_set_clean = int(attr_set_value)
            except (TypeError, ValueError):
                attr_set_clean = attr_set_value
            payload_by_sku.setdefault(sku_value, {})["attribute_set_id"] = (
                attr_set_clean
            )

            baseline_row: pd.Series | None
            if baseline_idx is not None and sku in baseline_idx.index:
                baseline_row = baseline_idx.loc[sku]
                if isinstance(baseline_row, pd.DataFrame):
                    baseline_row = baseline_row.iloc[-1]
            else:
                baseline_row = None

            for code in attr_cols:
                new_raw = row.get(code)
                old_raw = (
                    baseline_row.get(code)
                    if isinstance(baseline_row, pd.Series) and code in baseline_row.index
                    else None
                )

                if _values_equal(new_raw, old_raw):
                    continue

                meta = meta_for_set.get(code) or {}
                attr_code = meta.get("attribute_code") or code

                try:
                    normalized = normalize_for_magento(attr_code, new_raw, meta_cache)
                except Exception as exc:  # pragma: no cover - safety guard
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": repr(new_raw),
                            "hint_examples": f"normalize error: {exc}",
                            "expected_codes": "",
                        }
                    )
                    continue

                if normalized is None:
                    if _is_blank_value(new_raw):
                        continue
                    meta_info: dict[str, object] = {}
                    if isinstance(meta_cache, AttributeMetaCache):
                        meta_info = meta_cache.get(attr_code) or meta
                    elif isinstance(meta_for_set, dict):
                        meta_info = meta or {}
                    examples = list(meta_info.get("valid_examples") or [])
                    if not examples:
                        options = _meta_options(meta_info)
                        for label in options:
                            if label and label not in examples:
                                examples.append(label)
                            if len(examples) >= 5:
                                break
                    expected_codes: list[str] = []
                    values_to_labels = meta_info.get("values_to_labels")
                    if isinstance(values_to_labels, dict):
                        expected_codes = sorted(
                            [
                                str(value)
                                for value in values_to_labels.values()
                                if value not in (None, "")
                            ]
                        )[:10]
                    errors.append(
                        {
                            "sku": sku_value,
                            "attribute": attr_code,
                            "raw": repr(new_raw),
                            "hint_examples": "\n".join(examples[:5]),
                            "expected_codes": ", ".join(expected_codes),
                        }
                    )
                    continue

                payload_by_sku.setdefault(sku_value, {})[attr_code] = normalized

            entry = payload_by_sku.get(sku_value)
            if entry and set(entry.keys()) == {"attribute_set_id"}:
                payload_by_sku.pop(sku_value, None)

    if not payload_by_sku and not errors:
        st.info("No changes to save.")
        return

    api_base = st.session_state.get("ai_api_base")
    if not api_base:
        st.warning("Magento API is not initialized.")
        return

    ok_skus: set[str] = set()
    for sku, attrs in payload_by_sku.items():
        try:
            apply_product_update(session, api_base, sku, attrs)
            ok_skus.add(sku)
        except Exception as exc:  # pragma: no cover - network interaction
            errors.append(
                {
                    "sku": sku,
                    "attribute": "*batch*",
                    "raw": repr(attrs),
                    "hint_examples": str(exc),
                    "expected_codes": "",
                }
            )

    if ok_skus:
        st.success("Updated SKUs:")
        st.markdown("\n".join(f"- `{sku}`" for sku in sorted(ok_skus)))

    if errors:
        import pandas as _pd

        st.warning("Some rows failed; review and fix:")
        st.dataframe(
            _pd.DataFrame(
                errors,
                columns=[
                    "sku",
                    "attribute",
                    "raw",
                    "hint_examples",
                    "expected_codes",
                ],
            ),
            use_container_width=True,
        )

    if ok_skus:
        for set_id, wide_df in wide_map.items():
            if not isinstance(wide_df, pd.DataFrame) or wide_df.empty:
                continue
            current_idx = wide_df.set_index("sku")
            baseline_df = baseline_map.get(set_id)
            if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
                baseline_idx = baseline_df.set_index("sku")
            else:
                baseline_idx = pd.DataFrame(columns=current_idx.columns)
                baseline_idx.index.name = "sku"
            updated_idx = baseline_idx.copy()
            changed = False
            for sku in ok_skus:
                if sku not in current_idx.index:
                    continue
                row = current_idx.loc[sku]
                updated_idx.loc[sku, :] = row
                changed = True
            if changed:
                result_df = updated_idx.reset_index()
                for column in result_df.columns:
                    result_df[column] = result_df[column].astype(object)
                step2_state["wide_synced"][set_id] = result_df


def load_items(
    session,
    base_url,
    *,
    attr_set_id: int | None = None,
    products: list[dict] | None = None,
):
    if attr_set_id is None:
        attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)

    rows = []
    prog = st.progress(0.0, text="Loading products…")
    total_hint = 0

    if products is None:
        iterator = iter_products_by_attr_set(session, base_url, attr_set_id)
        for product, total in iterator:
            total_hint = total or total_hint
            status = int(
                _get_custom_attr_value(product, "status", product.get("status", 1))
                or 1
            )
            visibility = int(
                _get_custom_attr_value(
                    product, "visibility", product.get("visibility", 4)
                )
                or 4
            )
            type_id = product.get("type_id", "simple")
            if visibility == 1 or type_id not in _ALLOWED_TYPES:
                continue

            rows.append(
                {
                    "sku": product.get("sku", ""),
                    "name": product.get("name", ""),
                    "created_at": product.get("created_at", ""),
                    "status": status,
                }
            )

            if total_hint:
                done = len(rows) / max(total_hint, 1)
                done = min(done, 1.0)
                prog.progress(done, text=f"Loading products… {int(done * 100)}%")
    else:
        total_hint = len(products)
        for idx, product in enumerate(products, start=1):
            product_attr_id = product.get("attribute_set_id")
            if product_attr_id is not None and attr_set_id is not None:
                try:
                    attr_value = int(product_attr_id)
                except (TypeError, ValueError):
                    attr_value = None
                if attr_value is not None and attr_value != int(attr_set_id):
                    continue

            status = int(
                _get_custom_attr_value(product, "status", product.get("status", 1))
                or 1
            )
            visibility = int(
                _get_custom_attr_value(
                    product, "visibility", product.get("visibility", 4)
                )
                or 4
            )
            type_id = product.get("type_id", "simple")
            if visibility == 1 or type_id not in _ALLOWED_TYPES:
                continue

            rows.append(
                {
                    "sku": product.get("sku", ""),
                    "name": product.get("name", ""),
                    "created_at": product.get("created_at", ""),
                    "status": status,
                }
            )

            if total_hint:
                done = idx / max(total_hint, 1)
                done = min(done, 1.0)
                prog.progress(done, text=f"Loading products… {int(done * 100)}%")

    prog.progress(1.0, text="Products loaded")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

    if DEBUG and "status" in df.columns:
        st.write("STATUS COUNTS:", df["status"].value_counts(dropna=False))

    st.info(
        f"STEP A — products in 'Default' after status/visibility/type: {len(df)}"
    )

    source_code = detect_default_source_code(session, base_url)
    prog2 = st.progress(0.0, text=f"Fetching MSI ({source_code})…")
    src_items = get_source_items(session, base_url, source_code=source_code)
    prog2.progress(1.0, text=f"MSI fetched ({source_code}): {len(src_items)} rows")
    st.info(f"STEP B — MSI source_items rows: {len(src_items)}")
    pos_qty = sum(1 for s in src_items if float(s.get("quantity", 0)) > 0)
    st.info(f"STEP B+ — MSI items with qty>0: {pos_qty}")

    qty_map = {item.get("sku"): float(item.get("quantity", 0)) for item in src_items}
    df["qty"] = df["sku"].map(qty_map).fillna(0.0)
    have_qty = int((df["qty"] > 0).sum())
    st.info(f"STEP C — products in set having qty>0: {have_qty}")

    df_pos = df[df["qty"] > 0].copy()
    df_pos["backorders"] = 0

    zero_qty_skus = df.loc[df["qty"] <= 0, "sku"].tolist()
    st.info(f"STEP D — zero-qty skus to check backorders: {len(zero_qty_skus)}")
    total_backorder_tasks = len(zero_qty_skus)
    prog3 = st.progress(0.0, text=f"Checking backorders… 0/{total_backorder_tasks}")

    if total_backorder_tasks:
        def _progress_cb(completed: int):
            if completed % 50 == 0 or completed == total_backorder_tasks:
                ratio = completed / max(total_backorder_tasks, 1)
                prog3.progress(
                    ratio,
                    text=f"Checking backorders… {completed}/{total_backorder_tasks}",
                )

        backorders_map = get_backorders_parallel(
            session,
            base_url,
            zero_qty_skus,
            progress_cb=_progress_cb,
        )
    else:
        backorders_map = {}

    prog3.progress(1.0, text="Backorders complete")

    df_zero = df[df["qty"] <= 0].copy()
    df_zero["backorders"] = df_zero["sku"].map(backorders_map).fillna(0).astype(int)
    df_bo2 = df_zero[df_zero["backorders"] == 2]
    st.info(f"STEP E — backorders==2 count: {len(df_bo2)}")

    out = pd.concat([df_pos, df_bo2], ignore_index=True)
    st.success(f"STEP F — final rows (qty>0 OR bo=2): {len(out)}")

    if out.empty:
        return pd.DataFrame(columns=["sku", "name", "attribute set", "created_at"])

    out["attribute set"] = _DEF_ATTR_SET_NAME
    return out[["sku", "name", "attribute set", "created_at"]]


magento_base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
magento_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
magento_session = get_session(auth_token=magento_token)
st.session_state["mg_session"] = magento_session
st.session_state["mg_base_url"] = magento_base_url

if not st.session_state.get("_mg_auth_logged"):
    st.write(
        "Auth header:",
        bool(st.session_state["mg_session"].headers.get("Authorization")),
    )
    st.session_state["_mg_auth_logged"] = True

session = st.session_state.get("mg_session")
base_url = st.session_state.get("mg_base_url")
if not session or not base_url:
    st.error("Magento session is not initialized")
    st.stop()

st.info("Let’s find items need to be updated.")

if "step1_editor_mode" not in st.session_state:
    st.session_state["step1_editor_mode"] = "all"

c1, c2 = st.columns(2)
btn_all = c1.button("📦 Load All", key="btn_load_all")
btn_50 = c2.button("⚡ Load 50 (fast)", key="btn_load_50_fast")

if btn_all:
    requested_run_mode: str | None = "all"
elif btn_50:
    requested_run_mode = "fast50"
else:
    requested_run_mode = None

if requested_run_mode:
    run_mode = requested_run_mode
    other_mode = "fast50" if run_mode == "all" else "all"
    limit = 50 if run_mode == "fast50" else None
    enabled_only = True if run_mode == "fast50" else None
    minimal_fields = run_mode == "fast50"
    st.session_state["step1_editor_mode"] = run_mode
    st.session_state["show_attributes_trigger"] = False

    for prefix in ("default_products", "df_original", "df_edited", "attribute_sets"):
        st.session_state.pop(f"{prefix}_{other_mode}", None)

    cache_key = f"default_products_{run_mode}"
    df_original_key = f"df_original_{run_mode}"
    df_edited_key = f"df_edited_{run_mode}"
    attribute_sets_key = f"attribute_sets_{run_mode}"

    status = st.status("Loading default products…", expanded=True)
    pbar = st.progress(0)
    data: list[dict] | None = None
    try:
        pbar.progress(5)
        status.update(label="Preparing request…")
        attr_set_id = get_attr_set_id(session, base_url, name=_DEF_ATTR_SET_NAME)
        data = get_default_products(
            session,
            base_url,
            attr_set_id=attr_set_id,
            qty_min=0,
            limit=limit,
            minimal_fields=minimal_fields,
            enabled_only=enabled_only,
        )
        st.session_state[cache_key] = data
        pbar.progress(60)
        status.update(label="Parsing response…")
        df_items = load_items(
            session,
            base_url,
            attr_set_id=attr_set_id,
            products=data,
        )
        pbar.progress(90)
        status.update(label="Building table…")
        if df_items.empty:
            st.warning("No items match the Default set filter criteria.")
            st.session_state.pop(df_original_key, None)
            st.session_state.pop(df_edited_key, None)
            st.session_state.pop(attribute_sets_key, None)
            _reset_step2_state()
        else:
            df_ui = df_items.copy()
            df_ui["created_at"] = pd.to_datetime(
                df_ui["created_at"], errors="coerce"
            )
            df_ui = df_ui.sort_values("created_at", ascending=False).reset_index(drop=True)
            st.session_state[df_original_key] = df_ui.copy()
            try:
                attribute_sets = list_attribute_sets(session, base_url)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Failed to fetch attribute sets: {exc}")
                attribute_sets = {}
            st.session_state[attribute_sets_key] = attribute_sets
            df_for_edit = df_ui.copy()
            if "hint" not in df_for_edit.columns:
                df_for_edit["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            column_order = [
                col for col in cols_order if col in df_for_edit.columns
            ]
            lead_cols = [c for c in ("sku", "name") if c in column_order]
            tail_cols = [c for c in column_order if c not in ("sku", "name")]
            column_order = lead_cols + tail_cols
            df_for_edit = df_for_edit[column_order]
            st.session_state[df_edited_key] = df_for_edit.copy()
            _reset_step2_state()
        pbar.progress(100)
        status.update(
            state="complete",
            label=f"Loaded {len(data or [])} items",
        )
    except Exception as exc:  # pragma: no cover - UI error handling
        status.update(state="error", label="Failed to load products")
        st.error(f"Error: {exc}")

current_run_mode = st.session_state.get("step1_editor_mode", "all")
df_original_key = f"df_original_{current_run_mode}"
df_edited_key = f"df_edited_{current_run_mode}"
attribute_sets_key = f"attribute_sets_{current_run_mode}"
editor_key = f"editor_step1_{current_run_mode}"

if df_original_key in st.session_state:
    df_ui = st.session_state[df_original_key]
    attribute_sets = st.session_state.get(attribute_sets_key, {})

    if df_ui.empty:
        st.warning("No items match the Default set filter criteria.")
    elif not attribute_sets:
        st.warning("Unable to load attribute sets for editing.")
        st.dataframe(df_ui, use_container_width=True)
    else:
        if df_edited_key not in st.session_state:
            df_init = df_ui.copy()
            if "hint" not in df_init.columns:
                df_init["hint"] = ""
            cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
            column_order = [col for col in cols_order if col in df_init.columns]
            lead_cols = [c for c in ("sku", "name") if c in column_order]
            tail_cols = [c for c in column_order if c not in ("sku", "name")]
            column_order = lead_cols + tail_cols
            df_init = df_init[column_order]
            st.session_state[df_edited_key] = df_init.copy()

        df_base = st.session_state[df_edited_key].copy()

        if "hint" not in df_base.columns:
            df_base["hint"] = ""

        cols_order = ["sku", "name", "attribute set", "hint", "created_at"]
        column_order = [col for col in cols_order if col in df_base.columns]
        lead_cols = [c for c in ("sku", "name") if c in column_order]
        tail_cols = [c for c in column_order if c not in ("sku", "name")]
        column_order = lead_cols + tail_cols
        df_base = df_base[column_order]
        options = list(attribute_sets.keys())
        options.extend(
            name
            for name in df_base["attribute set"].dropna().unique()
            if name not in attribute_sets
        )
        options = list(dict.fromkeys(options))


        st.header("Assign the product type (attribute set).")
        st.caption(
            "For each item, choose which product type (attribute set) it belongs to. "
            "This lets us show the correct attributes later. "
            "Edit the **Product Type (attribute set)** column: replace **default** with the right value from the dropdown."
        )

        step2_active = st.session_state.get("show_attributes_trigger", False)

        if not step2_active:
            step1_column_config = {
                "sku": st.column_config.Column(
                    label="SKU", disabled=True, width="small"
                ),
                "name": st.column_config.TextColumn(
                    label="Name", disabled=False, width="medium"
                ),
                "attribute set": st.column_config.SelectboxColumn(
                    label="Product Type (attribute set)",
                    help="Change attribute set",
                    options=options,
                    required=True,
                ),
                "hint": st.column_config.TextColumn("Hint"),
                "created_at": st.column_config.DatetimeColumn("Created At", disabled=True),
            }
            st.session_state["step1_column_config_cache"] = step1_column_config
            st.session_state["step1_disabled_cols_cache"] = [
                "sku",
                "created_at",
            ]
            col_cfg, disabled_cols = _build_column_config_for_step1_like(step="step1")
            if "sku" in df_base.columns:
                col_cfg["sku"] = st.column_config.Column(
                    label="SKU", disabled=True, width="small"
                )
            if "name" in df_base.columns:
                col_cfg["name"] = st.column_config.TextColumn(
                    label="Name", disabled=False, width="medium"
                )
            disabled_cols = [col for col in disabled_cols if col != "name"]
            if "attribute set" in col_cfg:
                cfg = col_cfg["attribute set"]
                if hasattr(cfg, "label"):
                    cfg.label = "Product Type (attribute set)"
                elif hasattr(cfg, "_label"):
                    cfg._label = "Product Type (attribute set)"
            if "attribute_set" in col_cfg:
                cfg = col_cfg["attribute_set"]
                if hasattr(cfg, "label"):
                    cfg.label = "Product Type (attribute set)"
                elif hasattr(cfg, "_label"):
                    cfg._label = "Product Type (attribute set)"
            if "attribute_set_id" in df_base.columns:
                col_cfg["attribute_set_id"] = st.column_config.NumberColumn(
                    label="Product Type (attribute set)",
                    disabled=False,
                )
            column_order = _pin_sku_name(column_order, list(df_base.columns))
            edited_df = st.data_editor(
                df_base,
                column_config=col_cfg,
                disabled=disabled_cols,
                column_order=column_order,
                use_container_width=True,
                num_rows="fixed",
                key=editor_key,
            )

            go_attrs = st.button(
                "🔎 Show attributes",
                key="btn_show_attrs",
                help="Build attribute editor for selected items",
            )
            if isinstance(edited_df, pd.DataFrame) and go_attrs:
                st.session_state[df_edited_key] = edited_df.copy()
                st.session_state["show_attributes_trigger"] = True
                st.session_state["_go_step2_requested"] = True
                _reset_step2_state()
                st.rerun()
        else:
            st.header("Filling in attributes.")
            if st.session_state.get("_go_step2_requested"):
                st.session_state["_go_step2_requested"] = False

            if df_edited_key not in st.session_state or df_original_key not in st.session_state:
                st.warning("No updated products.")
                st.session_state["show_attributes_trigger"] = False
            else:
                df_new = st.session_state[df_edited_key].copy()
                df_old = st.session_state[df_original_key].copy()

                required_cols = {"sku", "attribute set"}
                if not (
                    required_cols.issubset(df_new.columns)
                    and required_cols.issubset(df_old.columns)
                ):
                    st.warning("No updated products.")
                    st.session_state["show_attributes_trigger"] = False
                else:
                    df_new_idx = df_new.set_index("sku")
                    df_old_idx = df_old.set_index("sku")
                    common_skus = df_new_idx.index.intersection(df_old_idx.index)
                    if common_skus.empty:
                        st.warning("No updated products.")
                        st.session_state["show_attributes_trigger"] = False
                    else:
                        df_new_common = df_new_idx.loc[common_skus]
                        new_sets = df_new_common["attribute set"].fillna("")
                        old_sets = df_old_idx.loc[common_skus, "attribute set"].fillna("")
                        diff_mask = new_sets != old_sets
                        df_changed = df_new_common.loc[diff_mask].reset_index()

                        if df_changed.empty:
                            st.warning("No updated products.")
                            st.session_state["show_attributes_trigger"] = False
                        else:
                            api_base = st.session_state.get("ai_api_base")
                            attr_sets_map = st.session_state.get("ai_attr_sets_map")
                            setup_failed = False

                            html_error = False
                            try:
                                if not api_base:
                                    with st.spinner("🔌 Connecting to Magento…"):
                                        api_base = probe_api_base(session, base_url)
                                    st.session_state["ai_api_base"] = api_base
                                if not attr_sets_map and api_base:
                                    with st.spinner("📚 Loading attribute sets…"):
                                        attr_sets_map = get_attribute_sets_map(
                                            session, api_base
                                        )
                                    st.session_state["ai_attr_sets_map"] = attr_sets_map
                            except HTMLResponseError as exc:
                                st.error(str(exc))
                                html_error = True
                            except Exception as exc:  # pragma: no cover - UI interaction
                                st.warning(
                                    f"Failed to prepare Magento connection: {exc}"
                                )
                                setup_failed = True

                            if html_error:
                                st.session_state["show_attributes_trigger"] = False
                                st.stop()

                            if setup_failed or not api_base or not attr_sets_map:
                                st.session_state["show_attributes_trigger"] = False
                            else:
                                status2 = st.status(
                                    "Building attribute editor…", expanded=True
                                )
                                progress = st.progress(0)

                                def _pupdate(pct: int, msg: str) -> None:
                                    clamped = max(0, min(int(pct), 100))
                                    progress.progress(clamped)
                                    status2.update(label=msg)

                                _pupdate(5, "Collecting selected sets…")
                                selected_set_ids: list[int] = []
                                step2_state = _ensure_step2_state()
                                step2_state.setdefault("staged", {})

                                meta_cache: AttributeMetaCache | None = step2_state.get(
                                    "meta_cache"
                                )
                                if not isinstance(meta_cache, AttributeMetaCache):
                                    meta_cache = AttributeMetaCache(session, api_base)
                                    step2_state["meta_cache"] = meta_cache

                                if not step2_state["dfs"]:
                                    (
                                        dfs_by_set,
                                        column_configs,
                                        disabled_cols_map,
                                        meta_cache,
                                        row_meta_map,
                                        set_names,
                                    ) = build_attributes_df(
                                        df_changed,
                                        session,
                                        api_base,
                                        attribute_sets,
                                        attr_sets_map,
                                        meta_cache,
                                    )
                                    selected_set_ids = sorted(dfs_by_set.keys())
                                    total_sets = len(selected_set_ids) or 1

                                    for i, set_id in enumerate(selected_set_ids, start=1):
                                        df_set = dfs_by_set.get(set_id)
                                        if not isinstance(df_set, pd.DataFrame):
                                            continue

                                        stored_df = df_set.copy(deep=True)
                                        step2_state["dfs"][set_id] = stored_df
                                        step2_state["original"][set_id] = stored_df.copy(
                                            deep=True
                                        )
                                        step2_state["column_config"][set_id] = (
                                            column_configs.get(set_id, {})
                                        )
                                        step2_state["disabled"][set_id] = (
                                            disabled_cols_map.get(
                                                set_id,
                                                [
                                                    "sku",
                                                    "name",
                                                    "attribute_code",
                                                ],
                                            )
                                        )
                                        step2_state["row_meta"][set_id] = row_meta_map.get(
                                            set_id, {}
                                        )
                                        step2_state["set_names"][set_id] = set_names.get(
                                            set_id, str(set_id)
                                        )
                                        wide_df = make_wide(stored_df)
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        for column in wide_df.columns:
                                            wide_df[column] = wide_df[column].astype(object)
                                        step2_state["wide"][set_id] = wide_df
                                        step2_state["wide_orig"][set_id] = wide_df.copy(
                                            deep=True
                                        )
                                        step2_state["wide_synced"][set_id] = wide_df.copy(
                                            deep=True
                                        )
                                        attr_cols = [
                                            col
                                            for col in wide_df.columns
                                            if col not in ("sku", "name")
                                        ]
                                        cacheable = isinstance(
                                            meta_cache, AttributeMetaCache
                                        )
                                        if DEBUG:
                                            st.write(
                                                "DEBUG attr_cols:",
                                                sorted(attr_cols),
                                            )
                                        resolved_map: dict[str, str] = {}
                                        effective_codes: list[str] = []
                                        pct_resolve = 10 + int(
                                            20 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_resolve,
                                            f"Resolving codes for set {set_id} ({i}/{total_sets})…",
                                        )
                                        if cacheable:
                                            resolved_map = meta_cache.resolve_codes_in_set(
                                                set_id, ["brand", "condition"]
                                            )
                                            seen_effective: set[str] = set()
                                            for code in attr_cols:
                                                if not isinstance(code, str):
                                                    continue
                                                effective = resolved_map.get(code, code)
                                                if effective and effective not in seen_effective:
                                                    seen_effective.add(effective)
                                                    effective_codes.append(effective)
                                        pct_meta = 30 + int(
                                            10 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_meta,
                                            f"Fetching metadata (set {set_id})…",
                                        )
                                        if cacheable:
                                            if effective_codes:
                                                meta_cache.build_and_set_static_for(
                                                    effective_codes, store_id=0
                                                )
                                            dbg_source = getattr(
                                                meta_cache, "_debug_http", {}
                                            )
                                            dbg_snapshot = (
                                                dbg_source.copy()
                                                if isinstance(dbg_source, dict)
                                                else {}
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "SET ATTRS DEBUG:",
                                                    dbg_snapshot.get(f"set_{set_id}"),
                                                    dbg_snapshot.get(f"set_{set_id}_err"),
                                                )
                                                st.write("RESOLVED:", resolved_map)
                                        wide_meta = step2_state.setdefault("wide_meta", {})
                                        meta_map: dict[str, dict] = {}
                                        if cacheable:
                                            for code in attr_cols:
                                                actual_code = resolved_map.get(code, code)
                                                cached = meta_cache.get(actual_code) or {}
                                                if not isinstance(cached, dict):
                                                    cached = {}
                                                meta_map[code] = cached
                                        else:
                                            meta_map = {code: {} for code in attr_cols}
                                        wide_meta[set_id] = meta_map
                                        resolved_storage = step2_state.setdefault(
                                            "resolved_codes", {}
                                        )
                                        resolved_storage[set_id] = resolved_map
                                        _apply_categories_fallback(meta_map)
                                        _ensure_wide_meta_options(
                                            meta_map,
                                            step2_state["wide"].get(set_id),
                                        )
                                        meta_for_debug = wide_meta[set_id]
                                        if DEBUG:
                                            def _meta_shot(code: str) -> dict:
                                                m = meta_for_debug.get(code, {})
                                                if not isinstance(m, dict):
                                                    m = {}
                                                return {
                                                    "frontend_input": m.get("frontend_input"),
                                                    "options_count": len(m.get("options") or []),
                                                    "first_options": (m.get("options") or [])[:5],
                                                    "has_maps": bool(m.get("options_map"))
                                                    and bool(m.get("values_to_labels")),
                                                    "options_source": m.get("options_source"),
                                                }

                                            counts: dict[str, int] = {}
                                            for meta in meta_for_debug.values():
                                                if not isinstance(meta, dict):
                                                    continue
                                                t = str(meta.get("frontend_input") or "text").lower()
                                                counts[t] = counts.get(t, 0) + 1

                                            st.caption(f"DEBUG set_id={set_id}")
                                            st.write("brand:", _meta_shot("brand"))
                                            st.write("condition:", _meta_shot("condition"))
                                            st.write("DEBUG types distribution:", counts)

                                        step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                            meta_for_debug,
                                            sample_df=step2_state["wide"].get(set_id),
                                        )

                                step2_state["meta_cache"] = meta_cache

                                if not selected_set_ids:
                                    selected_set_ids = sorted(step2_state["dfs"].keys())

                                for set_id, df0 in step2_state["dfs"].items():
                                    if "value" in df0.columns:
                                        df0["value"] = df0["value"].astype(object)

                                if step2_state["dfs"] and not step2_state["wide"]:
                                    ordered_sets = (
                                        selected_set_ids
                                        or sorted(step2_state["dfs"].keys())
                                    )
                                    total_sets = len(ordered_sets) or 1
                                    for i, set_id in enumerate(ordered_sets, start=1):
                                        df_existing = step2_state["dfs"].get(set_id)
                                        if not isinstance(df_existing, pd.DataFrame):
                                            continue
                                        wide_df = make_wide(df_existing)
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        for column in wide_df.columns:
                                            wide_df[column] = wide_df[column].astype(object)
                                        step2_state["wide"][set_id] = wide_df
                                        step2_state["wide_orig"][set_id] = wide_df.copy(
                                            deep=True
                                        )
                                        if set_id not in step2_state["wide_synced"]:
                                            step2_state["wide_synced"][set_id] = wide_df.copy(
                                                deep=True
                                            )
                                        attr_cols = [
                                            col
                                            for col in wide_df.columns
                                            if col not in ("sku", "name")
                                        ]
                                        cacheable = isinstance(
                                            meta_cache, AttributeMetaCache
                                        )
                                        if DEBUG:
                                            st.write(
                                                "DEBUG attr_cols:",
                                                sorted(attr_cols),
                                            )
                                        resolved_map: dict[str, str] = {}
                                        effective_codes: list[str] = []
                                        pct_resolve = 10 + int(
                                            20 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_resolve,
                                            f"Resolving codes for set {set_id} ({i}/{total_sets})…",
                                        )
                                        if cacheable:
                                            resolved_map = meta_cache.resolve_codes_in_set(
                                                set_id, ["brand", "condition"]
                                            )
                                            seen_effective: set[str] = set()
                                            for code in attr_cols:
                                                if not isinstance(code, str):
                                                    continue
                                                effective = resolved_map.get(code, code)
                                                if effective and effective not in seen_effective:
                                                    seen_effective.add(effective)
                                                    effective_codes.append(effective)
                                        pct_meta = 30 + int(
                                            10 * i / max(total_sets, 1)
                                        )
                                        _pupdate(
                                            pct_meta,
                                            f"Fetching metadata (set {set_id})…",
                                        )
                                        if cacheable:
                                            if effective_codes:
                                                meta_cache.build_and_set_static_for(
                                                    effective_codes, store_id=0
                                                )
                                            dbg_source = getattr(
                                                meta_cache, "_debug_http", {}
                                            )
                                            dbg_snapshot = (
                                                dbg_source.copy()
                                                if isinstance(dbg_source, dict)
                                                else {}
                                            )
                                            if DEBUG:
                                                st.write(
                                                    "SET ATTRS DEBUG:",
                                                    dbg_snapshot.get(f"set_{set_id}"),
                                                    dbg_snapshot.get(f"set_{set_id}_err"),
                                                )
                                                st.write("RESOLVED:", resolved_map)
                                        wide_meta = step2_state.setdefault("wide_meta", {})
                                        meta_map: dict[str, dict] = {}
                                        if cacheable:
                                            for code in attr_cols:
                                                actual_code = resolved_map.get(code, code)
                                                cached = meta_cache.get(actual_code) or {}
                                                if not isinstance(cached, dict):
                                                    cached = {}
                                                meta_map[code] = cached
                                        else:
                                            meta_map = {code: {} for code in attr_cols}
                                        wide_meta[set_id] = meta_map
                                        resolved_storage = step2_state.setdefault(
                                            "resolved_codes", {}
                                        )
                                        resolved_storage[set_id] = resolved_map
                                        _apply_categories_fallback(meta_map)
                                        _ensure_wide_meta_options(
                                            meta_map,
                                            step2_state["wide"].get(set_id),
                                        )
                                        meta_for_debug = wide_meta[set_id]
                                        if DEBUG:
                                            def _meta_shot(code: str) -> dict:
                                                m = meta_for_debug.get(code, {})
                                                if not isinstance(m, dict):
                                                    m = {}
                                                return {
                                                    "frontend_input": m.get("frontend_input"),
                                                    "options_count": len(m.get("options") or []),
                                                    "first_options": (m.get("options") or [])[:5],
                                                    "has_maps": bool(m.get("options_map"))
                                                    and bool(m.get("values_to_labels")),
                                                    "options_source": m.get("options_source"),
                                                }

                                            counts: dict[str, int] = {}
                                            for meta in meta_for_debug.values():
                                                if not isinstance(meta, dict):
                                                    continue
                                                t = str(meta.get("frontend_input") or "text").lower()
                                                counts[t] = counts.get(t, 0) + 1

                                            st.caption(f"DEBUG set_id={set_id}")
                                            st.write("brand:", _meta_shot("brand"))
                                            st.write("condition:", _meta_shot("condition"))
                                            st.write("DEBUG types distribution:", counts)

                                        step2_state["wide_colcfg"][set_id] = build_wide_colcfg(
                                            meta_for_debug,
                                            sample_df=step2_state["wide"].get(set_id),
                                        )

                                if step2_state["wide"]:
                                    selected_set_ids = sorted(
                                        set(selected_set_ids)
                                        | set(step2_state["wide"].keys())
                                    )

                                _pupdate(40, "Fetching metadata from Magento…")

                                if (
                                    "categories_tree" not in step2_state
                                    or "categories_v2l" not in step2_state
                                ):
                                    try:
                                        categories_tree = get_categories_tree(
                                            session, api_base
                                        )
                                    except Exception:
                                        categories_tree = {}
                                    if not isinstance(categories_tree, dict):
                                        categories_tree = {}

                                    def _flatten_tree(node: Any, prefix: str = "") -> list[dict[str, str]]:
                                        out: list[dict[str, str]] = []
                                        if not isinstance(node, dict):
                                            return out
                                        label = str(node.get("name") or "").strip()
                                        cat_id = str(node.get("id") or "").strip()
                                        path = f"{prefix} / {label}".strip(" /")
                                        if cat_id and label:
                                            out.append({
                                                "id": cat_id,
                                                "label": label,
                                                "path": path,
                                            })
                                        children = node.get("children_data") or []
                                        for child in children:
                                            out.extend(_flatten_tree(child, path))
                                        return out

                                    flat: list[dict[str, str]] = []
                                    if categories_tree:
                                        flat = _flatten_tree(categories_tree)

                                    step2_state["categories_tree"] = categories_tree
                                    step2_state["categories_flat"] = flat
                                    v2l = {
                                        item["id"]: item["path"]
                                        for item in flat
                                        if item.get("id") and item.get("path")
                                    }
                                    l2v = {
                                        item["path"]: item["id"]
                                        for item in flat
                                        if item.get("id") and item.get("path")
                                    }
                                    step2_state["categories_v2l"] = v2l
                                    step2_state["categories_l2v"] = l2v

                                categories_meta = step2_state.get("categories_meta")
                                if not isinstance(categories_meta, dict) or not categories_meta.get(
                                    "options"
                                ):
                                    try:
                                        with st.spinner("📂 Loading categories…"):
                                            categories_meta = ensure_categories_meta(
                                                meta_cache,
                                                session,
                                                api_base,
                                                store_id=0,
                                            )
                                        if not isinstance(categories_meta, dict):
                                            categories_meta = (
                                                meta_cache.get("categories")
                                                if isinstance(
                                                    meta_cache, AttributeMetaCache
                                                )
                                                else {}
                                            )
                                        st.session_state["step2_categories_failed"] = False
                                    except Exception as exc:  # pragma: no cover - UI interaction
                                        st.warning(
                                            f"Failed to load categories: {exc}"
                                        )
                                        categories_meta = (
                                            meta_cache.get("categories")
                                            if isinstance(meta_cache, AttributeMetaCache)
                                        else {}
                                    )
                                    st.session_state["step2_categories_failed"] = True
                                else:
                                    st.session_state["step2_categories_failed"] = False

                                if not isinstance(categories_meta, dict):
                                    categories_meta = {}
                                step2_state["categories_meta"] = categories_meta

                                step2_state["categories_all_labels"] = [
                                    o.get("label", "")
                                    for o in (categories_meta.get("options") or [])
                                    if isinstance(o, dict)
                                ]

                                wide_meta_all = step2_state.get("wide_meta")
                                if isinstance(wide_meta_all, dict):
                                    for sid, meta_map_all in wide_meta_all.items():
                                        if not isinstance(meta_map_all, dict):
                                            continue
                                        categories_slot = meta_map_all.setdefault(
                                            "categories", {}
                                        )
                                        categories_slot["frontend_input"] = "multiselect"
                                        categories_slot["options"] = (
                                            categories_meta.get("options", []) or []
                                        )
                                        categories_slot["options_map"] = (
                                            categories_meta.get("options_map", {}) or {}
                                        )
                                        categories_slot["values_to_labels"] = (
                                            categories_meta.get("values_to_labels", {}) or {}
                                        )

                                _pupdate(45, "Preparing categories (full tree)…")

                                st.session_state["show_attrs"] = bool(
                                    step2_state["wide"]
                                )

                                completed = False
                                if not step2_state["wide"]:
                                    st.warning("No attributes to display.")
                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )
                                    completed = True
                                elif st.session_state.get("show_attrs"):
                                    wide_meta = step2_state.setdefault("wide_meta", {})
                                    categories_meta = step2_state.get("categories_meta", {})
                                    if not isinstance(categories_meta, dict):
                                        categories_meta = {}
                                    labels_all = [
                                        opt.get("label")
                                        for opt in (categories_meta.get("options") or [])
                                        if isinstance(opt, dict) and opt.get("label")
                                    ]
                                    step2_state[
                                        "categories_all_labels"
                                    ] = labels_all.copy()
                                    total_rows = sum(
                                        len(df)
                                        for df in step2_state["wide"].values()
                                        if isinstance(df, pd.DataFrame)
                                    )
                                    normalized_rows = 0
                                    if total_rows == 0:
                                        _pupdate(70, "Normalizing values…")
                                    config_stage_logged = False
                                    render_counter = 0
                                    total_sets_render = len(selected_set_ids) or 1
                                    for set_id in sorted(step2_state["wide"].keys()):
                                        wide_df = step2_state["wide"].get(set_id)
                                        if not isinstance(wide_df, pd.DataFrame):
                                            continue

                                        wide_df = wide_df.copy(deep=True)
                                        if "attribute_set_id" in wide_df.columns:
                                            wide_df["attribute_set_id"] = set_id
                                        else:
                                            wide_df.insert(1, "attribute_set_id", set_id)
                                        step2_state["wide"][set_id] = wide_df

                                        meta_map = wide_meta.get(set_id, {})
                                        if not isinstance(meta_map, dict):
                                            meta_map = {}
                                            wide_meta[set_id] = meta_map

                                        _apply_categories_fallback(meta_map)
                                        df_ref = _coerce_for_ui(wide_df, meta_map)
                                        _ensure_wide_meta_options(meta_map, df_ref)

                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "categories" in df_ref.columns
                                        ):
                                            df_ref["categories"] = df_ref["categories"].apply(
                                                lambda v: list(v)
                                                if isinstance(v, (list, tuple, set))
                                                else ([] if v in (None, "") else [str(v).strip()])
                                            )

                                        string_cols = [
                                            col
                                            for col in ("brand", "condition")
                                            if isinstance(df_ref, pd.DataFrame)
                                            and col in df_ref.columns
                                        ]
                                        if string_cols:
                                            df_ref[string_cols] = df_ref[
                                                string_cols
                                            ].astype("string")

                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "sku" in df_ref.columns
                                            and total_rows > 0
                                        ):
                                            sku_values = df_ref["sku"].tolist()
                                            for idx, sku in enumerate(sku_values, start=1):
                                                normalized_rows += 1
                                                if (
                                                    idx == len(sku_values)
                                                    or idx % 25 == 0
                                                    or normalized_rows == total_rows
                                                ):
                                                    pct = 45 + int(
                                                        25
                                                        * normalized_rows
                                                        / max(total_rows, 1)
                                                    )
                                                    _pupdate(
                                                        pct,
                                                        f"Normalizing values… SKU {sku} "
                                                        f"({normalized_rows}/{total_rows})",
                                                    )

                                        if DEBUG:
                                            st.caption(f"DEBUG set_id={set_id}")
                                            st.write(
                                                "META snapshot:",
                                                {
                                                    code: meta_map.get(code)
                                                    for code in (
                                                        "brand",
                                                        "condition",
                                                        "categories",
                                                    )
                                                },
                                            )

                                        if not config_stage_logged:
                                            _pupdate(75, "Composing column configs…")
                                            config_stage_logged = True

                                        column_config = build_wide_colcfg(
                                            meta_map, sample_df=df_ref
                                        )
                                        if (
                                            isinstance(df_ref, pd.DataFrame)
                                            and "categories" in df_ref.columns
                                        ):
                                            existing_cfg = column_config.get("categories")
                                            label = (
                                                getattr(existing_cfg, "label", None)
                                                or getattr(existing_cfg, "_label", None)
                                                or "categories"
                                            )
                                            column_config["categories"] = (
                                                st.column_config.MultiselectColumn(
                                                    label, options=labels_all
                                                )
                                            )
                                        step2_state["wide_colcfg"][set_id] = column_config

                                        if not isinstance(df_ref, pd.DataFrame):
                                            continue

                                        attribute_set_names = step2_state.get(
                                            "set_names", {}
                                        )

                                        for current_set_id, group in df_ref.groupby(
                                            "attribute_set_id"
                                        ):
                                            render_counter += 1
                                            pct_render = 80 + int(
                                                15
                                                * render_counter
                                                / max(total_sets_render, 1)
                                            )
                                            _pupdate(
                                                pct_render,
                                                f"Rendering table: set {current_set_id} "
                                                f"({render_counter}/{total_sets_render})…",
                                            )

                                            raw_title = step2_state["set_names"].get(
                                                current_set_id, str(current_set_id)
                                            )
                                            set_name = (
                                                raw_title
                                                if isinstance(raw_title, str)
                                                and raw_title.strip()
                                                else str(current_set_id)
                                            )
                                            icon = _attr_set_icon(set_name)
                                            st.subheader(
                                                f"{icon} {set_name} ({len(group)} items)"
                                            )

                                            group = group.reset_index(drop=True)
                                            if "attribute_set_id" in group.columns:
                                                group = group.drop(
                                                    columns=["attribute_set_id"]
                                                )

                                            set_name = attribute_set_names.get(
                                                current_set_id, ""
                                            )
                                            ordered_columns = build_column_order_for_set(
                                                list(group.columns), set_name
                                            )

                                            wide_meta_map = step2_state.get(
                                                "wide_meta", {}
                                            )
                                            if not isinstance(wide_meta_map, dict):
                                                wide_meta_map = {}
                                            meta_map_current = wide_meta_map.get(
                                                current_set_id, {}
                                            )
                                            if not isinstance(meta_map_current, dict):
                                                meta_map_current = {}

                                            for col in list(group.columns):
                                                meta = (meta_map_current or {}).get(
                                                    col, {}
                                                ) or {}
                                                ftype = (
                                                    meta.get("frontend_input") or ""
                                                ).lower()
                                                if ftype in ("select", "multiselect"):
                                                    _inject_empty_option(meta)
                                                if ftype == "multiselect":
                                                    l2v = {
                                                        str(k): str(v)
                                                        for k, v in (
                                                            meta.get("options_map")
                                                            or {}
                                                        ).items()
                                                    }
                                                    group[col] = (
                                                        group[col]
                                                        .apply(
                                                            lambda x: _coerce_multiselect_cell(
                                                                x, l2v
                                                            )
                                                        )
                                                        .astype(object)
                                                    )

                                            df_view = group[ordered_columns].copy()
                                            wcfg = (
                                                step2_state.get("wide_colcfg", {})
                                                .get(current_set_id, {})
                                            )
                                            if not isinstance(wcfg, dict):
                                                wcfg = {}

                                            gb = GridOptionsBuilder.from_dataframe(df_view)
                                            gb.configure_default_column(
                                                editable=True,
                                                wrapText=False,
                                                autoHeight=False,
                                            )

                                            if "sku" in df_view.columns:
                                                gb.configure_column(
                                                    "sku",
                                                    pinned="left",
                                                    editable=False,
                                                    width=120,
                                                    headerName="SKU",
                                                )

                                            if "name" in df_view.columns:
                                                gb.configure_column(
                                                    "name",
                                                    pinned="left",
                                                    editable=True,
                                                    width=280,
                                                    headerName="Name",
                                                )

                                            if "price" in df_view.columns:
                                                gb.configure_column(
                                                    "price",
                                                    editable=False,
                                                    headerName="Price",
                                                )

                                            header_overrides = {
                                                "guitarstylemultiplechoice": "Guitar style",
                                                "amp_type": "Amp type",
                                                "effect_type": "Effect type",
                                                "accessory_type": "Accessory type",
                                                "short_description": "Short Description",
                                                "cases_covers": "Cases & Covers",
                                            }

                                            cats_v2l_state = (
                                                step2_state.get("categories_v2l") or {}
                                            )
                                            cats_v2l = {
                                                str(k): str(v)
                                                for k, v in cats_v2l_state.items()
                                            }
                                            cats_tree_state = step2_state.get("categories_tree")
                                            if isinstance(cats_tree_state, dict):
                                                cats_tree_children = (
                                                    cats_tree_state.get("children_data", []) or []
                                                )
                                            elif isinstance(cats_tree_state, list):
                                                cats_tree_children = cats_tree_state
                                            else:
                                                cats_tree_children = []

                                            for col in ordered_columns:
                                                if col not in df_view.columns:
                                                    continue
                                                if col in ("sku", "name", "price"):
                                                    continue

                                                cfg = wcfg.get(col)
                                                header_name = header_overrides.get(col)
                                                if isinstance(cfg, dict):
                                                    header_name = (
                                                        header_name
                                                        or cfg.get("label")
                                                        or cfg.get("header")
                                                    )
                                                elif cfg is not None:
                                                    header_name = (
                                                        header_name
                                                        or getattr(cfg, "label", None)
                                                        or getattr(cfg, "_label", None)
                                                    )

                                                if not header_name:
                                                    header_name = col.replace("_", " ").title()

                                                column_kwargs: dict[str, Any] = {}
                                                if header_name:
                                                    column_kwargs["headerName"] = header_name

                                                meta = (meta_map_current or {}).get(col, {}) or {}
                                                ftype = (
                                                    meta.get("frontend_input") or ""
                                                ).lower()
                                                if ftype in ("select", "multiselect"):
                                                    _inject_empty_option(meta)
                                                labels = [
                                                    str(o.get("label", "")).strip()
                                                    for o in (meta.get("options") or [])
                                                    if isinstance(o, dict)
                                                    and str(o.get("label", "")).strip()
                                                ]
                                                v2l = {
                                                    str(k): str(v)
                                                    for k, v in (
                                                        meta.get("values_to_labels")
                                                        or {}
                                                    ).items()
                                                }
                                                l2v = {
                                                    str(k): str(v)
                                                    for k, v in (
                                                        meta.get("options_map") or {}
                                                    ).items()
                                                }

                                            if col == "short_description":
                                                gb.configure_column(
                                                    col,
                                                    editable=True,
                                                    cellEditor=HtmlEditor,
                                                    cellEditorPopup=True,
                                                    cellEditorPopupPosition="over",
                                                    valueFormatter=JsCode(
                                                        """
                        function(p){
                          var s = String(p.value||"");
                          var div=document.createElement('div'); div.innerHTML=s;
                          return (div.textContent||div.innerText||"").replace(/\s+/g,' ').trim();
                        }
                        """
                                                        ),
                                                        **column_kwargs,
                                                    )
                                                    continue

                                                if col == "categories":
                                                    cats_formatter = JsCode(
                                                        f"""
                        function(p){{
                          var m={json.dumps(cats_v2l)};
                          var v=p.value;
                          if(Array.isArray(v)) return v.map(x=>m[String(x)]||String(x)).join(\", \");
                          return v==null?\"\":(m[String(v)]||String(v));
                        }}
                        """
                                                    )
                                                    gb.configure_column(
                                                        col,
                                                        editable=True,
                                                        cellRenderer=TagOneLineRenderer,
                                                        cellRendererParams={"v2l": cats_v2l},
                                                        cellEditor=TreeMultiEditor,
                                                        cellEditorParams={
                                                            "tree": cats_tree_children,
                                                            "v2l": cats_v2l,
                                                        },
                                                        cellEditorPopup=True,
                                                        cellEditorPopupPosition="over",
                                                        valueFormatter=cats_formatter,
                                                        **column_kwargs,
                                                    )
                                                    continue

                                                if ftype == "boolean":
                                                    gb.configure_column(
                                                        col,
                                                        editable=True,
                                                        cellEditor="agCheckboxCellEditor",
                                                        cellRenderer="agCheckboxCellRenderer",
                                                        **column_kwargs,
                                                    )
                                                    continue

                                                if ftype == "select":
                                                    gb.configure_column(
                                                        col,
                                                        editable=True,
                                                        cellEditor=SearchableSelectEditor,
                                                        cellEditorParams={
                                                            "values": labels,
                                                            "l2v": l2v,
                                                        },
                                                        cellEditorPopup=True,
                                                        cellEditorPopupPosition="over",
                                                        valueFormatter=JsCode(
                                                            f"function(p){{var m={json.dumps(v2l)};var v=p.value;return m[String(v)]||String(v||'');}}"
                                                        ),
                                                        **column_kwargs,
                                                    )
                                                    continue

                                                if ftype == "multiselect":
                                                    gb.configure_column(
                                                        col,
                                                        editable=True,
                                                        cellRenderer=TagCellRenderer,
                                                        cellRendererParams={"v2l": v2l},
                                                        cellEditor=TagMultiEditor,
                                                        cellEditorParams={
                                                            "values": labels,
                                                            "v2l": v2l,
                                                            "l2v": l2v,
                                                        },
                                                        cellEditorPopup=True,
                                                        cellEditorPopupPosition="over",
                                                        valueFormatter=JsCode(
                                                            f"""
                      function(p){{
                        var m={json.dumps(v2l)};
                        var v=p.value;
                        if(Array.isArray(v)) return v.map(x=>m[String(x)]||String(x)).join(\", \");
                        return v==null?\"\":(m[String(v)]||String(v));
                      }}
                    """
                                                        ),
                                                        **column_kwargs,
                                                    )
                                                    continue

                                                gb.configure_column(
                                                    col,
                                                    editable=True,
                                                    **column_kwargs,
                                                )

                                            gb.configure_grid_options(
                                                domLayout="autoHeight",
                                                rowHeight=34,
                                                popupParent=JsCode("document.body"),
                                                suppressColumnMove=True,
                                                singleClickEdit=True,
                                                stopEditingWhenCellsLoseFocus=True,
                                                rowSelection="none",
                                            )

                                            if st.checkbox(
                                                f"DEBUG options {current_set_id}",
                                                key=f"dbg_{current_set_id}",
                                            ):
                                                st.write(
                                                    "meta_map keys:",
                                                    list((meta_map_current or {}).keys())[:10],
                                                )
                                                for probe in (
                                                    "brand",
                                                    "condition",
                                                    "categories",
                                                ):
                                                    if probe in (meta_map_current or {}):
                                                        st.write(
                                                            probe,
                                                            {
                                                                "type": (
                                                                    meta_map_current[probe].get(
                                                                        "frontend_input"
                                                                    )
                                                                ),
                                                                "opts": len(
                                                                    _meta_options_list(
                                                                        meta_map_current.get(
                                                                            probe, {}
                                                                        )
                                                                    )
                                                                ),
                                                                "v2l": list(
                                                                    _v2l(
                                                                        meta_map_current.get(
                                                                            probe, {}
                                                                        )
                                                                    ).items()
                                                                )[:5],
                                                            },
                                                        )

                                            grid = AgGrid(
                                                df_view,
                                                gridOptions=gb.build(),
                                                update_mode=GridUpdateMode.VALUE_CHANGED,
                                                allow_unsafe_jscode=True,
                                                fit_columns_on_grid_load=False,
                                                theme="balham",
                                                key=f"aggrid_set_{current_set_id}",
                                            )

                                            grid_data = grid.get("data", None)

                                            if grid_data is None:
                                                editor_df = df_view.copy()
                                            elif isinstance(grid_data, pd.DataFrame):
                                                editor_df = grid_data.copy()
                                            elif isinstance(grid_data, list):
                                                editor_df = pd.DataFrame(grid_data)
                                            elif isinstance(grid_data, dict):
                                                editor_df = (
                                                    pd.DataFrame([grid_data])
                                                    if grid_data
                                                    else df_view.copy()
                                                )
                                            else:
                                                editor_df = df_view.copy()

                                            editor_df = editor_df.reindex(columns=df_view.columns)

                                            for column in editor_df.columns:
                                                meta_for_column = (
                                                    meta_map_current.get(column, {})
                                                    if isinstance(meta_map_current, dict)
                                                    else {}
                                                )
                                                if not isinstance(meta_for_column, dict):
                                                    meta_for_column = {}
                                                ctype = str(
                                                    meta_for_column.get(
                                                        "frontend_input"
                                                    )
                                                    or ""
                                                ).lower()
                                                if ctype == "multiselect":
                                                    editor_df[column] = (
                                                        editor_df[column]
                                                        .apply(
                                                            lambda v: (
                                                                v
                                                                if isinstance(v, list)
                                                                else (
                                                                    []
                                                                    if v
                                                                    in (
                                                                        None,
                                                                        "",
                                                                    )
                                                                    else [
                                                                        str(v)
                                                                    ]
                                                                )
                                                            )
                                                        )
                                                        .astype(object)
                                                    )
                                                else:
                                                    editor_df[column] = editor_df[
                                                        column
                                                    ].astype(object)

                                            base_synced = (
                                                step2_state.get("wide_synced", {})
                                                .get(current_set_id)
                                            )
                                            if (
                                                isinstance(base_synced, pd.DataFrame)
                                                and "attribute_set_id"
                                                in base_synced.columns
                                            ):
                                                base_synced = base_synced.drop(
                                                    columns=["attribute_set_id"]
                                                )
                                            if _df_differs(editor_df, base_synced):
                                                step2_state["staged"][
                                                    current_set_id
                                                ] = editor_df.copy(deep=True)
                                            else:
                                                step2_state["staged"].pop(
                                                    current_set_id, None
                                                )

                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )
                                    completed = True
                                if not completed:
                                    _pupdate(100, "Attributes ready")
                                    status2.update(
                                        state="complete", label="Attributes ready"
                                    )

                                if st.session_state.get("show_attrs"):
                                    st.markdown("---")
                                    c1, c2 = st.columns([1, 1])
                                    btn_save = c1.button(
                                        "💾 Save to Magento",
                                        key="btn_step2_save_bottom",
                                    )
                                    btn_reset = c2.button(
                                        "🔄 Reset all",
                                        key="btn_step2_reset_bottom",
                                    )

                                    if btn_save:
                                        save_step2_to_magento()

                                    if btn_reset:
                                        _reset_step2_state()
                                        st.rerun()
