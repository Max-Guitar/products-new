import os, time, requests, pandas as pd, streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="Default Set In-Stock Browser", layout="wide")

# --- Secrets ---
MAGENTO_BASE_URL = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
MAGENTO_ADMIN_TOKEN = st.secrets["MAGENTO_ADMIN_TOKEN"]
API = MAGENTO_BASE_URL + "/rest/V1"

# --- Session with larger pool & retries ---
SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {MAGENTO_ADMIN_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
})
adapter = HTTPAdapter(
    pool_connections=80, pool_maxsize=80,
    max_retries=Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


def magento_get(path, params=None, timeout=60):
    r = SESSION.get(API + path, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_attr(item, code, default=None):
    for a in item.get("custom_attributes", []):
        if a.get("attribute_code") == code:
            return a.get("value")
    return default


def get_attr_set_id(name="Default"):
    data = magento_get("/products/attribute-sets/sets/list", {
        "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_name",
        "searchCriteria[filter_groups][0][filters][0][value]": name,
        "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
    })
    items = data.get("items", [])
    if not items:
        raise ValueError(f'Attribute set "{name}" not found')
    return int(items[0]["attribute_set_id"])


def iter_products_by_attr_set(attr_id, page_size=200, progress_cb=None):
    page, total = 1, None
    while True:
        params = {
            "searchCriteria[filter_groups][0][filters][0][field]": "attribute_set_id",
            "searchCriteria[filter_groups][0][filters][0][value]": attr_id,
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": page_size,
            "searchCriteria[currentPage]": page,
        }
        data = magento_get("/products", params)
        if total is None:
            total = int(data.get("total_count", 0))
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            yield it, total
        if page * page_size >= total:
            break
        page += 1
        if progress_cb:
            progress_cb(min(page * page_size / max(total, 1), 1.0))


def get_source_items(source_code="default", page_size=500, progress_cb=None):
    all_items, page, total = [], 1, None
    while True:
        params = {
            "searchCriteria[filter_groups][0][filters][0][field]": "source_code",
            "searchCriteria[filter_groups][0][filters][0][value]": source_code,
            "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
            "searchCriteria[pageSize]": page_size,
            "searchCriteria[currentPage]": page,
        }
        data = magento_get("/inventory/source-items", params)
        if total is None:
            total = int(data.get("total_count", 0))
        items = data.get("items", [])
        if not items:
            break
        all_items.extend(items)
        if page * page_size >= total:
            break
        page += 1
        if progress_cb and total:
            progress_cb(min(page * page_size / total, 1.0))
    return all_items


def get_backorders(sku, retries=3, timeout=20):
    sku_enc = quote(sku, safe="")
    for attempt in range(1, retries + 1):
        try:
            data = magento_get(f"/stockItems/{sku_enc}", timeout=timeout)
            return int(data.get("backorders", 0))
        except Exception:
            if attempt == retries:
                return 0
            time.sleep(min(8, 2 ** attempt))


def load_items():
    # 1) Products of attribute set "Default"
    attr_id = get_attr_set_id("Default")
    ph = st.empty()
    prog = st.progress(0.0, text="Loading products in 'Default' set…")
    rows = []
    count_total = 0
    for p, total in iter_products_by_attr_set(attr_id, progress_cb=prog.progress):
        if not count_total:
            count_total = total
        rows.append({
            "sku": p["sku"],
            "name": p["name"],
            "created_at": p.get("created_at", ""),
            "status": int(get_attr(p, "status", 1) or 1),
            "visibility": int(get_attr(p, "visibility", 4) or 4),
            "type_id": p.get("type_id", "simple"),
        })
    prog.progress(1.0, text=f"Loaded {len(rows)} / {count_total} products")

    df = pd.DataFrame(rows)
    # Active, visible, relevant types
    df = df[(df["status"] == 1) & (df["visibility"] != 1) & (df["type_id"].isin({"simple", "configurable"}))].copy()

    # 2) MSI quantities from source 'default'
    prog2 = st.progress(0.0, text="Fetching MSI quantities (source: default)…")
    src_items = get_source_items("default", progress_cb=prog2.progress)
    qty_map = {s["sku"]: float(s.get("quantity", 0)) for s in src_items}
    prog2.progress(1.0, text=f"Fetched MSI quantities: {len(src_items)} rows")

    df["qty"] = df["sku"].map(qty_map).fillna(0.0)

    # 3) Backorders for zero-qty SKUs
    zero_skus = df[df["qty"] <= 0]["sku"].tolist()
    back_map = {}
    if zero_skus:
        prog3 = st.progress(0.0, text=f"Checking backorders for {len(zero_skus)} zero-qty SKUs…")
        max_workers = min(16, max(4, len(zero_skus) // 400 + 8))
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(get_backorders, sku): sku for sku in zero_skus}
            for fut in as_completed(futures):
                sku = futures[fut]
                try:
                    back_map[sku] = fut.result()
                except Exception:
                    back_map[sku] = 0
                done += 1
                if done % 50 == 0:
                    prog3.progress(done / len(zero_skus), text=f"Checking backorders… {done}/{len(zero_skus)}")
        prog3.progress(1.0, text="Backorders check complete")
    else:
        st.info("No zero-qty SKUs — backorders check skipped.")

    df["backorders"] = df["sku"].map(back_map).fillna(0).astype(int)

    # 4) Final filter: qty>0 OR backorders==2
    df = df[(df["qty"] > 0) | (df["backorders"] == 2)].copy()

    # 5) Show table with required columns
    df_ui = pd.DataFrame({
        "sku": df["sku"],
        "name": df["name"],
        "attribute set": "Default",
        "date created": df["created_at"],
    })
    st.success(f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)")
    df_ui["date created"] = pd.to_datetime(df_ui["date created"], errors="coerce")
    df_ui = df_ui.sort_values("date created", ascending=False).reset_index(drop=True)
    st.dataframe(df_ui, use_container_width=True)
    return df_ui


st.title("Default Set — In-Stock & Backorder Browser")
st.caption("Filters: attribute set = Default; qty > 0 OR backorders = 2 (Allow Qty Below 0 & Notify Customer).")

if st.button("Load items", type="primary"):
    try:
        load_items()
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")

