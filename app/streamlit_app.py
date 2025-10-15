from __future__ import annotations

import pandas as pd
import streamlit as st

from services.attributes import build_attributes_table_for_sku
from services.inventory import load_default_items
from utils.http import get_session


st.set_page_config(page_title="Default Set In-Stock Browser", layout="wide")

base_url = st.secrets["MAGENTO_BASE_URL"].rstrip("/")
auth_token = st.secrets["MAGENTO_ADMIN_TOKEN"]
session = get_session(auth_token)

st.title("Default Set — In-Stock & Backorder Browser")
st.caption(
    "Filters: attribute set = Default; qty > 0 OR backorders = 2 (Allow Qty Below 0 & Notify Customer)."
)

if st.button("Load items", type="primary"):
    try:
        df_items = load_default_items(session, base_url)
        if df_items.empty:
            st.warning("No items match the Default set filter criteria.")
        else:
            df_ui = df_items.copy()
            df_ui["date created"] = pd.to_datetime(
                df_ui["date created"], errors="coerce"
            )
            df_ui = df_ui.sort_values("date created", ascending=False).reset_index(drop=True)
            st.success(
                f"Found {len(df_ui)} products (Default; qty>0 OR backorders=2)"
            )
            st.dataframe(df_ui, use_container_width=True)
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"Error: {exc}")
else:
    st.info("Нажми **Load items** для загрузки и отображения товаров.")

st.divider()
st.header("Attributes lookup", anchor="attributes")
sku = st.text_input("SKU", placeholder="Enter product SKU")
include_empty = st.checkbox("Include empty attributes", value=False)

if st.button("Show attributes"):
    if not sku:
        st.warning("Please enter a SKU value first.")
    else:
        try:
            df_attrs = build_attributes_table_for_sku(
                session, base_url, sku.strip(), include_empty=include_empty
            )
            if df_attrs.empty:
                st.info("No attributes to display for the provided SKU.")
            else:
                st.dataframe(df_attrs, use_container_width=True)
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Error: {exc}")
