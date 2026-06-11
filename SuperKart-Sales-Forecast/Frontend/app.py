"""
SuperKart Sales Forecast — Streamlit Frontend
Provides an interactive UI for single and batch sales predictions.
"""

import streamlit as st
import pandas as pd
import requests

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")

st.title(" SuperKart — Sales Revenue Forecast")
st.write(
    "This tool predicts the **total sales revenue** (`Product_Store_Sales_Total`) "
    "for a product in a SuperKart store based on product and store attributes."
)

# ── Backend API URL ───────────────────────────────────────────────────────────
BACKEND_URL = "https://manasa92-superkart-backend.hf.space"

# ── Single Prediction ─────────────────────────────────────────────────────────
st.header("Single Product Prediction")
st.write("Adjust the sliders and dropdowns, then click **Predict Sales**.")

col1, col2 = st.columns(2)

with col1:
    product_weight       = st.slider("Product Weight (kg)", 4.0, 22.0, 12.0, 0.1)
    product_allocated_area = st.slider("Product Allocated Area (ratio)", 0.0, 0.15, 0.06, 0.005)
    product_mrp          = st.slider("Product MRP ($)", 30.0, 270.0, 140.0, 1.0)
    store_age            = st.slider("Store Age (years)", 4, 26, 14, 1)

with col2:
    sugar_content    = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_type     = st.selectbox("Product Type", [
        "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy",
        "Household", "Baking Goods", "Canned", "Health and Hygiene",
        "Meat", "Soft Drinks", "Breads", "Hard Drinks",
        "Others", "Starchy Foods", "Breakfast", "Seafood"
    ])
    store_size       = st.selectbox("Store Size", ["Medium", "High", "Small"])
    city_type        = st.selectbox("Store City Type", ["Tier 2", "Tier 1", "Tier 3"])
    store_type       = st.selectbox("Store Type", [
        "Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"
    ])

if st.button(" Predict Sales", type="primary"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Store_Age": store_age
    }

    try:
        response = requests.post(f"{BACKEND_URL}/v1/sales", json=payload, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Predicted_Sales"]
            st.success(f" Predicted Sales Revenue: **${predicted_sales:,.2f}**")
        else:
            st.error(f"API error: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# ── Batch Prediction ──────────────────────────────────────────────────────────
st.divider()
st.header("Batch Prediction")
st.write(
    "Upload a CSV file with columns: `Product_Weight`, `Product_Sugar_Content`, "
    "`Product_Allocated_Area`, `Product_Type`, `Product_MRP`, `Store_Size`, "
    "`Store_Location_City_Type`, `Store_Type`, `Store_Age`"
)

uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

if uploaded_file is not None:
    if st.button(" Predict Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/salesbatch",
                files={"file": uploaded_file},
                timeout=60
            )
            if response.status_code == 200:
                results = response.json()
                result_df = pd.DataFrame(results)
                st.success(f"Predictions generated for {len(result_df)} records.")
                st.dataframe(result_df)
                st.download_button(
                    label=" Download Predictions CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"API error: {response.status_code} — {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")