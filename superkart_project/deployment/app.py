import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# Load the saved model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="Sachinnb24/Super-kart-predition",
    filename="best_superkart_sales_model_v1.joblib",
)
model = joblib.load(model_path)

st.title("SuperKart — Sales Prediction")
st.write("Fill in the product and store details below to predict the total sales revenue.")

# ------------------------------------
# Numeric inputs
# ------------------------------------
Product_Weight = st.number_input("Product Weight (kg)", min_value=0.10, max_value=50.0, value=12.0, step=0.01)
Product_Allocated_Area = st.slider("Product Allocated Area Ratio", 0.000, 0.500, 0.050, step=0.001, format="%.3f")
Product_MRP = st.number_input("Product MRP (₹)", min_value=10.0, max_value=500.0, value=150.0, step=0.5)
Store_Establishment_Year = st.selectbox("Store Establishment Year", list(range(1985, 2024)))

# ------------------------------------
# Categorical inputs
# ------------------------------------
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Type = st.selectbox("Product Type", [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
    "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
    "Health and Hygiene", "Household", "Meat", "Others",
    "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
])
Store_Size = st.selectbox("Store Size", ["High", "Medium", "Small"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2",
])

# ------------------------------------
# Build input dataframe and predict
# ------------------------------------
input_data = pd.DataFrame([{
    "Product_Weight":           Product_Weight,
    "Product_Allocated_Area":   Product_Allocated_Area,
    "Product_MRP":              Product_MRP,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Product_Sugar_Content":    Product_Sugar_Content,
    "Product_Type":             Product_Type,
    "Store_Size":               Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type":               Store_Type,
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales Total: \u20b9 {prediction:,.2f}")
