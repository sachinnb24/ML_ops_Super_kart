import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from Hugging Face dataset space
DATASET_PATH = "hf://datasets/Sachinnb24/Super-kart-predition/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ------------------------------------
# Data Cleaning
# ------------------------------------

# Drop identifier columns — not useful as model features
df = df.drop(columns=["Product_Id", "Store_Id"])

# Standardize inconsistent Product_Sugar_Content values
# 'reg' is a misspelling of 'Regular'
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# Impute missing numeric values with column mean
df["Product_Weight"] = df["Product_Weight"].fillna(df["Product_Weight"].mean())

# Impute missing categorical values with mode
df["Store_Size"] = df["Store_Size"].fillna(df["Store_Size"].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------
# Define target and feature columns
# ------------------------------------
target = "Product_Store_Sales_Total"

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

X = df[numeric_features + categorical_features]
y = df[target]

# ------------------------------------
# Train-test split (80/20)
# ------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {Xtrain.shape}, Test size: {Xtest.shape}")

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload train/test splits back to Hugging Face dataset space
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="Sachinnb24/Super-kart-predition",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face.")
