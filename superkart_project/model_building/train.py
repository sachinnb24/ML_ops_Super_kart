import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os


mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("SuperKart-Sales-Prediction-Experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train and test data from Hugging Face dataset space
Xtrain_path = "hf://datasets/Sachinnb24/Super-kart-predition/Xtrain.csv"
Xtest_path  = "hf://datasets/Sachinnb24/Super-kart-predition/Xtest.csv"
ytrain_path = "hf://datasets/Sachinnb24/Super-kart-predition/ytrain.csv"
ytest_path  = "hf://datasets/Sachinnb24/Super-kart-predition/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

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

# Preprocessing: scale numeric features, one-hot encode categoricals
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    "xgbregressor__n_estimators":   [100, 150, 200],
    "xgbregressor__max_depth":      [3, 4, 5],
    "xgbregressor__learning_rate":  [0.05, 0.1, 0.2],
    "xgbregressor__colsample_bytree": [0.7, 0.8, 1.0],
    "xgbregressor__subsample":      [0.7, 0.8, 1.0],
}

# Build full pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():

    # Tune model with GridSearchCV (scoring: negative RMSE)
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Generate predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    # Evaluate model performance
    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    train_mae  = mean_absolute_error(ytrain, y_pred_train)
    train_r2   = r2_score(ytrain, y_pred_train)

    test_rmse  = np.sqrt(mean_squared_error(ytest, y_pred_test))
    test_mae   = mean_absolute_error(ytest, y_pred_test)
    test_r2    = r2_score(ytest, y_pred_test)

    print(f"Train RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")

    # Log evaluation metrics to MLflow
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "train_mae":  train_mae,
        "train_r2":   train_r2,
        "test_rmse":  test_rmse,
        "test_mae":   test_mae,
        "test_r2":    test_r2,
    })

    # Save the best model locally
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved: {model_path}")

    # Register the best model in Hugging Face Model Hub
    repo_id   = "Sachinnb24/Super-kart-predition"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Best model registered in Hugging Face Model Hub.")
