import os
import time
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from util import (
    load_data,
    preprocess_data,
    feature_engineering,
    one_hot_encode,
    remove_outliers,
    standardize_features,
    vectorized_feature_engineering
)

# Suppress warnings
warnings.filterwarnings("ignore")
start_time = time.time()
seed = 17

# Load and preprocess data
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train, df_test = load_data(script_dir)
df_train, df_test, cat_cols, num_cols, test_ids = preprocess_data(df_train, df_test, subsample_fraction=0.05)

# Feature engineering
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

# Extended categorical list
cat_cols_extended = cat_cols + [
    "Brand_Material", "Brand_Size", "Brand_Style", "Material_Size", "Material_Style",
    "Laptop_Waterproof", "Laptop_Style", "Color_Brand", "Color_Material",
    "Brand_Material_Size", "Style_Color_Impact"
]

# One-hot encode
df_train, df_test = one_hot_encode(df_train, df_test, cat_cols_extended)

# Remove outliers
df_train = remove_outliers(df_train)

# Separate target and features
y_train = df_train["Price"]
X_train = df_train.drop(columns=["Price"])
X_test = df_test.reindex(columns=X_train.columns, fill_value=0)

# Standardize and apply additional feature engineering
X_train, X_test = standardize_features(X_train, X_test, num_cols)
X_train, X_test = vectorized_feature_engineering(X_train, X_test, skewed_features=["Weight", "Compartments"])

# Train base models
xgb_best = xgb.XGBRegressor(
    objective="reg:squarederror",
    subsample=0.8,
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.01,
    colsample_bytree=0.6,
    tree_method="gpu_hist",
    random_state=seed
)
xgb_best.fit(X_train, y_train)

use_gpu = lgb.__version__ >= "3.2.0"
lgb_best = lgb.LGBMRegressor(
    num_leaves=50,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.8,
    device="gpu" if use_gpu else "cpu",
    random_state=seed
)
lgb_best.fit(X_train, y_train)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=seed,
    verbose=False
)
cat_model.fit(X_train, y_train)

# Stacking ensemble
stacked_model = StackingRegressor(
    estimators=[
        ("xgb", xgb_best),
        ("lgb", lgb_best),
        ("cat", cat_model),
    ],
    final_estimator=Ridge()
)
stacked_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, name):
    y_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print(f"{name} RMSE: {rmse:.4f}")
    return rmse

xgb_rmse = evaluate_model(xgb_best, "XGBoost")
lgb_rmse = evaluate_model(lgb_best, "LightGBM")
cat_rmse = evaluate_model(cat_model, "CatBoost")
stacked_rmse = evaluate_model(stacked_model, "Stacked Model")

# Output all RMSEs
print("\nModel RMSE Comparison:")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"LightGBM RMSE: {lgb_rmse:.4f}")
print(f"CatBoost RMSE: {cat_rmse:.4f}")
print(f"Stacked Model RMSE: {stacked_rmse:.4f}")

# Make predictions and save submission
y_pred = stacked_model.predict(X_test)
submission = pd.DataFrame({"id": test_ids, "Price": y_pred})
submission.to_csv("submission.csv", index=False)
print("\nPredictions saved to 'submission.csv'.")

# Output best hyperparameters (hardcoded)
print("\nBest Hyperparameters (Hardcoded):")
print("XGBoost: subsample=0.8, n_estimators=1000, max_depth=4, learning_rate=0.01, colsample_bytree=0.6")
print("LightGBM: num_leaves=50, n_estimators=500, learning_rate=0.01")
print("CatBoost: iterations=1000, learning_rate=0.05, depth=6")

# Runtime
end_time = time.time()
print(f"\nTotal Runtime: {end_time - start_time:.2f} seconds")
