import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from util import (
    load_data,
    preprocess_data,
    feature_engineering,
    one_hot_encode,
    remove_outliers,
    standardize_features,
    vectorized_feature_engineering
)

warnings.filterwarnings("ignore")
SEED = 17
SUBSAMPLE_FRACTION = 0.2

# -------------------------------
# Load and Preprocess Data
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train, df_test = load_data(script_dir)
df_train, df_test, cat_cols, num_cols, test_ids = preprocess_data(df_train, df_test, subsample_fraction=SUBSAMPLE_FRACTION)

# Feature Engineering
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

# Extended categorical columns
cat_cols_extended = cat_cols + [
    "Brand_Material", "Brand_Size", "Brand_Style", "Material_Size", "Material_Style",
    "Laptop_Waterproof", "Laptop_Style", "Color_Brand", "Color_Material",
    "Brand_Material_Size", "Style_Color_Impact"
]
for col in cat_cols_extended:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")

# One-hot Encoding and Outlier Removal
df_train, df_test = one_hot_encode(df_train, df_test, cat_cols_extended)
df_train = remove_outliers(df_train)

# Split features and target
y = df_train["Price"]
X = df_train.drop(columns=["Price"])
X_test = df_test.reindex(columns=X.columns, fill_value=0)

# Standardization and Feature Refinement
X, X_test = standardize_features(X, X_test, num_cols)
X, X_test = vectorized_feature_engineering(X, X_test, skewed_features=["Weight", "Compartments"])

# Log-transform the target
y = np.log1p(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# -------------------------------
# Train LightGBM Model
# -------------------------------
lgb_model = LGBMRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=SEED
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# -------------------------------
# Evaluation and Visualization
# -------------------------------
y_pred_train = np.expm1(lgb_model.predict(X_train))
y_train_original = np.expm1(y_train)
y_pred_val = np.expm1(lgb_model.predict(X_val))
y_val_original = np.expm1(y_val)

train_rmse = np.sqrt(mean_squared_error(y_train_original, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_val))
print(f"\n✅ LightGBM Train RMSE: {train_rmse:.4f}")
print(f"✅ LightGBM Validation RMSE: {val_rmse:.4f}")

# plt.figure(figsize=(8, 6))
# plt.bar(["Train RMSE", "Validation RMSE"], [train_rmse, val_rmse], color=["#6A5ACD", "#20B2AA"])
# plt.ylabel("RMSE")
# plt.title("LightGBM: Training vs. Validation RMSE")
# plt.grid(axis="y")
# plt.text(0, train_rmse + 1, f"{train_rmse:.2f}", ha="center", fontsize=12)
# plt.text(1, val_rmse + 1, f"{val_rmse:.2f}", ha="center", fontsize=12)
# plt.tight_layout()
# plt.show()

# -------------------------------
# Final Prediction and Submission
# -------------------------------
y_pred_test = np.expm1(lgb_model.predict(X_test))
submission = pd.DataFrame({"id": test_ids, "Price": y_pred_test})
submission.to_csv("submissionL.csv", index=False)
print("\n📁 Submission file 'submission.csv' saved successfully!")