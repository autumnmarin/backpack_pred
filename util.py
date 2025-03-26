import os
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
seed = 87

# -------------------------------
# Custom KerasRegressor Wrapper
# -------------------------------
class CustomKerasRegressor(KerasRegressor, BaseEstimator, RegressorMixin):
    @classmethod
    def __sklearn_tags__(cls):
        # This minimal tag set informs scikit-learn of key estimator properties.
        return {"requires_fit": True, "non_deterministic": True}

# Force-patch if necessary
CustomKerasRegressor.__sklearn_tags__ = classmethod(lambda cls: {"requires_fit": True, "non_deterministic": True})

# -------------------------------
# Utility Functions
# -------------------------------

def load_data(script_dir):
    train_path = os.path.join(script_dir, "data", "train.csv")
    train_extra_path = os.path.join(script_dir, "data", "training_extra.csv")
    test_path = os.path.join(script_dir, "data", "test.csv")
    
    df_train = pd.read_csv(train_path, low_memory=False)
    df_train_extra = pd.read_csv(train_extra_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)
    
    # Concatenate training datasets
    df_train = pd.concat([df_train, df_train_extra], ignore_index=True)
    return df_train, df_test

def preprocess_data(df_train, df_test, subsample_fraction=0.01, target="Price"):
    df_train = df_train.sample(frac=subsample_fraction, random_state=seed)
    df_train.rename(columns={"Weight Capacity (kg)": "Weight"}, inplace=True)
    df_test.rename(columns={"Weight Capacity (kg)": "Weight"}, inplace=True)

    if "id" in df_train.columns:
        df_train.drop(columns=["id"], inplace=True)
    if "id" in df_test.columns:
        test_ids = df_test["id"].copy()
        df_test.drop(columns=["id"], inplace=True)
    else:
        raise ValueError("The test.csv file must contain an 'id' column.")

    categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
    numerical_cols = ["Compartments", "Weight"]

    df_train[categorical_cols] = df_train[categorical_cols].fillna("Unknown")
    df_test[categorical_cols] = df_test[categorical_cols].fillna("Unknown")

    imputer = SimpleImputer(strategy="mean")
    df_train[numerical_cols] = imputer.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = imputer.transform(df_test[numerical_cols])

    print("Preprocessing complete. Training data shape:", df_train.shape)
    return df_train, df_test, categorical_cols, numerical_cols, test_ids

def feature_engineering(df):
    df["Brand_Material"] = (df["Brand"].astype(str) + "_" + df["Material"].astype(str)).astype("category")
    df["Brand_Size"] = (df["Brand"].astype(str) + "_" + df["Size"].astype(str)).astype("category")
    df["Brand_Style"] = (df["Brand"].astype(str) + "_" + df["Style"].astype(str)).astype("category")
    df["Material_Size"] = (df["Material"].astype(str) + "_" + df["Size"].astype(str)).astype("category")
    df["Material_Style"] = (df["Material"].astype(str) + "_" + df["Style"].astype(str)).astype("category")
    df["Compartments_Squared"] = df["Compartments"] ** 2
    df["Weight_Squared"] = df["Weight"] ** 2
    df["Compartments_Weight"] = df["Compartments"] * df["Weight"]
    df["Laptop_Waterproof"] = (df["Laptop Compartment"].astype(str) + "_" + df["Waterproof"].astype(str)).astype("category")
    df["Laptop_Style"] = (df["Laptop Compartment"].astype(str) + "_" + df["Style"].astype(str)).astype("category")
    df["Color_Brand"] = (df["Color"].astype(str) + "_" + df["Brand"].astype(str)).astype("category")
    df["Color_Material"] = (df["Color"].astype(str) + "_" + df["Material"].astype(str)).astype("category")
    df["Log_Weight"] = np.log1p(df["Weight"])
    df["Log_Compartments"] = np.log1p(df["Compartments"])
    df["Many_Compartments"] = (df["Compartments"] > df["Compartments"].median()).astype(int)
    df["Heavy_Capacity"] = (df["Weight"] > df["Weight"].median()).astype(int)
    df["Weight_Minus_Compartments"] = df["Weight"] - df["Compartments"]
    df["Weight_to_Compartments"] = df["Weight"] / (df["Compartments"] + 1)
    df["Brand_Material_Size"] = (df["Brand"].astype(str) + "_" + df["Material"].astype(str) + "_" + df["Size"].astype(str)).astype("category")
    df["Style_Color_Impact"] = (df["Style"].astype(str) + "_" + df["Color"].astype(str)).astype("category")
    return df

def one_hot_encode(df_train, df_test, categorical_cols_extended):
    df_train = pd.get_dummies(df_train, columns=categorical_cols_extended, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_cols_extended, drop_first=True)
    df_test = df_test.reindex(columns=df_train.columns.drop("Price"), fill_value=0)
    return df_train, df_test

def remove_outliers(df):
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]
    return df

def standardize_features(X_train, X_test, numerical_cols):
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test

def vectorized_feature_engineering(X_train, X_test, skewed_features=["Weight", "Compartments"]):
    for df in [X_train, X_test]:
        df[skewed_features] = df[skewed_features].clip(lower=0).fillna(0)
        for col in skewed_features:
            df[f"{col}_log"] = np.log1p(df[col])
        df["Weight_Compartments"] = df["Weight"] * df["Compartments"]
    return X_train, X_test

from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y, name="Model", cv=5):
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    rmse_scores = -scores
    mean_rmse = rmse_scores.mean()
    print(f"📊 {name} Cross-Validated RMSE: {mean_rmse:.4f}")
    return mean_rmse
