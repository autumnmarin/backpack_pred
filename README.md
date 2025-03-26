![Image](https://github.com/user-attachments/assets/d4c7c895-bfe9-49d4-a810-501d2aa8f7d2)

# 🎒 What’s in the Bag? Breaking Down Price Prediction with ML

> *Can a blend of real-world intuition and modern machine learning predict backpack prices better than gradient boosting alone?*  
> This project dives deep into structured tabular modeling to answer just that—using domain-inspired feature engineering, optimized boosting models, and ensemble stacking to push predictive accuracy to new lows (RMSE-wise!).  

---

### 🧠 TL;DR

A diverse feature space was engineered from product metadata like weight, compartments, brand, and material, and a suite of regression models were compared head-to-head.
📉 Best model: **Stacked Ensemble** (XGBoost + LightGBM + CatBoost with Ridge)  
🏆 **Kaggle Private Score: 38.91 RMSE** — Top **22%** on the leaderboard

---

## 📌 Project Overview

This regression task predicts the **price of backpacks** based on structured attributes like:

- Brand, Size, Material, Style
- Number of compartments
- Whether it's waterproof or has a laptop compartment
- Weight capacity (in kg)

The challenge: pricing is nonlinear, brand-sensitive, and influenced by subtle feature interactions.

---

## 🎯 Modeling Strategy

### 🔨 Feature Engineering
Rather than using basic one-hot encoding alone, domain-driven feature combinations were constructed:

- `Brand_Material`, `Laptop_Waterproof`, and `Color_Style`
- Numerical interactions: `Weight * Compartments`, `Weight^2`, `Log(Weight)`
- Binary indicators: *Heavy Capacity*, *Many Compartments*

All models used **the exact same feature space** via a shared `util.py` module for fair benchmarking.

---

## 🤖 Models Compared

| Model               | Notebook RMSE | Kaggle Private | Kaggle Public |
|--------------------|----------------|----------------|----------------|
| Stacked Ensemble    | **38.87**       | **38.91**       | **39.11**       |
| XGBoost (Optuna)    | 40.40           | 40.48           | 40.64           |
| CatBoost (Base)     | 40.42           | 40.46           | 40.63           |
| LightGBM (Base)     | 40.45           | 40.47           | 40.65           |

> 🧪 *Neural nets were tested but underperformed—this dataset favors structured gradient boosting.*

---

## 📊 Visual Comparison

<img src="https://github.com/user-attachments/assets/1f9cc530-97ab-4e69-a991-792d5043eaa5" width = 600>



---

## 🔍 Findings

1. **Feature engineering mattered more than model complexity**.
2. **Stacking worked** — combining models captured different biases.
3. **Optuna tuning helped XGBoost**, but wasn’t enough to beat the stack.
4. **Standardization + log-target transformation** stabilized performance.

---

## 🛠️ Tools & Stack

- Python 3.10
- scikit-learn, XGBoost, LightGBM, CatBoost
- PyCaret for rapid baseline comparison
- Optuna for hyperparameter optimization
- Matplotlib for final visualizations

---

## 🧭 Next Steps

- Add SHAP or permutation-based feature importance
- Try TabNet or gradient-boosted decision forests for contrast
- Deploy a small Streamlit app to test predictions live

---

## 🥇 Result

Top 22% on Kaggle  
Final RMSE: **38.91**  
More importantly: clean, testable code and reusable utilities

---









![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=autumnmarin.backpack_pred)
