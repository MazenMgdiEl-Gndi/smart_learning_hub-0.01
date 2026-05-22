"""
PROJECT 2 — Regression Model: Predicting Student Average Score
Dataset: ../data/student_data.csv

Run: python regression_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/student_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/regression_model.joblib")
TARGET     = "average_score"

NUMERIC_FEATURES = [
    "study_hours", "attendance_pct", "assignments_done",
    "login_days", "late_submissions", "quiz_retries", "engagement_score"
]
CATEGORICAL_FEATURES = ["gender", "parental_education", "lunch", "test_preparation"]

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PROJECT 2 — REGRESSION: Predict Average Score")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Loaded {df.shape[0]} students, {df.shape[1]} columns")
print(f"   Target: '{TARGET}'  mean={df[TARGET].mean():.1f}  std={df[TARGET].std():.1f}")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL_FEATURES),
])

# ── Train & Evaluate ───────────────────────────────────────────────────────────
models = {
    "Linear Regression"  : LinearRegression(),
    "Decision Tree"      : DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print(f"\n{'Model':<24} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
print("-" * 50)

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = mean_squared_error(y_test, preds) ** 0.5
    r2    = r2_score(y_test, preds)
    results[name] = {"pipe": pipe, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"  {name:<22} {mae:>7.2f} {rmse:>7.2f} {r2:>7.3f}")

best_name = max(results, key=lambda n: results[n]["r2"])
best      = results[best_name]
print(f"\n🏆 Best: {best_name}  (R² = {best['r2']:.3f})")

# ── Feature Importance ─────────────────────────────────────────────────────────
rf_pipe = results["Random Forest"]["pipe"]
ohe_cols = list(rf_pipe.named_steps["prep"]
                        .named_transformers_["cat"]
                        .named_steps["enc"]
                        .get_feature_names_out(CATEGORICAL_FEATURES))
all_feat = NUMERIC_FEATURES + ohe_cols
imps = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                 index=all_feat).sort_values(ascending=False)
print("\n🔍 Top 5 Features (Random Forest):")
for feat, val in imps.head(5).items():
    print(f"   {feat:<30} {val:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best["pipe"], MODEL_PATH)
print(f"\n💾 Saved → {MODEL_PATH}")

# ── Predict New Student ────────────────────────────────────────────────────────
def predict_score(study_h, attendance, assignments, logins,
                  late_sub=2, retries=1, gender="female",
                  parental_edu="bachelor's degree",
                  lunch="standard", test_prep="completed"):
    pipe = joblib.load(MODEL_PATH)
    eng  = 0.4*(study_h/12) + 0.3*(attendance/100) + \
           0.2*(assignments/100) + 0.1*(logins/90)
    row  = pd.DataFrame([{
        "study_hours": study_h, "attendance_pct": attendance,
        "assignments_done": assignments, "login_days": logins,
        "late_submissions": late_sub, "quiz_retries": retries,
        "engagement_score": round(eng, 3),
        "gender": gender, "parental_education": parental_edu,
        "lunch": lunch, "test_preparation": test_prep,
    }])
    return round(float(pipe.predict(row)[0]), 2)

score = predict_score(7, 85, 90, 60)
print(f"\n🎯 Demo: 7h study | 85% attend | 90% assignments | 60 logins")
print(f"   → Predicted Score: {score}")
print("\n✅ Project 2 complete!")

if __name__ == "__main__":
    pass
