"""
PROJECT 3 — Supervised Learning: Classify Learner Risk Level
Target: risk_label  (at_risk / average / strong_learner)
Dataset: ../data/student_data.csv

Run: python classification_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/student_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/classification_model.joblib")
TARGET     = "risk_label"

NUMERIC_FEATURES     = ["study_hours","attendance_pct","assignments_done",
                         "login_days","late_submissions","quiz_retries","engagement_score"]
CATEGORICAL_FEATURES = ["gender","parental_education","lunch","test_preparation"]

print("=" * 60)
print("PROJECT 3 — CLASSIFICATION: Learner Risk Level")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 {df.shape[0]} students  |  Target: '{TARGET}'")
print(f"   Class distribution:\n{df[TARGET].value_counts().to_string()}")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")

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

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)"           : KNeighborsClassifier(n_neighbors=5),
}

print(f"\n{'Model':<24} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7}")
print("-" * 57)

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec   = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
    results[name] = {"pipe": pipe, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "preds": preds}
    print(f"  {name:<22} {acc:>7.3f} {prec:>7.3f} {rec:>8.3f} {f1:>7.3f}")

best_name = max(results, key=lambda n: results[n]["f1"])
best      = results[best_name]
print(f"\n🏆 Best: {best_name}  (F1={best['f1']:.3f})")

print("\n📋 Detailed Report (Best Model):")
print(classification_report(y_test, best["preds"], zero_division=0))

print("🔲 Confusion Matrix (Best Model):")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, best["preds"], labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in labels],
                         columns=[f"Pred:{l}" for l in labels])
print(cm_df.to_string())

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
for f, v in imps.head(5).items():
    print(f"   {f:<30} {v:.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best["pipe"], MODEL_PATH)
print(f"\n💾 Saved → {MODEL_PATH}")

# ── Predict New Student ────────────────────────────────────────────────────────
def predict_risk(study_h, attendance, assignments, logins,
                 late_sub=5, retries=3, gender="male",
                 parental_edu="high school", lunch="free/reduced",
                 test_prep="none"):
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
    label = pipe.predict(row)[0]
    proba = pipe.predict_proba(row)[0]
    classes = pipe.classes_
    proba_dict = {c: round(float(p), 3) for c, p in zip(classes, proba)}
    return label, proba_dict

label, proba = predict_risk(3, 55, 60, 20)
print(f"\n🎯 Demo: 3h study | 55% attend | 60% assignments | 20 logins")
print(f"   → Risk Label: {label}")
print(f"   → Probabilities: {proba}")
print("\n✅ Project 3 complete!")

if __name__ == "__main__":
    pass
