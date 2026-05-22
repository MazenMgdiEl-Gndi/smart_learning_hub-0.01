"""
==============================================================
  PROJECT 1 — End-to-End Machine Learning Pipeline
==============================================================

THEME : Learning Analytics & Intelligent Study Assistant
GOAL  : Build a complete ML pipeline from raw data to prediction
LEVEL : Beginner-friendly — every step is explained

WHAT THIS PIPELINE DOES (in order):
  Stage 1 → Load data
  Stage 2 → Clean data
  Stage 3 → Feature engineering
  Stage 4 → Train/test split
  Stage 5 → Scale features
  Stage 6 → Train model
  Stage 7 → Evaluate model
  Stage 8 → Save model
  Stage 9 → Predict on new student input

HOW TO RUN:
  python project1_pipeline.py

OUTPUT FILES:
  models/trained_model.pkl        ← saved trained model
  models/scaler.pkl               ← saved feature scaler
  models/feature_names.pkl        ← saved feature list
  outputs/evaluation_report.txt   ← model performance report
  outputs/feature_importance.png  ← chart of important features
  outputs/actual_vs_predicted.png ← chart of predictions vs reality
==============================================================
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib          # for saving/loading models
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/student_performance.csv"
MODELS_DIR  = "models"
OUTPUT_DIR  = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

print("=" * 60)
print("  PROJECT 1 — End-to-End ML Pipeline")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# STAGE 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n📂  STAGE 1: Loading data …")

df = pd.read_csv(DATA_PATH)
print(f"   ✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════
# STAGE 2 — CLEAN DATA
# ══════════════════════════════════════════════════════════════
print("\n🧹  STAGE 2: Cleaning data …")

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"   Duplicates removed : {before - len(df)}")

# Clip to valid ranges
df["attendance_pct"]   = df["attendance_pct"].clip(0, 100)
df["study_hours"]      = df["study_hours"].clip(0, 24)
df["final_score"]      = df["final_score"].clip(0, 100)
df["missed_deadlines"] = df["missed_deadlines"].clip(0, None)

print(f"   ✅ Clean dataset: {len(df)} rows")


# ══════════════════════════════════════════════════════════════
# STAGE 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n⚙️   STAGE 3: Feature engineering …")

# Create new useful features from existing ones

# engagement_score: how actively the student uses the platform
df["engagement_score"] = (
    df["login_frequency"] * 0.3 +
    df["assignments_done"] * 0.4 +
    df["quizzes_solved"] * 0.3
).round(2)

# submission_rate: assignments done vs missed deadlines (higher = better)
df["submission_rate"] = (
    df["assignments_done"] / (df["assignments_done"] + df["missed_deadlines"] + 1)
).round(3)

# study_efficiency: study hours weighted by revision
df["study_efficiency"] = (
    df["study_hours"] * (1 + df["revision_sessions"] * 0.1)
).round(2)

print("   New features created:")
print("      • engagement_score  (login + assignments + quizzes combined)")
print("      • submission_rate   (assignments / total expected)")
print("      • study_efficiency  (study hours × revision bonus)")
print(f"   ✅ Dataset now has {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════
# STAGE 4 — DEFINE FEATURES AND TARGET, THEN SPLIT
# ══════════════════════════════════════════════════════════════
print("\n✂️   STAGE 4: Splitting into train and test sets …")

# These are the input columns (what the model learns from)
FEATURE_COLS = [
    "study_hours",
    "attendance_pct",
    "assignments_done",
    "quizzes_solved",
    "login_frequency",
    "missed_deadlines",
    "retries",
    "revision_sessions",
    "time_per_task_min",
    "engagement_score",
    "submission_rate",
    "study_efficiency",
]

# This is what we want to predict
TARGET_COL = "final_score"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set : {len(X_train)} students")
print(f"   Testing set  : {len(X_test)} students")
print(f"   Features used: {len(FEATURE_COLS)}")


# ══════════════════════════════════════════════════════════════
# STAGE 5 — SCALE FEATURES
# ══════════════════════════════════════════════════════════════
print("\n⚖️   STAGE 5: Scaling features …")

# StandardScaler converts all numbers to the same scale
# so no single feature dominates just because it's bigger
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn scale from train data
X_test_scaled  = scaler.transform(X_test)         # apply same scale to test

print("   ✅ Features scaled (mean=0, std=1)")


# ══════════════════════════════════════════════════════════════
# STAGE 6 — TRAIN MULTIPLE MODELS AND COMPARE
# ══════════════════════════════════════════════════════════════
print("\n🤖  STAGE 6: Training models …\n")

models = {
    "Linear Regression"         : LinearRegression(),
    "Decision Tree"             : DecisionTreeRegressor(random_state=42, max_depth=6),
    "Random Forest"             : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"         : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Cross-validation score (more reliable than single split)
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=5, scoring="r2")

    results[name] = {
        "model" : model,
        "y_pred": y_pred,
        "MAE"   : round(mae, 3),
        "RMSE"  : round(rmse, 3),
        "R2"    : round(r2, 3),
        "CV_R2" : round(cv_scores.mean(), 3),
    }

    print(f"   {name}")
    print(f"      MAE  = {mae:.2f}   (lower is better)")
    print(f"      RMSE = {rmse:.2f}   (lower is better)")
    print(f"      R²   = {r2:.3f}  (closer to 1.0 is better)")
    print(f"      CV R²= {cv_scores.mean():.3f}  (cross-validated, more reliable)")
    print()


# ══════════════════════════════════════════════════════════════
# STAGE 7 — EVALUATE: PICK THE BEST MODEL
# ══════════════════════════════════════════════════════════════
print("\n🏆  STAGE 7: Selecting best model …")

# Pick model with highest R² on test set
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
best_model = best["model"]
y_pred_best = best["y_pred"]

print(f"\n   🥇 Best model: {best_name}")
print(f"      MAE  = {best['MAE']}   (on average, predictions are off by {best['MAE']:.1f} points)")
print(f"      RMSE = {best['RMSE']}")
print(f"      R²   = {best['R2']}   (model explains {best['R2']*100:.1f}% of score variation)")


# ── Chart 1: Feature Importance (only works for tree-based models) ─────────────
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature"   : FEATURE_COLS,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                   color="#4C9BE8", edgecolor="white")
    ax.set_title(f"Feature Importance — {best_name}", fontsize=13)
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"\n   ✅ Feature importance chart saved → {path}")

# ── Chart 2: Actual vs Predicted ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.4, color="#4C9BE8", s=30)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
ax.plot([min_val, max_val], [min_val, max_val],
        "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")
ax.set_title(f"Actual vs Predicted — {best_name}\nR² = {best['R2']}", fontsize=12)
ax.legend()
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
plt.savefig(path)
plt.close()
print(f"   ✅ Actual vs Predicted chart saved → {path}")

# ── Chart 3: Model comparison bar chart ───────────────────────────────────────
compare_df = pd.DataFrame([
    {"Model": k, "R²": v["R2"], "MAE": v["MAE"]}
    for k, v in results.items()
]).sort_values("R²", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#4CAF50" if m == best_name else "#4C9BE8" for m in compare_df["Model"]]

axes[0].barh(compare_df["Model"], compare_df["R²"], color=colors, edgecolor="white")
axes[0].set_title("R² Score (higher = better)")
axes[0].set_xlabel("R²")
axes[0].axvline(0, color="gray", linewidth=0.5)

axes[1].barh(compare_df["Model"], compare_df["MAE"], color=colors, edgecolor="white")
axes[1].set_title("MAE — Mean Absolute Error (lower = better)")
axes[1].set_xlabel("MAE (points)")

plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(path)
plt.close()
print(f"   ✅ Model comparison chart saved → {path}")


# ══════════════════════════════════════════════════════════════
# STAGE 8 — SAVE THE BEST MODEL
# ══════════════════════════════════════════════════════════════
print("\n💾  STAGE 8: Saving model …")

joblib.dump(best_model,   os.path.join(MODELS_DIR, "trained_model.pkl"))
joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_names.pkl"))

print(f"   ✅ Model saved   → {MODELS_DIR}/trained_model.pkl")
print(f"   ✅ Scaler saved  → {MODELS_DIR}/scaler.pkl")
print(f"   ✅ Features saved→ {MODELS_DIR}/feature_names.pkl")


# ══════════════════════════════════════════════════════════════
# STAGE 9 — PREDICT ON A NEW STUDENT
# ══════════════════════════════════════════════════════════════
print("\n🔮  STAGE 9: Predicting for new students …\n")

def predict_student(student_data: dict) -> dict:
    """
    Takes a dictionary of student features and returns a predicted score.

    Parameters:
        student_data (dict): feature values for one student

    Returns:
        dict with predicted score and interpreted label
    """
    # Load saved artifacts
    model_     = joblib.load(os.path.join(MODELS_DIR, "trained_model.pkl"))
    scaler_    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feat_cols_ = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Build the engineered features
    student_data["engagement_score"] = (
        student_data["login_frequency"] * 0.3 +
        student_data["assignments_done"] * 0.4 +
        student_data["quizzes_solved"] * 0.3
    )
    student_data["submission_rate"] = (
        student_data["assignments_done"] /
        (student_data["assignments_done"] + student_data["missed_deadlines"] + 1)
    )
    student_data["study_efficiency"] = (
        student_data["study_hours"] * (1 + student_data["revision_sessions"] * 0.1)
    )

    # Convert to DataFrame row
    row = pd.DataFrame([student_data])[feat_cols_]

    # Scale
    row_scaled = scaler_.transform(row)

    # Predict
    score = float(model_.predict(row_scaled)[0])
    score = round(np.clip(score, 0, 100), 1)

    # Classify
    if score < 40:
        status = "⚠️  At Risk"
        advice = "Needs immediate attention. Focus on assignments and deadlines."
    elif score < 70:
        status = "🟡 Average"
        advice = "On track. Increase revision sessions and reduce missed deadlines."
    else:
        status = "🟢 Strong"
        advice = "Excellent performance. Keep up current habits."

    return {"predicted_score": score, "status": status, "advice": advice}


# ── Example 1: A struggling student ───────────────────────────────────────────
student_1 = {
    "study_hours"      : 2.0,
    "attendance_pct"   : 55.0,
    "assignments_done" : 4,
    "quizzes_solved"   : 2,
    "login_frequency"  : 5,
    "missed_deadlines" : 7,
    "retries"          : 1,
    "revision_sessions": 0,
    "time_per_task_min": 45.0,
}

result_1 = predict_student(student_1)
print("   📌 Student 1 — Struggling")
print(f"      Input  : {student_1['study_hours']}h study, {student_1['attendance_pct']}% attendance, {student_1['missed_deadlines']} missed deadlines")
print(f"      Score  : {result_1['predicted_score']} / 100")
print(f"      Status : {result_1['status']}")
print(f"      Advice : {result_1['advice']}")

# ── Example 2: A strong student ───────────────────────────────────────────────
student_2 = {
    "study_hours"      : 6.5,
    "attendance_pct"   : 92.0,
    "assignments_done" : 17,
    "quizzes_solved"   : 13,
    "login_frequency"  : 25,
    "missed_deadlines" : 1,
    "retries"          : 2,
    "revision_sessions": 6,
    "time_per_task_min": 28.0,
}

result_2 = predict_student(student_2)
print("\n   📌 Student 2 — Strong Learner")
print(f"      Input  : {student_2['study_hours']}h study, {student_2['attendance_pct']}% attendance, {student_2['missed_deadlines']} missed deadlines")
print(f"      Score  : {result_2['predicted_score']} / 100")
print(f"      Status : {result_2['status']}")
print(f"      Advice : {result_2['advice']}")

# ── Example 3: An average student ─────────────────────────────────────────────
student_3 = {
    "study_hours"      : 4.0,
    "attendance_pct"   : 73.0,
    "assignments_done" : 9,
    "quizzes_solved"   : 7,
    "login_frequency"  : 14,
    "missed_deadlines" : 3,
    "retries"          : 2,
    "revision_sessions": 3,
    "time_per_task_min": 32.0,
}

result_3 = predict_student(student_3)
print("\n   📌 Student 3 — Average Learner")
print(f"      Input  : {student_3['study_hours']}h study, {student_3['attendance_pct']}% attendance, {student_3['missed_deadlines']} missed deadlines")
print(f"      Score  : {result_3['predicted_score']} / 100")
print(f"      Status : {result_3['status']}")
print(f"      Advice : {result_3['advice']}")


# ══════════════════════════════════════════════════════════════
# SAVE TEXT REPORT
# ══════════════════════════════════════════════════════════════
report_lines = [
    "=" * 60,
    "  PROJECT 1 — ML PIPELINE EVALUATION REPORT",
    "=" * 60,
    "",
    "MODEL COMPARISON",
    "-" * 40,
]
for name, r in results.items():
    marker = " ← BEST" if name == best_name else ""
    report_lines += [
        f"  {name}{marker}",
        f"    MAE  = {r['MAE']}",
        f"    RMSE = {r['RMSE']}",
        f"    R²   = {r['R2']}",
        f"    CV R²= {r['CV_R2']}",
        "",
    ]

report_lines += [
    "BEST MODEL",
    "-" * 40,
    f"  Name : {best_name}",
    f"  R²   : {best['R2']} (explains {best['R2']*100:.1f}% of score variation)",
    f"  MAE  : {best['MAE']} (average prediction error = {best['MAE']:.1f} points)",
    "",
    "FEATURE ENGINEERING ADDED",
    "-" * 40,
    "  • engagement_score  = login × 0.3 + assignments × 0.4 + quizzes × 0.3",
    "  • submission_rate   = assignments / (assignments + missed + 1)",
    "  • study_efficiency  = study_hours × (1 + revision_sessions × 0.1)",
    "",
    "SAVED FILES",
    "-" * 40,
    "  models/trained_model.pkl",
    "  models/scaler.pkl",
    "  models/feature_names.pkl",
    "",
    "NEXT STEPS",
    "-" * 40,
    "  → Project 2: Focus on regression depth & interpretation",
    "  → Project 3: Classify learner_status with these same features",
    "  → Project 7: Load this saved model in the final platform",
    "=" * 60,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\n\n📄  Report saved → {report_path}")

# ── DONE ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  PROJECT 1 COMPLETE!")
print(f"  Charts  → outputs/")
print(f"  Model   → models/trained_model.pkl")
print(f"  Report  → outputs/evaluation_report.txt")
print("=" * 60)
