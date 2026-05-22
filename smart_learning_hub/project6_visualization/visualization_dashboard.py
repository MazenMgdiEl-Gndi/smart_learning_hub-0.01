"""
PROJECT 6 — Visualization + Storytelling Dashboard
Generates a 4-part visual report saved as PNG files:
  Part 1: Dataset overview
  Part 2: Pattern exploration
  Part 3: Model performance
  Part 4: Actionable insights

Run: python visualization_dashboard.py
(Requires Projects 2 & 3 models to be trained first)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/student_data.csv")
REG_MODEL  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/regression_model.joblib")
CLF_MODEL  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/classification_model.joblib")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

PALETTE = {
    "at_risk"       : "#E74C3C",
    "average"       : "#F39C12",
    "strong_learner": "#27AE60",
    "primary"       : "#2C3E50",
    "secondary"     : "#3498DB",
    "light"         : "#ECF0F1",
}

df = pd.read_csv(DATA_PATH)
print(f"📂 Loaded {len(df)} students")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Dataset Overview (2x2)
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Part 1: Who Are Our Students?", fontsize=16, fontweight="bold",
              color=PALETTE["primary"], y=1.01)

# 1A — Risk distribution
ax = axes[0, 0]
counts = df["risk_label"].value_counts()
colors = [PALETTE[l] for l in counts.index]
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Learner Risk Distribution", fontweight="bold")
ax.set_ylabel("Number of Students")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val}\n({val/len(df)*100:.0f}%)", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, counts.max() * 1.2)
ax.spines[["top","right"]].set_visible(False)

# 1B — Score distribution histogram
ax = axes[0, 1]
for label, color in PALETTE.items():
    if label in ["at_risk","average","strong_learner"]:
        subset = df[df["risk_label"]==label]["average_score"]
        ax.hist(subset, bins=20, alpha=0.7, color=color, label=label.replace("_"," ").title())
ax.axvline(df["average_score"].mean(), color=PALETTE["primary"], linestyle="--",
           linewidth=2, label=f"Mean={df['average_score'].mean():.1f}")
ax.set_title("Score Distribution by Risk Group", fontweight="bold")
ax.set_xlabel("Average Score")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

# 1C — Study hours boxplot by risk
ax = axes[1, 0]
groups = [df[df["risk_label"]==l]["study_hours"].dropna().values
          for l in ["at_risk","average","strong_learner"]]
bp = ax.boxplot(groups, patch_artist=True,
                medianprops={"color":"white","linewidth":2})
for patch, color in zip(bp["boxes"],
                        [PALETTE["at_risk"],PALETTE["average"],PALETTE["strong_learner"]]):
    patch.set_facecolor(color)
ax.set_xticklabels(["At Risk","Average","Strong Learner"])
ax.set_title("Study Hours by Risk Group", fontweight="bold")
ax.set_ylabel("Daily Study Hours")
ax.spines[["top","right"]].set_visible(False)

# 1D — Test prep effect
ax = axes[1, 1]
prep_means = df.groupby(["test_preparation","risk_label"])["average_score"].mean().unstack()
prep_means.plot(kind="bar", ax=ax, color=[PALETTE["at_risk"],PALETTE["average"],PALETTE["strong_learner"]],
                edgecolor="white", rot=0)
ax.set_title("Test Prep vs Avg Score by Risk Group", fontweight="bold")
ax.set_xlabel("Test Preparation")
ax.set_ylabel("Average Score")
ax.legend(title="Risk Label", fontsize=8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "fig1_dataset_overview.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: fig1_dataset_overview.png")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Pattern Exploration
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Part 2: What Patterns Drive Performance?", fontsize=16,
              fontweight="bold", color=PALETTE["primary"])

# 2A — Correlation heatmap (numeric only)
import matplotlib.colors as mcolors
numeric_df = df[["study_hours","attendance_pct","assignments_done",
                  "login_days","late_submissions","engagement_score","average_score"]].dropna()
corr = numeric_df.corr()
ax = axes[0]
cmap = plt.cm.RdYlGn
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(corr.iloc[i,j]) < 0.7 else "white")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature Correlation Heatmap", fontweight="bold")

# 2B — Scatter: study hours vs score
ax = axes[1]
for label, color in [("at_risk",PALETTE["at_risk"]),
                      ("average",PALETTE["average"]),
                      ("strong_learner",PALETTE["strong_learner"])]:
    sub = df[df["risk_label"]==label]
    ax.scatter(sub["study_hours"], sub["average_score"], c=color, alpha=0.5,
               s=20, label=label.replace("_"," ").title())
ax.set_xlabel("Study Hours / Day")
ax.set_ylabel("Average Score")
ax.set_title("Study Hours vs Score", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# 2C — Attendance vs assignments by risk
ax = axes[2]
for label, color in [("at_risk",PALETTE["at_risk"]),
                      ("average",PALETTE["average"]),
                      ("strong_learner",PALETTE["strong_learner"])]:
    sub = df[df["risk_label"]==label]
    ax.scatter(sub["attendance_pct"], sub["assignments_done"], c=color, alpha=0.5,
               s=20, label=label.replace("_"," ").title())
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Assignments Done (%)")
ax.set_title("Attendance vs Assignment Completion", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig2_patterns.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: fig2_patterns.png")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Model Results (if models exist)
# ─────────────────────────────────────────────────────────────────────────────
models_exist = os.path.exists(REG_MODEL) and os.path.exists(CLF_MODEL)
if models_exist:
    NUMERIC_FEATURES     = ["study_hours","attendance_pct","assignments_done",
                             "login_days","late_submissions","quiz_retries","engagement_score"]
    CATEGORICAL_FEATURES = ["gender","parental_education","lunch","test_preparation"]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_reg = df["average_score"]
    y_clf = df["risk_label"]

    _, X_test, _, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _,      _, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42,
                                                stratify=y_clf)

    reg_pipe = joblib.load(REG_MODEL)
    clf_pipe = joblib.load(CLF_MODEL)
    reg_preds = reg_pipe.predict(X_test)
    clf_preds = clf_pipe.predict(X_test)

    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Part 3: What Do Our Models Predict?", fontsize=16,
                  fontweight="bold", color=PALETTE["primary"])

    # 3A — Regression: actual vs predicted
    ax = axes[0]
    ax.scatter(y_reg_test, reg_preds, alpha=0.4, color=PALETTE["secondary"], s=20)
    lims = [min(y_reg_test.min(), reg_preds.min()), max(y_reg_test.max(), reg_preds.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Regression: Actual vs Predicted Score", fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # 3B — Confusion matrix
    ax = axes[1]
    labels = sorted(y_clf_test.unique())
    cm = confusion_matrix(y_clf_test, clf_preds, labels=labels)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short_labels = [l.replace("_"," ").title() for l in labels]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Classification: Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "fig3_model_results.png"), dpi=150, bbox_inches="tight")
    print("✅ Saved: fig3_model_results.png")
    plt.close(fig3)
else:
    print("⚠️  Models not found — skipping Figure 3. Run Projects 2 & 3 first.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Actionable Insights
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle("Part 4: What Should Students Do Next?", fontsize=16,
              fontweight="bold", color=PALETTE["primary"])

# 4A — Average metrics by risk group
ax = axes[0]
metrics = ["study_hours","attendance_pct","assignments_done","engagement_score"]
labels_order = ["at_risk","average","strong_learner"]
x = np.arange(len(metrics))
width = 0.25
for i, (label, color) in enumerate(zip(labels_order, [PALETTE["at_risk"],
                                                       PALETTE["average"],
                                                       PALETTE["strong_learner"]])):
    means = [df[df["risk_label"]==label][m].mean() for m in metrics]
    # Normalize for visual comparison
    norms = [means[0]/12*100, means[1], means[2], means[3]*100]
    ax.bar(x + i*width, norms, width, label=label.replace("_"," ").title(),
           color=color, alpha=0.85, edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels(["Study Hours\n(/12h norm'd)","Attendance\n(%)","Assignments\n(%)","Engagement\n(×100)"],
                   fontsize=9)
ax.set_ylabel("Normalized Value")
ax.set_title("Avg Behavior Metrics by Risk Group", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

# 4B — Improvement simulation: what if at-risk students improved?
ax = axes[1]
at_risk_df = df[df["risk_label"]=="at_risk"].copy()
if len(at_risk_df) > 0 and models_exist:
    current_scores = reg_pipe.predict(at_risk_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    improved = at_risk_df.copy()
    improved["study_hours"]      = (improved["study_hours"].fillna(3) + 2).clip(0,12)
    improved["attendance_pct"]   = (improved["attendance_pct"].fillna(60) + 10).clip(0,100)
    improved["assignments_done"] = (improved["assignments_done"].fillna(60) + 15).clip(0,100)
    improved["engagement_score"] = (
        0.4*(improved["study_hours"]/12) + 0.3*(improved["attendance_pct"]/100) +
        0.2*(improved["assignments_done"]/100) + 0.1*(improved["login_days"].fillna(30)/90)
    )
    improved_scores = reg_pipe.predict(improved[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    ax.hist(current_scores, bins=15, alpha=0.6, color=PALETTE["at_risk"],
            label=f"Current (mean={current_scores.mean():.1f})")
    ax.hist(improved_scores, bins=15, alpha=0.6, color=PALETTE["strong_learner"],
            label=f"After Improvement (mean={improved_scores.mean():.1f})")
    ax.set_xlabel("Predicted Average Score")
    ax.set_ylabel("Count")
    ax.set_title("At-Risk Students: Current vs Simulated Improvement", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
else:
    ax.text(0.5, 0.5, "No at-risk students\nor models not loaded",
            ha="center", va="center", transform=ax.transAxes, fontsize=14)

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "fig4_insights.png"), dpi=150, bbox_inches="tight")
print("✅ Saved: fig4_insights.png")
plt.close(fig4)

print("\n✅ Project 6 complete! All charts saved.")
print("   fig1_dataset_overview.png")
print("   fig2_patterns.png")
print("   fig3_model_results.png  (requires trained models)")
print("   fig4_insights.png       (requires trained models)")
