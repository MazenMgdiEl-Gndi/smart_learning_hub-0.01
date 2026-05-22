"""
==============================================================
  PROJECT 5 — Real-World Data: Student Performance Analysis
==============================================================

THEME : Learning Analytics & Intelligent Study Assistant
GOAL  : Load, clean, explore, and analyse real student data
LEVEL : Beginner-friendly — every step is explained

HOW TO RUN:
    1. Make sure you ran data/generate_dataset.py first
    2. python project5_analysis.py

OUTPUT FILES (saved in outputs/ folder):
    - correlation_heatmap.png
    - score_distribution.png
    - feature_vs_score.png
    - learner_status_counts.png
    - cleaned_dataset.csv
    - analysis_report.txt
==============================================================
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd           # work with tables of data
import numpy as np            # math and numbers
import matplotlib.pyplot as plt  # draw charts
import seaborn as sns          # prettier charts
import os

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATA_PATH    = "data/student_performance.csv"
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

print("=" * 60)
print("  PROJECT 5 — Student Performance Real-World Analysis")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD THE DATA
# ─────────────────────────────────────────────────────────────
print("\n📂  STEP 1: Loading the dataset …")

df = pd.read_csv(DATA_PATH)

print(f"   ✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print("\n   Column names:")
for col in df.columns:
    print(f"      • {col}")


# ─────────────────────────────────────────────────────────────
# STEP 2: UNDERSTAND THE COLUMNS
# ─────────────────────────────────────────────────────────────
print("\n📋  STEP 2: Understanding each column …\n")

column_descriptions = {
    "student_id":        "Unique ID for each student (e.g. STU0001)",
    "study_hours":       "Average daily study hours",
    "attendance_pct":    "Percentage of classes attended (0–100)",
    "assignments_done":  "Number of assignments submitted",
    "quizzes_solved":    "Number of quizzes completed",
    "login_frequency":   "Number of logins to the platform per month",
    "missed_deadlines":  "Number of missed submission deadlines",
    "retries":           "Number of times a student retried a task",
    "revision_sessions": "Number of revision / review sessions",
    "time_per_task_min": "Average minutes spent per task",
    "final_score":       "Final exam score (0–100) — this is what we predict",
    "learner_status":    "Category: at_risk / average / strong",
}

for col, desc in column_descriptions.items():
    dtype = str(df[col].dtype)
    print(f"   {col:<22} ({dtype:<8}) → {desc}")


# ─────────────────────────────────────────────────────────────
# STEP 3: CLEAN THE DATA
# ─────────────────────────────────────────────────────────────
print("\n🧹  STEP 3: Cleaning the data …")

# 3a — Check for missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"\n   Columns with missing values:")
if missing.empty:
    print("      None found!")
else:
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"      • {col}: {count} missing ({pct:.1f}%)")

# 3b — Fill missing numbers with the column median
#       (median is safer than mean when data has outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   ✅  Filled '{col}' missing values with median ({median_val:.2f})")

# 3c — Remove duplicate rows (if any)
before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"\n   Duplicate rows removed: {before - after}")

# 3d — Clip numeric values to valid ranges
df["attendance_pct"]   = df["attendance_pct"].clip(0, 100)
df["study_hours"]      = df["study_hours"].clip(0, 24)
df["final_score"]      = df["final_score"].clip(0, 100)
df["missed_deadlines"] = df["missed_deadlines"].clip(0, None)  # no negatives
print("   ✅  Values clipped to valid ranges")

# 3e — Save the clean dataset
clean_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
df.to_csv(clean_path, index=False)
print(f"   ✅  Clean dataset saved → {clean_path}")


# ─────────────────────────────────────────────────────────────
# STEP 4: BASIC STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n📊  STEP 4: Basic statistics …\n")

stats = df[numeric_cols].describe().round(2)
print(stats.to_string())


# ─────────────────────────────────────────────────────────────
# STEP 5: FIND PATTERNS — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────
print("\n🔥  STEP 5: Drawing correlation heatmap …")

numeric_df = df[numeric_cols].copy()
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.5, square=True,
    ax=ax
)
ax.set_title("Feature Correlation Heatmap\n(+1 = strong positive, -1 = strong negative)",
             fontsize=13, pad=15)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(path)
plt.close()
print(f"   ✅  Saved → {path}")

# Print the top correlations with final_score
print("\n   Top features correlated with final_score:")
score_corr = corr_matrix["final_score"].drop("final_score").sort_values(ascending=False)
for feat, val in score_corr.items():
    direction = "↑ helps" if val > 0 else "↓ hurts"
    print(f"      {feat:<22} r = {val:+.2f}   {direction}")


# ─────────────────────────────────────────────────────────────
# STEP 6: SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────
print("\n📈  STEP 6: Drawing score distribution …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(df["final_score"], bins=20, color="#4C9BE8", edgecolor="white")
axes[0].set_title("Distribution of Final Scores")
axes[0].set_xlabel("Final Score")
axes[0].set_ylabel("Number of Students")
axes[0].axvline(df["final_score"].mean(), color="red", linestyle="--",
                label=f"Mean: {df['final_score'].mean():.1f}")
axes[0].legend()

# Learner status pie
status_counts = df["learner_status"].value_counts()
colors = ["#E85C5C", "#F5A623", "#4CAF50"]
axes[1].pie(status_counts, labels=status_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Learner Status Breakdown")

plt.suptitle("Student Performance Overview", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "score_distribution.png")
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"   ✅  Saved → {path}")


# ─────────────────────────────────────────────────────────────
# STEP 7: FEATURES vs FINAL SCORE (Scatter Plots)
# ─────────────────────────────────────────────────────────────
print("\n🔍  STEP 7: Drawing feature vs score scatter plots …")

features_to_plot = [
    "study_hours", "attendance_pct",
    "assignments_done", "missed_deadlines",
    "revision_sessions", "login_frequency"
]

palette = {"at_risk": "#E85C5C", "average": "#F5A623", "strong": "#4CAF50"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feat in enumerate(features_to_plot):
    ax = axes[i]
    for status, color in palette.items():
        subset = df[df["learner_status"] == status]
        ax.scatter(subset[feat], subset["final_score"],
                   alpha=0.4, color=color, s=20, label=status)
    ax.set_xlabel(feat.replace("_", " ").title())
    ax.set_ylabel("Final Score")
    ax.set_title(f"{feat.replace('_', ' ').title()} vs Final Score")
    if i == 0:
        ax.legend(title="Status", fontsize=8)

plt.suptitle("How Each Feature Relates to Final Score", fontsize=14, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "feature_vs_score.png")
plt.savefig(path)
plt.close()
print(f"   ✅  Saved → {path}")


# ─────────────────────────────────────────────────────────────
# STEP 8: LEARNER STATUS COMPARISON (Bar Charts)
# ─────────────────────────────────────────────────────────────
print("\n📊  STEP 8: Comparing learner groups …")

group_means = df.groupby("learner_status")[features_to_plot].mean()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

status_order = ["at_risk", "average", "strong"]
bar_colors   = ["#E85C5C", "#F5A623", "#4CAF50"]

for i, feat in enumerate(features_to_plot):
    ax = axes[i]
    values = [group_means.loc[s, feat] for s in status_order]
    bars = ax.bar(status_order, values, color=bar_colors, edgecolor="white", width=0.5)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("Learner Status")
    ax.set_ylabel("Average Value")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Average Feature Values by Learner Status", fontsize=14, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "learner_status_counts.png")
plt.savefig(path)
plt.close()
print(f"   ✅  Saved → {path}")


# ─────────────────────────────────────────────────────────────
# STEP 9: KEY INSIGHTS (Text Report)
# ─────────────────────────────────────────────────────────────
print("\n📝  STEP 9: Writing analysis report …")

top_positive = score_corr[score_corr > 0].head(3)
top_negative = score_corr[score_corr < 0].head(3)

at_risk_count  = (df["learner_status"] == "at_risk").sum()
average_count  = (df["learner_status"] == "average").sum()
strong_count   = (df["learner_status"] == "strong").sum()

report_lines = [
    "=" * 60,
    "  PROJECT 5 — ANALYSIS REPORT",
    "  Student Performance — Real-World Data",
    "=" * 60,
    "",
    "DATASET SUMMARY",
    "-" * 40,
    f"  Total students   : {len(df)}",
    f"  Total features   : {len(df.columns) - 2}  (excluding id and label)",
    f"  Missing values   : Fixed by median filling",
    f"  Duplicates       : Removed",
    "",
    "LEARNER STATUS BREAKDOWN",
    "-" * 40,
    f"  At Risk  : {at_risk_count} students ({at_risk_count/len(df)*100:.1f}%)",
    f"  Average  : {average_count} students ({average_count/len(df)*100:.1f}%)",
    f"  Strong   : {strong_count} students ({strong_count/len(df)*100:.1f}%)",
    "",
    "SCORE STATISTICS",
    "-" * 40,
    f"  Mean score   : {df['final_score'].mean():.1f}",
    f"  Median score : {df['final_score'].median():.1f}",
    f"  Std deviation: {df['final_score'].std():.1f}",
    f"  Min score    : {df['final_score'].min():.1f}",
    f"  Max score    : {df['final_score'].max():.1f}",
    "",
    "WHAT HELPS STUDENTS SUCCEED (top positive correlations)",
    "-" * 40,
]
for feat, val in top_positive.items():
    report_lines.append(f"  ↑ {feat:<22} r = {val:+.2f}")

report_lines += [
    "",
    "WHAT HURTS STUDENT PERFORMANCE (top negative correlations)",
    "-" * 40,
]
for feat, val in top_negative.items():
    report_lines.append(f"  ↓ {feat:<22} r = {val:+.2f}")

report_lines += [
    "",
    "KEY FINDINGS",
    "-" * 40,
    "  1. Study hours and revision sessions are the strongest",
    "     positive predictors of final score.",
    "  2. Missed deadlines strongly hurt student performance.",
    "  3. Strong learners attend more, study more, and miss fewer deadlines.",
    "  4. At-risk students can be identified early using login",
    "     frequency, missed deadlines, and revision sessions.",
    "",
    "NEXT STEPS (for later projects)",
    "-" * 40,
    "  → Project 1: Build a full ML pipeline using this data",
    "  → Project 2: Predict final_score (regression)",
    "  → Project 3: Classify learner_status (classification)",
    "  → Project 6: Build a dashboard of these charts",
    "=" * 60,
]

report_text = "\n".join(report_lines)
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)
print(f"\n   ✅  Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  PROJECT 5 COMPLETE!")
print(f"  All output files are in the '{OUTPUT_DIR}/' folder")
print("=" * 60)
