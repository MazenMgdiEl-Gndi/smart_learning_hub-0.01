"""
==============================================================
  PROJECT 7 — Smart Learning Analytics Hub
  الدمج الكامل لكل المشاريع في نظام واحد
==============================================================

ده المشروع الرئيسي اللي بيشغّل كل حاجة بالترتيب الصح:

  ✅ Project 5 → تحليل البيانات الحقيقية
  ✅ Project 1 → Pipeline كامل من البيانات للـ Model
  ✅ Project 2 → Regression: توقع الدرجة الرقمية
  ✅ Project 3 → Classification: تصنيف مستوى الطالب
  ✅ Project 4 → LLM Agent: مساعد ذكي بالـ AI
  ✅ Project 6 → Visualization: رسوم بيانية احترافية

HOW TO RUN:
  python run_all.py

OUTPUT:
  كل الملفات الناتجة هتتحفظ في فولدر outputs/
==============================================================
"""

import os
import sys
import subprocess
import time

# ── إعداد المسارات ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def section(title):
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)

def step(msg):
    print(f"\n  ▶  {msg}")

def ok(msg):
    print(f"  ✅  {msg}")

def warn(msg):
    print(f"  ⚠️   {msg}")


# ══════════════════════════════════════════════════════════════
# STEP 0 — توليد البيانات (لو مش موجودة)
# ══════════════════════════════════════════════════════════════
section("STEP 0 — توليد بيانات الطلاب")

data_path = os.path.join(BASE_DIR, "data", "student_performance.csv")

if os.path.exists(data_path):
    ok(f"البيانات موجودة بالفعل → {data_path}")
else:
    step("جاري توليد بيانات الطلاب ...")
    import subprocess
    gen_script = os.path.join(BASE_DIR, "data", "generate_dataset.py")
    # Run generate_dataset with correct working directory
    result = subprocess.run(
        [sys.executable, gen_script],
        cwd=os.path.join(BASE_DIR, "data"),
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ok("تم توليد البيانات بنجاح")
        print(result.stdout)
    else:
        print("ERROR:", result.stderr)
        sys.exit(1)

# تأكد إن فولدر outputs موجود
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)


# ══════════════════════════════════════════════════════════════
# دالة مساعدة لتشغيل أي مشروع
# ══════════════════════════════════════════════════════════════
def run_project(script_path, working_dir, project_name):
    """تشغيل مشروع معين وإظهار النتيجة."""
    step(f"تشغيل {project_name} ...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=working_dir,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        ok(f"{project_name} اكتمل في {elapsed:.1f} ثانية")
        # اطبع آخر 8 سطور من الـ output
        lines = result.stdout.strip().split("\n")
        for line in lines[-8:]:
            print(f"     {line}")
    else:
        warn(f"{project_name} فشل!")
        print(result.stderr[-500:])
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════
# PROJECT 5 — تحليل البيانات الحقيقية
# ══════════════════════════════════════════════════════════════
section("PROJECT 5 — Real-World Data Analysis")

# المشروع 5 بيقرأ من data/ وبيحفظ في outputs/
# محتاجين نشغله من فولدر الـ project5_real_data
# لكن هيحاول يوصل لـ data/ relative — نعدّل المسار

import importlib.util, types

def load_and_run_p5():
    """تحميل وتشغيل project5_analysis.py مع تعديل المسارات."""
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    DATA_PATH  = os.path.join(BASE_DIR, "data", "student_performance.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    df = pd.read_csv(DATA_PATH)
    step(f"بيانات المشروع 5: {df.shape[0]} طالب, {df.shape[1]} عمود")

    # تنظيف
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df.drop_duplicates(inplace=True)
    df["attendance_pct"]   = df["attendance_pct"].clip(0, 100)
    df["study_hours"]      = df["study_hours"].clip(0, 24)
    df["final_score"]      = df["final_score"].clip(0, 100)
    df["missed_deadlines"] = df["missed_deadlines"].clip(0, None)

    # حفظ البيانات المنظفة
    clean_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
    df.to_csv(clean_path, index=False)

    # Correlation heatmap
    sns.set_theme(style="whitegrid", font_scale=1.0)
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Correlation Heatmap — Project 5", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p5_correlation_heatmap.png"), dpi=120)
    plt.close()

    # Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["final_score"], bins=20, color="#4C9BE8", edgecolor="white")
    axes[0].axvline(df["final_score"].mean(), color="red", linestyle="--",
                    label=f"Mean: {df['final_score'].mean():.1f}")
    axes[0].set_title("Score Distribution")
    axes[0].set_xlabel("Final Score")
    axes[0].legend()
    status_counts = df["learner_status"].value_counts()
    axes[1].pie(status_counts, labels=status_counts.index,
                autopct="%1.1f%%", colors=["#E85C5C","#F5A623","#4CAF50"],
                wedgeprops={"edgecolor":"white"})
    axes[1].set_title("Learner Status Breakdown")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p5_score_distribution.png"), dpi=120)
    plt.close()

    ok("Project 5 اكتملت الرسوم البيانية")
    return df

p5_df = load_and_run_p5()


# ══════════════════════════════════════════════════════════════
# PROJECT 1 — End-to-End ML Pipeline
# ══════════════════════════════════════════════════════════════
section("PROJECT 1 — End-to-End ML Pipeline")

def run_p1(df_clean):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.linear_model    import LinearRegression
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree            import DecisionTreeRegressor
    from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    df = df_clean.copy()

    # Feature Engineering
    df["engagement_score"] = (
        df["login_frequency"] * 0.3 +
        df["assignments_done"] * 0.4 +
        df["quizzes_solved"] * 0.3
    ).round(2)
    df["submission_rate"] = (
        df["assignments_done"] / (df["assignments_done"] + df["missed_deadlines"] + 1)
    ).round(3)
    df["study_efficiency"] = (
        df["study_hours"] * (1 + df["revision_sessions"] * 0.1)
    ).round(2)

    FEATURE_COLS = [
        "study_hours", "attendance_pct", "assignments_done",
        "quizzes_solved", "login_frequency", "missed_deadlines",
        "retries", "revision_sessions", "time_per_task_min",
        "engagement_score", "submission_rate", "study_efficiency",
    ]
    TARGET_COL = "final_score"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Linear Regression"  : LinearRegression(),
        "Decision Tree"      : DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        results[name] = {"model": model, "y_pred": y_pred,
                         "R2": round(r2,3), "MAE": round(mae,3), "RMSE": round(rmse,3)}
        print(f"     {name:<25} R²={r2:.3f}  MAE={mae:.2f}")

    best_name = max(results, key=lambda k: results[k]["R2"])
    best = results[best_name]
    ok(f"أفضل موديل: {best_name}  (R²={best['R2']})")

    # حفظ الموديل
    joblib.dump(best["model"],   os.path.join(MODELS_DIR, "p1_model.pkl"))
    joblib.dump(scaler,          os.path.join(MODELS_DIR, "p1_scaler.pkl"))
    joblib.dump(FEATURE_COLS,    os.path.join(MODELS_DIR, "p1_features.pkl"))

    # رسم actual vs predicted
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, best["y_pred"], alpha=0.4, color="#4C9BE8", s=30)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn,mx],[mn,mx],"r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title(f"Project 1 — Actual vs Predicted\n{best_name}  R²={best['R2']}")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p1_actual_vs_predicted.png"), dpi=120)
    plt.close()

    return best_name, FEATURE_COLS

p1_best, p1_features = run_p1(p5_df)


# ══════════════════════════════════════════════════════════════
# PROJECT 2 — Regression (توقع الدرجة)
# ══════════════════════════════════════════════════════════════
section("PROJECT 2 — Regression: Predict Final Score")

def run_p2(df_clean):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler, OneHotEncoder
    from sklearn.impute          import SimpleImputer
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model    import LinearRegression
    from sklearn.tree            import DecisionTreeRegressor
    from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline        import Pipeline
    from sklearn.compose         import ColumnTransformer

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    df = df_clean.copy()
    df["engagement_score"] = (
        df["login_frequency"]*0.3 + df["assignments_done"]*0.4 + df["quizzes_solved"]*0.3
    ).round(2)
    df["submission_rate"] = (
        df["assignments_done"] / (df["assignments_done"] + df["missed_deadlines"] + 1)
    ).round(3)
    df["study_efficiency"] = (
        df["study_hours"] * (1 + df["revision_sessions"] * 0.1)
    ).round(2)

    NUM_FEATS = ["study_hours","attendance_pct","assignments_done","quizzes_solved",
                 "login_frequency","missed_deadlines","retries","revision_sessions",
                 "time_per_task_min","engagement_score","submission_rate","study_efficiency"]
    TARGET = "final_score"

    X = df[NUM_FEATS]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])

    models = {
        "Linear Regression" : LinearRegression(),
        "Decision Tree"     : DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    print(f"\n     {'Model':<24} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("     " + "-"*50)
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = mean_squared_error(y_test, preds)**0.5
        r2    = r2_score(y_test, preds)
        results[name] = {"pipe":pipe, "mae":mae, "rmse":rmse, "r2":r2}
        print(f"     {name:<24} {mae:>7.2f} {rmse:>7.2f} {r2:>7.3f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best = results[best_name]
    ok(f"أفضل موديل: {best_name}  (R²={best['r2']:.3f})")

    joblib.dump(best["pipe"], os.path.join(MODELS_DIR, "p2_regression.joblib"))
    ok(f"تم حفظ الموديل → models/p2_regression.joblib")
    return best["pipe"], NUM_FEATS

p2_pipe, p2_features = run_p2(p5_df)


# ══════════════════════════════════════════════════════════════
# PROJECT 3 — Classification (تصنيف مستوى الطالب)
# ══════════════════════════════════════════════════════════════
section("PROJECT 3 — Classification: Learner Status")

def run_p3(df_clean):
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.impute          import SimpleImputer
    from sklearn.linear_model    import LogisticRegression
    from sklearn.tree            import DecisionTreeClassifier
    from sklearn.ensemble        import RandomForestClassifier
    from sklearn.neighbors       import KNeighborsClassifier
    from sklearn.metrics         import accuracy_score, f1_score, classification_report
    from sklearn.pipeline        import Pipeline

    MODELS_DIR = os.path.join(BASE_DIR, "models")

    df = df_clean.copy()
    df["engagement_score"] = (
        df["login_frequency"]*0.3 + df["assignments_done"]*0.4 + df["quizzes_solved"]*0.3
    ).round(2)
    df["submission_rate"] = (
        df["assignments_done"] / (df["assignments_done"] + df["missed_deadlines"] + 1)
    ).round(3)
    df["study_efficiency"] = (
        df["study_hours"] * (1 + df["revision_sessions"] * 0.1)
    ).round(2)

    NUM_FEATS = ["study_hours","attendance_pct","assignments_done","quizzes_solved",
                 "login_frequency","missed_deadlines","retries","revision_sessions",
                 "time_per_task_min","engagement_score","submission_rate","study_efficiency"]
    TARGET = "learner_status"

    X = df[NUM_FEATS]; y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)"           : KNeighborsClassifier(n_neighbors=5),
    }

    print(f"\n     {'Model':<24} {'Accuracy':>9} {'F1':>7}")
    print("     " + "-"*43)
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        results[name] = {"pipe":pipe, "acc":acc, "f1":f1}
        print(f"     {name:<24} {acc:>9.3f} {f1:>7.3f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    ok(f"أفضل موديل: {best_name}  (F1={best['f1']:.3f})")

    joblib.dump(best["pipe"], os.path.join(MODELS_DIR, "p3_classifier.joblib"))
    ok(f"تم حفظ الموديل → models/p3_classifier.joblib")
    return best["pipe"], NUM_FEATS

p3_pipe, p3_features = run_p3(p5_df)


# ══════════════════════════════════════════════════════════════
# PROJECT 4 — LLM Agent (مساعد ذكي)
# ══════════════════════════════════════════════════════════════
section("PROJECT 4 — LLM Study Agent")

def run_p4_demo():
    """تشغيل ديمو للـ Agent بدون API key (mock mode)."""
    import json

    MOCK_RESPONSE = {
        "summary": "الطالب قضى ساعتين في تعلم Python وفهم المتغيرات والدوال بشكل جيد، لكنه واجه صعوبة في الحلقات التكرارية.",
        "strengths": ["أكمل فصلي المتغيرات والدوال", "استمر في الدراسة لمدة ساعتين كاملتين"],
        "weaknesses": ["صعوبة في for loops وrange()", "لم يكمل كل التمارين"],
        "next_session_plan": {
            "focus_topic": "For loops والحلقات التكرارية في Python",
            "recommended_activities": [
                "شاهد فيديو مخصص عن Python loops على YouTube",
                "حل 5 تمارين على HackerRank Easy",
                "أعد كتابة التمارين السابقة باستخدام الحلقات"
            ],
            "estimated_time_minutes": 60
        },
        "motivation_message": "صعوبة الحلقات طبيعية جداً — جلسة واحدة مركزة كافية لتكسير هذا الحاجز!"
    }

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        step("تم العثور على ANTHROPIC_API_KEY — سيتم استخدام الـ API الحقيقي")
    else:
        step("لا يوجد API Key — تشغيل في وضع Mock Demo")

    step("مثال على رد الـ Agent:")
    print()
    print(f"     📝 ملخص: {MOCK_RESPONSE['summary']}")
    print(f"\n     ✅ نقاط القوة:")
    for s in MOCK_RESPONSE["strengths"]:
        print(f"        • {s}")
    print(f"\n     ⚠️  نقاط التحسين:")
    for w in MOCK_RESPONSE["weaknesses"]:
        print(f"        • {w}")
    plan = MOCK_RESPONSE["next_session_plan"]
    print(f"\n     📅 الجلسة القادمة ({plan['estimated_time_minutes']} دقيقة):")
    print(f"        موضوع: {plan['focus_topic']}")
    for act in plan["recommended_activities"]:
        print(f"        → {act}")
    print(f"\n     💬 {MOCK_RESPONSE['motivation_message']}")
    ok("Project 4 جاهز — لتفعيل الـ AI الحقيقي: export ANTHROPIC_API_KEY=your_key")

run_p4_demo()


# ══════════════════════════════════════════════════════════════
# PROJECT 6 — Visualization Dashboard
# ══════════════════════════════════════════════════════════════
section("PROJECT 6 — Visualization Dashboard")

def run_p6(df_clean, p2_model, p3_model, feat_cols):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    PALETTE = {"at_risk":"#E74C3C", "average":"#F39C12", "strong":"#27AE60"}

    df = df_clean.copy()
    df["engagement_score"] = (
        df["login_frequency"]*0.3 + df["assignments_done"]*0.4 + df["quizzes_solved"]*0.3
    ).round(2)
    df["submission_rate"] = (
        df["assignments_done"] / (df["assignments_done"] + df["missed_deadlines"] + 1)
    ).round(3)
    df["study_efficiency"] = (
        df["study_hours"] * (1 + df["revision_sessions"] * 0.1)
    ).round(2)

    X = df[feat_cols]
    y_reg = df["final_score"]
    y_clf = df["learner_status"]
    _, X_test, _, y_rt = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _,      _, y_ct = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    reg_preds = p2_model.predict(X_test)
    clf_preds = p3_model.predict(X_test)

    # ── Figure 1: Dataset Overview ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dashboard — Part 1: Dataset Overview", fontsize=15, fontweight="bold")

    # Risk distribution
    ax = axes[0, 0]
    counts = df["learner_status"].value_counts()
    colors = [PALETTE.get(l, "#95A5A6") for l in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Learner Status Distribution"); ax.set_ylabel("Students")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                f"{v}\n({v/len(df)*100:.0f}%)", ha="center", fontsize=10)
    ax.set_ylim(0, counts.max()*1.2)
    ax.spines[["top","right"]].set_visible(False)

    # Score histogram
    ax = axes[0, 1]
    for label, color in PALETTE.items():
        sub = df[df["learner_status"]==label]["final_score"]
        ax.hist(sub, bins=15, alpha=0.7, color=color, label=label)
    ax.axvline(df["final_score"].mean(), color="#2C3E50", linestyle="--",
               label=f"Mean={df['final_score'].mean():.1f}")
    ax.set_title("Score Distribution by Group"); ax.set_xlabel("Final Score")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    # Study hours boxplot
    ax = axes[1, 0]
    groups = [df[df["learner_status"]==l]["study_hours"].dropna().values
              for l in ["at_risk","average","strong"]]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops={"color":"white","linewidth":2})
    for patch, color in zip(bp["boxes"], [PALETTE["at_risk"],PALETTE["average"],PALETTE["strong"]]):
        patch.set_facecolor(color)
    ax.set_xticklabels(["At Risk","Average","Strong"])
    ax.set_title("Study Hours by Group"); ax.set_ylabel("Hours/day")
    ax.spines[["top","right"]].set_visible(False)

    # Attendance vs score scatter
    ax = axes[1, 1]
    for label, color in PALETTE.items():
        sub = df[df["learner_status"]==label]
        ax.scatter(sub["attendance_pct"], sub["final_score"], c=color,
                   alpha=0.5, s=20, label=label)
    ax.set_xlabel("Attendance %"); ax.set_ylabel("Final Score")
    ax.set_title("Attendance vs Final Score")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p6_fig1_overview.png"), dpi=130, bbox_inches="tight")
    plt.close()
    ok("p6_fig1_overview.png")

    # ── Figure 2: Patterns ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dashboard — Part 2: Patterns", fontsize=15, fontweight="bold")

    # Correlation heatmap (subset)
    ax = axes[0]
    subset_cols = ["study_hours","attendance_pct","assignments_done",
                   "missed_deadlines","engagement_score","final_score"]
    corr = df[subset_cols].corr()
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(subset_cols))); ax.set_yticks(range(len(subset_cols)))
    ax.set_xticklabels([c.replace("_","\n") for c in subset_cols], fontsize=7)
    ax.set_yticklabels([c.replace("_"," ") for c in subset_cols], fontsize=7)
    for i in range(len(subset_cols)):
        for j in range(len(subset_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap")

    # Study hours vs score
    ax = axes[1]
    for label, color in PALETTE.items():
        sub = df[df["learner_status"]==label]
        ax.scatter(sub["study_hours"], sub["final_score"], c=color,
                   alpha=0.5, s=20, label=label)
    ax.set_xlabel("Study Hours/Day"); ax.set_ylabel("Final Score")
    ax.set_title("Study Hours vs Score"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    # Missed deadlines vs score
    ax = axes[2]
    for label, color in PALETTE.items():
        sub = df[df["learner_status"]==label]
        ax.scatter(sub["missed_deadlines"], sub["final_score"], c=color,
                   alpha=0.5, s=20, label=label)
    ax.set_xlabel("Missed Deadlines"); ax.set_ylabel("Final Score")
    ax.set_title("Missed Deadlines vs Score"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p6_fig2_patterns.png"), dpi=130, bbox_inches="tight")
    plt.close()
    ok("p6_fig2_patterns.png")

    # ── Figure 3: Model Results ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dashboard — Part 3: Model Results", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_rt, reg_preds, alpha=0.4, color="#3498DB", s=20)
    mn, mx = min(y_rt.min(), reg_preds.min()), max(y_rt.max(), reg_preds.max())
    ax.plot([mn,mx],[mn,mx],"r--", linewidth=2, label="Perfect")
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title("Regression — Actual vs Predicted"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    labels = sorted(y_ct.unique())
    cm = confusion_matrix(y_ct, clf_preds, labels=labels)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Classification — Confusion Matrix")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p6_fig3_models.png"), dpi=130, bbox_inches="tight")
    plt.close()
    ok("p6_fig3_models.png")

    # ── Figure 4: Insights ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dashboard — Part 4: Actionable Insights", fontsize=15, fontweight="bold")

    ax = axes[0]
    metrics = ["study_hours","attendance_pct","assignments_done","missed_deadlines"]
    status_order = ["at_risk","average","strong"]
    x = np.arange(len(metrics)); w = 0.25
    for i, (s, c) in enumerate(zip(status_order,
                                    [PALETTE["at_risk"],PALETTE["average"],PALETTE["strong"]])):
        means = [df[df["learner_status"]==s][m].mean() for m in metrics]
        norms = [means[0]/12*100, means[1], means[2]/20*100, max(0,100-means[3]*10)]
        ax.bar(x+i*w, norms, w, label=s, color=c, alpha=0.85, edgecolor="white")
    ax.set_xticks(x+w)
    ax.set_xticklabels(["Study\nHours","Attendance","Assignments","Deadline\nScore"], fontsize=9)
    ax.set_title("Behavior Metrics by Group (Normalized)"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    at_risk = df[df["learner_status"]=="at_risk"].copy()
    if len(at_risk) > 0:
        current = p2_model.predict(at_risk[feat_cols])
        improved = at_risk.copy()
        improved["study_hours"] = (improved["study_hours"]+2).clip(0,12)
        improved["attendance_pct"] = (improved["attendance_pct"]+10).clip(0,100)
        improved["missed_deadlines"] = (improved["missed_deadlines"]-2).clip(0,None)
        improved["engagement_score"] = (
            improved["login_frequency"]*0.3 + improved["assignments_done"]*0.4 + improved["quizzes_solved"]*0.3
        )
        improved["submission_rate"] = (
            improved["assignments_done"]/(improved["assignments_done"]+improved["missed_deadlines"]+1)
        )
        improved["study_efficiency"] = improved["study_hours"]*(1+improved["revision_sessions"]*0.1)
        improved_preds = p2_model.predict(improved[feat_cols])
        ax.hist(current, bins=15, alpha=0.6, color=PALETTE["at_risk"],
                label=f"الآن (mean={current.mean():.1f})")
        ax.hist(improved_preds, bins=15, alpha=0.6, color=PALETTE["strong"],
                label=f"بعد التحسين (mean={improved_preds.mean():.1f})")
        ax.set_xlabel("Predicted Score"); ax.set_ylabel("Count")
        ax.set_title("At-Risk: Current vs If Improved"); ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p6_fig4_insights.png"), dpi=130, bbox_inches="tight")
    plt.close()
    ok("p6_fig4_insights.png")

run_p6(p5_df, p2_pipe, p3_pipe, p2_features)


# ══════════════════════════════════════════════════════════════
# PROJECT 7 — النظام المدمج: تحليل طالب جديد
# ══════════════════════════════════════════════════════════════
section("PROJECT 7 — Smart Hub: Analyze a New Student")

def run_integrated_demo(p2_model, p3_model, feat_cols, df_all):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    PALETTE = {"at_risk":"#E74C3C", "average":"#F39C12", "strong":"#27AE60"}

    # ── بيانات طالب تجريبي ────────────────────────────────────────────────────
    student = {
        "study_hours"       : 3.0,
        "attendance_pct"    : 65.0,
        "assignments_done"  : 8,
        "quizzes_solved"    : 5,
        "login_frequency"   : 10,
        "missed_deadlines"  : 4,
        "retries"           : 2,
        "revision_sessions" : 2,
        "time_per_task_min" : 35.0,
    }
    student["engagement_score"] = (
        student["login_frequency"]*0.3 + student["assignments_done"]*0.4 + student["quizzes_solved"]*0.3
    )
    student["submission_rate"] = (
        student["assignments_done"] / (student["assignments_done"] + student["missed_deadlines"] + 1)
    )
    student["study_efficiency"] = student["study_hours"] * (1 + student["revision_sessions"]*0.1)

    import pandas as pd
    row = pd.DataFrame([student])[feat_cols]

    pred_score = round(float(p2_model.predict(row)[0]), 1)
    pred_score = max(0, min(100, pred_score))
    pred_label = p3_model.predict(row)[0]
    pred_proba = {c: round(float(p),3) for c, p in zip(p3_model.classes_, p3_model.predict_proba(row)[0])}

    # ── طباعة النتائج ────────────────────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              تحليل الطالب التجريبي                 │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  ساعات الدراسة     : {student['study_hours']} ساعة/يوم                  │")
    print(f"  │  نسبة الحضور       : {student['attendance_pct']}%                        │")
    print(f"  │  الواجبات المكتملة : {student['assignments_done']} / 20                       │")
    print(f"  │  الواجبات المتأخرة : {student['missed_deadlines']}                           │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  🎯 الدرجة المتوقعة : {pred_score} / 100                    │")
    print(f"  │  📊 التصنيف        : {pred_label.upper():<20}           │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  احتمالات التصنيف:                                  │")
    for label, prob in pred_proba.items():
        bar = "█" * int(prob*25)
        print(f"  │   {label:<10} {bar:<25} {prob:.0%}  │")
    print("  └─────────────────────────────────────────────────────┘")

    # ── نصائح مخصصة ─────────────────────────────────────────────────────────
    advice = {
        "at_risk": [
            "⚠️  أنت في منطقة الخطر — لكن يمكنك التحسن بسرعة",
            "   → زد ساعات الدراسة اليومية إلى 5 ساعات على الأقل",
            "   → حضور الكلاسات إلزامي — الحضور من أقوى المؤشرات",
            "   → أكمل كل الواجبات المتأخرة قبل البدء في مادة جديدة",
        ],
        "average": [
            "📈 أداؤك في المنطقة المتوسطة — قريب من المستوى الجيد",
            "   → ركز على رفع نسبة إكمال الواجبات فوق 80%",
            "   → جلسات المراجعة تحسّن الدرجة بشكل ملحوظ",
            "   → حافظ على انتظام الحضور",
        ],
        "strong": [
            "🏆 أداء ممتاز! أنت من الطلاب المتميزين",
            "   → استمر في نفس الروتين الدراسي",
            "   → يمكنك مساعدة زملائك في الفهم",
            "   → ابدأ المراجعة النهائية مبكراً للحفاظ على مستواك",
        ],
    }
    print()
    for line in advice.get(pred_label, []):
        print(f"  {line}")

    # ── Dashboard Chart ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Project 7 — Smart Learning Analytics Hub\nStudent Report",
                 fontsize=13, fontweight="bold", color="#2C3E50")

    # Panel 1: Student metrics bar
    ax = axes[0]
    metrics      = ["Study\nHours","Attendance","Assignments\nDone","Engagement"]
    student_vals = [
        student["study_hours"]/12*100,
        student["attendance_pct"],
        student["assignments_done"]/20*100,
        student["engagement_score"]/10*100,
    ]
    bar_colors = ["#27AE60" if v>=70 else "#F39C12" if v>=50 else "#E74C3C"
                  for v in student_vals]
    bars = ax.bar(metrics, student_vals, color=bar_colors, edgecolor="white")
    ax.axhline(70, color="#27AE60", linestyle="--", linewidth=1, alpha=0.7, label="Target 70%")
    for b, v in zip(bars, student_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0,115); ax.set_title("Your Metrics"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    # Panel 2: Risk probability pie
    ax = axes[1]
    labels = list(pred_proba.keys())
    values = list(pred_proba.values())
    colors = [PALETTE.get(l,"#95A5A6") for l in labels]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
        textprops={"fontsize":9})
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(f"Risk: {pred_label.upper()}", fontweight="bold",
                 color=PALETTE.get(pred_label,"#2C3E50"))

    # Panel 3: Score vs population
    ax = axes[2]
    ax.hist(df_all["final_score"].dropna(), bins=25, color="#BDC3C7", edgecolor="white",
            label="All Students")
    ax.axvline(pred_score, color=PALETTE.get(pred_label,"#E74C3C"),
               linewidth=3, label=f"Your Score: {pred_score}")
    ax.axvline(df_all["final_score"].mean(), color="#2C3E50", linestyle="--",
               linewidth=1.5, label=f"Mean: {df_all['final_score'].mean():.1f}")
    ax.set_xlabel("Score"); ax.set_ylabel("Count"); ax.set_title("Your Score vs Class")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "p7_student_dashboard.png"), dpi=130, bbox_inches="tight")
    plt.close()
    ok("p7_student_dashboard.png محفوظ")

run_integrated_demo(p2_pipe, p3_pipe, p2_features, p5_df)


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("✅ كل المشاريع اكتملت بنجاح!")

output_dir = os.path.join(BASE_DIR, "outputs")
outputs = sorted(os.listdir(output_dir)) if os.path.exists(output_dir) else []

print(f"\n  📁 الملفات الناتجة في فولدر outputs/:")
for f in outputs:
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"     • {f:<40} ({size//1024} KB)")

print(f"\n  📁 الـ Models المحفوظة في فولدر models/:")
models_dir = os.path.join(BASE_DIR, "models")
for f in sorted(os.listdir(models_dir)):
    print(f"     • {f}")

print("""
  ══════════════════════════════════════════════════════
  الخلاصة:
    Project 5 → حللنا بيانات 500 طالب حقيقية
    Project 1 → بنينا Pipeline كامل من البداية للنهاية
    Project 2 → موديل Regression بيتوقع الدرجة الرقمية
    Project 3 → موديل Classification بيصنّف مستوى الطالب
    Project 4 → Agent ذكي بيقدم نصائح دراسية
    Project 6 → 4 رسوم بيانية احترافية
    Project 7 → نظام متكامل بيحلل أي طالب جديد
  ══════════════════════════════════════════════════════
""")
