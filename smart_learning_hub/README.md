# 🎓 Smart Learning Analytics Hub
## مشروع تحليل أداء الطلاب بالذكاء الاصطناعي

---

## 📁 هيكل المشروع

```
smart_learning_hub/
│
├── run_all.py                  ← ← ← شغّل ده بس! هيشغّل كل حاجة
│
├── data/
│   ├── generate_dataset.py     ← بيوّلد بيانات 500 طالب
│   └── student_performance.csv ← البيانات (بتتوّلد أوتوماتيك)
│
├── project1_pipeline/
│   └── project1_pipeline.py    ← Pipeline كامل من البيانات للتوقع
│
├── project2_regression/
│   └── regression_model.py     ← توقع الدرجة الرقمية للطالب
│
├── project3_classification/
│   └── classification_model.py ← تصنيف الطالب (at_risk/average/strong)
│
├── project4_llm_agent/
│   └── study_agent.py          ← مساعد ذكي بالـ AI
│
├── project5_real_data/
│   └── project5_analysis.py    ← تحليل البيانات الحقيقية
│
├── project6_visualization/
│   └── visualization_dashboard.py ← رسوم بيانية احترافية
│
├── models/                     ← الـ Models المحفوظة (بتتعمل أوتوماتيك)
└── outputs/                    ← كل الملفات الناتجة (صور وتقارير)
```

---

## 🚀 كيف تشغّل المشروع

### الخطوة 1 — تثبيت المكتبات (مرة واحدة بس)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### الخطوة 2 — تشغيل كل حاجة
```bash
python run_all.py
```

**خلاص! هو هيعمل كل حاجة تلقائياً.**

---

## 📊 إيه اللي هيحصل لما تشغّل؟

| المرحلة | المشروع | اللي بيحصل |
|---------|---------|------------|
| 0 | Data | توليد بيانات 500 طالب |
| 1 | Project 5 | تحليل البيانات ورسم هيت ماب |
| 2 | Project 1 | Pipeline كامل وحفظ الموديل |
| 3 | Project 2 | تدريب 4 موديلات Regression ومقارنتها |
| 4 | Project 3 | تدريب 4 موديلات Classification ومقارنتها |
| 5 | Project 4 | عرض مثال على الـ AI Agent |
| 6 | Project 6 | رسم 4 داشبورد احترافية |
| 7 | Project 7 | تحليل طالب جديد وعرض النتائج |

---

## 📁 الملفات الناتجة في outputs/

```
outputs/
├── cleaned_dataset.csv          ← البيانات بعد التنظيف
│
├── p5_correlation_heatmap.png   ← Project 5: العلاقات بين الـ Features
├── p5_score_distribution.png    ← Project 5: توزيع الدرجات
│
├── p1_actual_vs_predicted.png   ← Project 1: مقارنة التوقعات بالحقيقة
│
├── p6_fig1_overview.png         ← Project 6: نظرة عامة على البيانات
├── p6_fig2_patterns.png         ← Project 6: الأنماط والعلاقات
├── p6_fig3_models.png           ← Project 6: نتائج الموديلات
├── p6_fig4_insights.png         ← Project 6: رؤى وتوصيات
│
└── p7_student_dashboard.png     ← Project 7: تحليل الطالب الجديد
```

---

## 🔧 تشغيل مشروع واحد بشكل منفصل

لو عايز تشغّل مشروع واحد بس:

```bash
# توليد البيانات أولاً (لازم يتعمل قبل أي حاجة)
cd data && python generate_dataset.py && cd ..

# تشغيل مشروع 1 بشكل منفصل
cd project1_pipeline
python project1_pipeline.py

# تشغيل مشروع 2 بشكل منفصل
cd project2_regression
python regression_model.py

# وهكذا...
```

---

## 🤖 تفعيل الـ AI Agent (Project 4)

Project 4 بيشتغل بدون API Key في "وضع تجريبي".
لو عايز الـ AI الحقيقي:

1. روح على [console.anthropic.com](https://console.anthropic.com)
2. عمل API Key
3. شغّل:
```bash
export ANTHROPIC_API_KEY=your_key_here
python run_all.py
```

---

## 📚 شرح المصطلحات للمبتدئين

| المصطلح | المعنى بالبسيط |
|---------|---------------|
| **Regression** | الموديل بيتوقع رقم (زي الدرجة من 0-100) |
| **Classification** | الموديل بيتوقع فئة (زي at_risk / average / strong) |
| **R²** | كدة بيوصف دقة الموديل — كلما اقترب من 1 كلما كان أحسن |
| **MAE** | متوسط الخطأ في التوقع — كلما صغر كلما كان أحسن |
| **Feature** | أي معلومة بنديها للموديل (ساعات الدراسة، نسبة الحضور...) |
| **Pipeline** | سلسلة خطوات متصلة من تنظيف البيانات لحد التوقع |
| **Confusion Matrix** | جدول بيوضح أين أخطأ الموديل في التصنيف |
