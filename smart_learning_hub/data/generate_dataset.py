"""
generate_dataset.py
-------------------
This script creates a realistic synthetic student performance dataset.
Run this FIRST to generate the data file used in all other scripts.
"""

import pandas as pd
import numpy as np
import os

# Set a seed so results are always the same
np.random.seed(42)

NUM_STUDENTS = 500

# --- Generate each feature ---

study_hours       = np.random.normal(loc=4.5, scale=1.5, size=NUM_STUDENTS).clip(0, 12)
attendance        = np.random.normal(loc=75, scale=15, size=NUM_STUDENTS).clip(0, 100)
assignments_done  = np.random.randint(0, 20, size=NUM_STUDENTS)
quizzes_solved    = np.random.randint(0, 15, size=NUM_STUDENTS)
login_frequency   = np.random.randint(1, 30, size=NUM_STUDENTS)
missed_deadlines  = np.random.randint(0, 10, size=NUM_STUDENTS)
retries           = np.random.randint(0, 5, size=NUM_STUDENTS)
revision_sessions = np.random.randint(0, 8, size=NUM_STUDENTS)
time_per_task     = np.random.normal(loc=30, scale=10, size=NUM_STUDENTS).clip(5, 90)

# --- Calculate final score (based on realistic logic) ---
score = (
    study_hours       * 3.5  +
    attendance        * 0.3  +
    assignments_done  * 1.2  +
    quizzes_solved    * 1.5  -
    missed_deadlines  * 2.0  +
    revision_sessions * 2.0  +
    np.random.normal(0, 5, NUM_STUDENTS)  # add some noise
)

# Normalize score to 0–100
score = ((score - score.min()) / (score.max() - score.min()) * 100).round(1)

# --- Add some missing values (to make it realistic) ---
# Only float arrays can hold NaN; convert quizzes_solved to float first
study_hours    = study_hours.astype(float)
attendance     = attendance.astype(float)
quizzes_solved = quizzes_solved.astype(float)

for col_arr in [study_hours, attendance, quizzes_solved]:
    mask = np.random.choice([True, False], size=NUM_STUDENTS, p=[0.05, 0.95])
    col_arr[mask] = np.nan

# --- Create learner status label ---
def classify(s):
    if s < 40:
        return "at_risk"
    elif s < 70:
        return "average"
    else:
        return "strong"

# --- Build the DataFrame ---
df = pd.DataFrame({
    "student_id":        [f"STU{str(i).zfill(4)}" for i in range(1, NUM_STUDENTS + 1)],
    "study_hours":       study_hours.round(2),
    "attendance_pct":    attendance.round(1),
    "assignments_done":  assignments_done,
    "quizzes_solved":    quizzes_solved,
    "login_frequency":   login_frequency,
    "missed_deadlines":  missed_deadlines,
    "retries":           retries,
    "revision_sessions": revision_sessions,
    "time_per_task_min": time_per_task.round(1),
    "final_score":       score,
    "learner_status":    [classify(s) for s in score],
})

# --- Save ---
out_path = os.path.join(os.path.dirname(__file__), "student_performance.csv")
df.to_csv(out_path, index=False)
print(f"✅ Dataset saved → {out_path}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
