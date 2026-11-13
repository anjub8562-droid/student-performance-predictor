"""
student_train.py
-----------------
Trains a RandomForestRegressor on the Student Performance Factors dataset
to predict `Exam_Score`, tunes hyperparameters with RandomizedSearchCV, and
saves everything the Streamlit app needs to make predictions later.

Run:
    python student_train.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
DATA_PATH = "StudentPerformanceFactors.csv"
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------------------------------
# 2. Handle missing values
# --------------------------------------------------------------------------
# Teacher_Quality, Parental_Education_Level, Distance_from_Home have a small
# number of NaNs -> impute with the most frequent value (mode), which is
# the standard approach for categorical columns.
cat_cols_with_na = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
for col in cat_cols_with_na:
    df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------------------------------------------------------
# 3. Encode categorical columns with LabelEncoder
# --------------------------------------------------------------------------
categorical_cols = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# --------------------------------------------------------------------------
# 4. Split features / target
# --------------------------------------------------------------------------
X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

# keep a fixed column order -> needed so the app builds inputs identically
FEATURE_ORDER = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------------------------
# 5. Scale numeric features
# --------------------------------------------------------------------------
numeric_cols = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --------------------------------------------------------------------------
# 6. RandomForest + RandomizedSearchCV tuning
# --------------------------------------------------------------------------
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring="r2",
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

print("Starting RandomizedSearchCV tuning...")
random_search.fit(X_train_scaled, y_train)

print("\nBest params:", random_search.best_params_)
print("Best CV R2 score:", random_search.best_score_)

best_model = random_search.best_estimator_

# --------------------------------------------------------------------------
# 7. Evaluate on test set
# --------------------------------------------------------------------------
y_pred = best_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Test set performance ---")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# --------------------------------------------------------------------------
# 8. Save model, scaler, encoders, and metadata
# --------------------------------------------------------------------------
joblib.dump(best_model, "student_score_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(
    {
        "feature_order": FEATURE_ORDER,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    },
    "metadata.pkl",
)

print("\nSaved: student_score_model.pkl, scaler.pkl, encoders.pkl, metadata.pkl")
