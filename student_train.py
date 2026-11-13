"""
student_train.py
----------------
Trains a compact RandomForestRegressor on StudentPerformanceFactors.csv
and saves compressed artifacts (< 5 MB) for Streamlit deployment.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(BASE_DIR / "StudentPerformanceFactors.csv")

    target_col = "Exam_Score"
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    feature_order = X.columns.tolist()

    # Handle any missing values cleanly
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0]).astype(str)
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Fit label encoders
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaler on numeric columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Pruned RandomForest keeps file size ~2-4 MB and RAM usage very low
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    metadata = {
        "feature_order": feature_order,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

    joblib.dump(model, BASE_DIR / "student_score_model.pkl", compress=3)
    joblib.dump(scaler, BASE_DIR / "scaler.pkl", compress=3)
    joblib.dump(encoders, BASE_DIR / "encoders.pkl", compress=3)
    joblib.dump(metadata, BASE_DIR / "metadata.pkl", compress=3)
    print("Artifacts saved successfully.")


if __name__ == "__main__":
    main()
