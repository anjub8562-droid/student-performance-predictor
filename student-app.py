"""
student-app.py
---------------
Single-page Streamlit app that loads the trained RandomForest model
(student_score_model.pkl) plus the saved scaler / encoders, takes student
details as input, and predicts the exam score. If student_score_model.pkl
is not present in the repository (e.g., due to GitHub's 25 MB upload limit),
it automatically trains and saves the model on startup using student_train.py.

Run:
    streamlit run student-app.py
"""

import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered",
)

# Resolve the directory containing this script
BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Load model + preprocessing artifacts (cached so it only loads once)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "student_score_model.pkl"
    train_script = BASE_DIR / "student_train.py"

    # Train automatically if the >25MB model file couldn't be uploaded to GitHub
    if not model_path.exists():
        if not train_script.exists():
            st.error("Both `student_score_model.pkl` and `student_train.py` are missing.")
            st.stop()

        with st.spinner("Training model on first startup... this takes ~15 seconds."):
            result = subprocess.run(
                [sys.executable, str(train_script)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                st.error("Automatic model training failed. Error details:")
                st.code(result.stderr or result.stdout)
                st.stop()

    model = joblib.load(BASE_DIR / "student_score_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    encoders = joblib.load(BASE_DIR / "encoders.pkl")
    metadata = joblib.load(BASE_DIR / "metadata.pkl")
    return model, scaler, encoders, metadata


model, scaler, encoders, metadata = load_artifacts()

feature_order = metadata["feature_order"]
numeric_cols = metadata["numeric_cols"]
categorical_cols = metadata["categorical_cols"]

st.title("🎓 Student Exam Score Predictor")
st.write("Enter the student's study habits and background to predict their exam score.")

# --------------------------------------------------------------------------
# Input widgets
# --------------------------------------------------------------------------
st.subheader("📚 Study habits")
col1, col2 = st.columns(2)
with col1:
    hours_studied = st.number_input("Hours studied per week", min_value=0, max_value=50, value=20)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85)
    sleep_hours = st.number_input("Sleep hours per night", min_value=0, max_value=12, value=7)
with col2:
    previous_scores = st.number_input("Previous exam score", min_value=0, max_value=100, value=70)
    tutoring_sessions = st.number_input("Tutoring sessions per month", min_value=0, max_value=10, value=1)
    physical_activity = st.number_input("Physical activity (hrs/week)", min_value=0, max_value=10, value=3)

st.subheader("👨‍👩‍👧 Background & environment")
col3, col4 = st.columns(2)
with col3:
    parental_involvement = st.selectbox("Parental involvement", sorted(encoders["Parental_Involvement"].classes_))
    access_to_resources = st.selectbox("Access to resources", sorted(encoders["Access_to_Resources"].classes_))
    extracurricular = st.selectbox("Extracurricular activities", sorted(encoders["Extracurricular_Activities"].classes_))
    motivation_level = st.selectbox("Motivation level", sorted(encoders["Motivation_Level"].classes_))
    internet_access = st.selectbox("Internet access", sorted(encoders["Internet_Access"].classes_))
    family_income = st.selectbox("Family income", sorted(encoders["Family_Income"].classes_))
    teacher_quality = st.selectbox("Teacher quality", sorted(encoders["Teacher_Quality"].classes_))
with col4:
    school_type = st.selectbox("School type", sorted(encoders["School_Type"].classes_))
    peer_influence = st.selectbox("Peer influence", sorted(encoders["Peer_Influence"].classes_))
    learning_disabilities = st.selectbox("Learning disabilities", sorted(encoders["Learning_Disabilities"].classes_))
    parental_education = st.selectbox("Parental education level", sorted(encoders["Parental_Education_Level"].classes_))
    distance_from_home = st.selectbox("Distance from home", sorted(encoders["Distance_from_Home"].classes_))
    gender = st.selectbox("Gender", sorted(encoders["Gender"].classes_))

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if st.button("Predict Exam Score", type="primary"):
    raw_input = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Gender": gender,
    }
    input_df = pd.DataFrame([raw_input])

    # Apply label encoders to categorical columns
    for col in categorical_cols:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

    # Reorder columns exactly as during training
    input_df = input_df[feature_order]

    # Scale numeric columns using the saved scaler
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Exam Score: {prediction:.1f} / 100")

st.caption("Model: RandomForestRegressor tuned with RandomizedSearchCV")
