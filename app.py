import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Student Exam Score Predictor")

# Auto-train artifacts if student_score_model.pkl is missing
if not (BASE_DIR / "student_score_model.pkl").exists():
    subprocess.run([sys.executable, str(BASE_DIR / "student_train.py")], cwd=BASE_DIR, check=True)

model = joblib.load(BASE_DIR / "student_score_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
encoders = joblib.load(BASE_DIR / "encoders.pkl")
metadata = joblib.load(BASE_DIR / "metadata.pkl")

feature_order = metadata["feature_order"]
numeric_cols = metadata["numeric_cols"]
categorical_cols = metadata["categorical_cols"]


class StudentInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: float
    Previous_Scores: float
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: float
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: float
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


@app.get("/")
def serve_home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def serve_css():
    return FileResponse(BASE_DIR / "style.css")


@app.get("/script.js")
def serve_js():
    return FileResponse(BASE_DIR / "script.js")


@app.post("/predict")
def predict_score(data: StudentInput):
    input_df = pd.DataFrame([data.model_dump()])

    for col in categorical_cols:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

    input_df = input_df[feature_order]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = float(model.predict(input_df)[0])
    return {"predicted_score": round(prediction, 1)}