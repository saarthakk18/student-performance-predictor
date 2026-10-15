
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load the existing trained pipeline.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "rf_exam_score_pipeline.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load trained model at {MODEL_PATH}: {exc}"
    ) from exc

FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

app = FastAPI(
    title="Student Exam Score Prediction API",
    description="Predict exam scores using the saved Random Forest pipeline.",
    version="1.0.0",
)

# Local development origins. Add your deployed frontend origin later.
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

production_origin = os.getenv("FRONTEND_ORIGIN")
if production_origin:
    origins.append(production_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class StudentInput(BaseModel):
    Hours_Studied: int = Field(ge=0, le=24)
    Attendance: int = Field(ge=0, le=100)
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: int = Field(ge=0, le=24)
    Previous_Scores: int = Field(ge=0, le=100)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int = Field(ge=0, le=100)
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: int = Field(ge=0, le=24)
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(student: StudentInput):
    try:
        # Preserve the exact feature names and training order.
        input_data = pd.DataFrame(
            [student.model_dump()],
            columns=FEATURES,
        )

        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            raise ValueError("Model returned a non-finite prediction.")

        score = round(prediction, 2)

        # General study suggestions, not separate model predictions.
        recommendations = []

        if student.Attendance < 75:
            recommendations.append(
                "Improve attendance and attend classes regularly."
            )
        if student.Hours_Studied < 2:
            recommendations.append(
                "Build a consistent daily study routine."
            )
        if student.Sleep_Hours < 7:
            recommendations.append(
                "Aim for sufficient, regular sleep."
            )
        if student.Tutoring_Sessions < 2:
            recommendations.append(
                "Consider extra practice or academic support."
            )
        if student.Previous_Scores < 60:
            recommendations.append(
                "Revise fundamentals and practise past questions."
            )
        if not recommendations:
            recommendations.append(
                "Maintain your study habits and review weak topics."
            )

        return {
            "predicted_exam_score": score,
            "recommendations": recommendations,
            "model": "Random Forest Regressor",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
