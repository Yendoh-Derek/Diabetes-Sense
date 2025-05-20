from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    gender: str  # Example values: "male", "female"
    age: int
    hypertension: int  # 0 or 1
    heart_disease: int  # 0 or 1
    smoking_history: str  # "non-smoker", "current_smoker", "former_smoker"
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

class PredictionOutput(BaseModel):
    risk_score: float
    shap_values: List[float]
    explanation: str
