from fastapi import FastAPI
from app.schemas import PredictionInput, PredictionOutput
from app.utils import preprocess_input, get_shap_values
from app.model import model

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diabetes Risk Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    instance = preprocess_input(data)
    risk_score = model.predict_proba(instance)[0][1]  # class 1 probability
    shap_values = get_shap_values(instance)

    return PredictionOutput(
        risk_score=risk_score,
        shap_values=shap_values
    )
