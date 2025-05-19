import joblib
import os

# Load the model and explainer only once when the app starts
MODEL_PATH = os.path.join("models", "ensemble_model.pkl")
EXPLAINER_PATH = os.path.join("models", "shap_explainer.pkl")

model = joblib.load(MODEL_PATH)
explainer = joblib.load(EXPLAINER_PATH)
