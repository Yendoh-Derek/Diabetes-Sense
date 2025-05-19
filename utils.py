import numpy as np
from app.model import model, explainer

def preprocess_input(data):
    # Simple encoding (you must match the encoding used during training)
    gender_map = {"male": 0, "female": 1}
    smoking_map = {
        "non-smoker": 0,
        "former_smoker": 1,
        "current_smoker": 2
    }

    gender = gender_map.get(data.gender.lower(), 0)
    smoking = smoking_map.get(data.smoking_history.lower(), 0)

    features = np.array([[gender, data.age, data.hypertension, data.heart_disease,
                          smoking, data.bmi, data.HbA1c_level, data.blood_glucose_level]])
    return features

def get_shap_values(instance):
    shap_vals = explainer(instance)
    return shap_vals.values[0].tolist()
