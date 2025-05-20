import pandas as pd
from app.model import explainer, preprocessor

def preprocess_input(data):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([data.dict()])
    # Apply the saved preprocessor (ColumnTransformer or Pipeline)
    features = preprocessor.transform(df)
    return features

def get_shap_values(instance):
    shap_vals = explainer(instance)
    return shap_vals.values[0].tolist()
