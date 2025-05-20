import numpy as np
from fastapi import FastAPI
from app.schemas import PredictionInput, PredictionOutput
from app.utils import preprocess_input, get_shap_values
from app.model import base_model1, base_model2, base_model3, meta_model, preprocessor

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diabetes Risk Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # 1. Preprocess input (original features)
    instance = preprocess_input(data)  # shape: (1, n_features)

    # 2. Get predictions from base learners
    pred1 = base_model1.predict(instance)[0]
    pred2 = base_model2.predict(instance)[0]
    pred3 = base_model3.predict(instance)[0]

    # 3. Stack predictions for meta learner
    meta_features = np.array([[pred1, pred2, pred3]])
    risk_score = float(meta_model.predict(meta_features)[0])

    # 4. Pass original preprocessed features to surrogate SHAP explainer
    shap_values = get_shap_values(instance)

    # Format risk score as percentage with 2 decimal places
    risk_score_percent = round(risk_score * 100, 2)

    # Get feature names from preprocessor
    def get_feature_names(column_transformer):
        output_features = []
        for name, transformer, original_features in column_transformer.transformers_:
            if transformer == 'passthrough':
                output_features.extend(original_features)
            elif hasattr(transformer, 'get_feature_names_out'):
                output_features.extend(transformer.get_feature_names_out(original_features))
            elif hasattr(transformer, 'get_feature_names'):
                if hasattr(transformer, 'categories_'):
                    output_features.extend([f"{original_features[0]}_{cat}" for cat in transformer.categories_[0]])
                else:
                    output_features.extend(original_features)
            else:
                output_features.extend(original_features)
        return output_features

    feature_names = get_feature_names(preprocessor)
    shap_pairs = list(zip(feature_names, shap_values))
    # Sort by absolute SHAP value, descending
    shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
    # Format each as "• Feature (+0.30)"
    shap_explanations = [
        f"• {name} ({value:+.2f})" for name, value in shap_pairs_sorted
    ]

    explanation_text = (
        f"For a patient predicted to have a {risk_score_percent:.0f}% diabetes risk, SHAP reveals:\n" +
        "\n".join(shap_explanations[:10])  # Show top 10
    )

   
    return PredictionOutput(
        risk_score=risk_score_percent,
        shap_values=[round(val, 2) for _, val in shap_pairs_sorted[:10]],
        explanation=explanation_text
    )