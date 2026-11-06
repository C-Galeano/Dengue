"""Loads the trained model (models/model.pkl) and makes a prediction for a patient from the form."""
import joblib
import pandas as pd

# The SAME preprocessing used during training is reused, so a patient from the form
# is transformed exactly like the data used to train the model.
from src.preprocessing import MODEL_PATH, SYMPTOM_LABELS, age_to_group

PROBABLE = "Probable dengue"
UNLIKELY = "Unlikely"


class Predictor:
    def __init__(self, path=MODEL_PATH):
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.threshold = artifact["threshold"]
        self.info = {k: v for k, v in artifact.items() if k != "model"}  # Metrics for "About"

    def predict(self, age, sex, symptoms):
        """symptoms: dict {symptom_name: bool}. Returns (probability 0-1, label)."""
        row = {"age_group": age_to_group(age), "sex": sex,
               **{s: int(bool(symptoms.get(s))) for s in SYMPTOM_LABELS}}
        proba = float(self.model.predict_proba(pd.DataFrame([row])[self.features])[0, 1])
        return proba, (PROBABLE if proba >= self.threshold else UNLIKELY)
