"""Loads the trained model (models/modelo.pkl) and makes a prediction for a patient from the form."""
import joblib
import pandas as pd

# The SAME preprocessing used during training is reused, so a patient from the form
# is transformed exactly like the data used to train the model.
from src.preprocesamiento import MODEL_PATH, NOMBRES_SINTOMAS, edad_a_grupo

PROBABLE = "Probable dengue"
POCO_PROBABLE = "Unlikely"


class Predictor:
    def __init__(self, path=MODEL_PATH):
        artefacto = joblib.load(path)
        self.modelo = artefacto["modelo"]
        self.features = artefacto["features"]
        self.umbral = artefacto["umbral"]
        self.info = {k: v for k, v in artefacto.items() if k != "modelo"}  # Metrics for "About"

    def predecir(self, edad, sexo, sintomas):
        """sintomas: dict {nombre_sintoma: bool}. Returns (probability 0-1, label)."""
        fila = {"grupo_edad": edad_a_grupo(edad), "sexo": sexo,
                **{s: int(bool(sintomas.get(s))) for s in NOMBRES_SINTOMAS}}
        proba = float(self.modelo.predict_proba(pd.DataFrame([fila])[self.features])[0, 1])
        return proba, (PROBABLE if proba >= self.umbral else POCO_PROBABLE)
    