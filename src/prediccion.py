"""Carga el modelo entrenado (models/modelo.pkl) y predice para un paciente del formulario."""
import joblib
import pandas as pd

# Se reutiliza el MISMO preprocesamiento del entrenamiento, para que un paciente del formulario
# se transforme exactamente igual que los datos con que se entrenó el modelo.
from src.preprocesamiento import MODEL_PATH, NOMBRES_SINTOMAS, edad_a_grupo

PROBABLE = "Probable dengue"
POCO_PROBABLE = "Poco probable"


class Predictor:
    def __init__(self, path=MODEL_PATH):
        artefacto = joblib.load(path)
        self.modelo = artefacto["modelo"]
        self.features = artefacto["features"]
        self.umbral = artefacto["umbral"]
        self.info = {k: v for k, v in artefacto.items() if k != "modelo"}  # métricas para "Acerca"

    def predecir(self, edad, sexo, sintomas):
        """sintomas: dict {nombre_sintoma: bool}. Devuelve (probabilidad 0-1, etiqueta)."""
        fila = {"grupo_edad": edad_a_grupo(edad), "sexo": sexo,
                **{s: int(bool(sintomas.get(s))) for s in NOMBRES_SINTOMAS}}
        proba = float(self.modelo.predict_proba(pd.DataFrame([fila])[self.features])[0, 1])
        return proba, (PROBABLE if proba >= self.umbral else POCO_PROBABLE)
