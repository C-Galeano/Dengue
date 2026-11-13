"""Loads the trained model (models/model.pkl) and produces the differential diagnosis for one patient."""
import joblib

from src.explain import explain_patient
from src.preprocessing import MODEL_PATH, patient_frame

# Suggested next steps shown to the clinician. Generic guidance for a screening tool, not a prescription.
NEXT_STEPS = {
    "dengue": [
        "Confirm with NS1 antigen or RT-PCR (first 5 days) or IgM serology (after day 5).",
        "Order a complete blood count and follow platelets and hematocrit.",
        "Check for warning signs: abdominal pain, persistent vomiting, bleeding, lethargy, fluid accumulation.",
        "Avoid aspirin and NSAIDs; use paracetamol and oral hydration.",
    ],
    "chikungunya": [
        "Confirm with RT-PCR (first days) or IgM serology.",
        "Manage joint pain and inflammation; arthralgia can last weeks or months.",
        "Rule out dengue before giving NSAIDs, because both viruses co-circulate.",
    ],
    "other_febrile": [
        "Consider other causes of fever: influenza, COVID-19, malaria, leptospirosis, urinary or bacterial infection.",
        "Dengue is not ruled out: if the fever persists or warning signs appear, re-evaluate.",
    ],
}


class Predictor:
    def __init__(self, path=MODEL_PATH):
        artifact = joblib.load(path)
        self.model = artifact["model"]
        # One thread per prediction: n_jobs=-1 (saved at training time) makes LightGBM start one OpenMP thread
        # per core of the HOST machine. On a small cloud instance (Render free: 0.1 CPU) those threads fight
        # over a fraction of one CPU and a single prediction can exceed the 30 s worker timeout.
        if "n_jobs" in self.model.get_params():
            self.model.set_params(n_jobs=1)
        self.classes = artifact["classes"]
        self.class_labels = artifact["class_labels"]
        self.feature_labels = artifact["feature_labels"]
        self.dengue_threshold = artifact["dengue_threshold"]
        self.info = {k: v for k, v in artifact.items() if k != "model"}  # Metrics for "About"

    def predict(self, age, sex, days_since_onset, findings, state, month):
        """findings: dict {binary feature: bool}; state: IBGE code ("31"); month: 1-12.

        Returns a dict with the probability of each class, the diagnosis suggested by the decision rule,
        whether the dengue alert fired, and the SHAP explanation of the suggested diagnosis.
        """
        X = patient_frame(age, sex, days_since_onset, findings, state, month)
        proba = self.model.predict_proba(X)[0]
        dengue = self.classes.index("dengue")
        alert = bool(proba[dengue] >= self.dengue_threshold)
        index = dengue if alert else int(proba.argmax())
        return {
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
            "prediction": self.classes[index],
            "dengue_alert": alert,
            "explanation": explain_patient(self.model, X, index, self.feature_labels),
        }
