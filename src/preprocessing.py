"""Model variables shared by training (src/train.py) and the web application.

The arbovirus classifier uses the clinical fields of Brazil's SINAN notification form for dengue and
chikungunya (the same form for both diseases). Keeping the definitions in one module guarantees that a
patient entered in the web form is transformed EXACTLY like the training data.

Raw SINAN column names (``FEBRE``, ``ARTRALGIA``…) are in Portuguese because that is how DATASUS
publishes them; they only appear in the mappings below.
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------------------------------------------------------------- Target classes
CLASSES = ["dengue", "chikungunya", "other_febrile"]
CLASS_LABELS = {
    "dengue": "Dengue",
    "chikungunya": "Chikungunya",
    "other_febrile": "Other febrile illness",
}

# ---------------------------------------------------------------- Clinical features
# key = clean name used by the model and the app; value = (SINAN column, readable label)
SYMPTOMS = {
    "fever": ("FEBRE", "Fever"),
    "headache": ("CEFALEIA", "Headache"),
    "retro_orbital_pain": ("DOR_RETRO", "Pain behind the eyes"),
    "myalgia": ("MIALGIA", "Muscle pain"),
    "arthralgia": ("ARTRALGIA", "Joint pain"),
    "arthritis": ("ARTRITE", "Joint swelling (arthritis)"),
    "back_pain": ("DOR_COSTAS", "Back pain"),
    "rash": ("EXANTEMA", "Skin rash"),
    "conjunctivitis": ("CONJUNTVIT", "Red eyes (conjunctivitis)"),
    "nausea": ("NAUSEA", "Nausea"),
    "vomiting": ("VOMITO", "Vomiting"),
}
# Findings from the physical exam or a basic blood count, available at the first consultation.
CLINICAL_FINDINGS = {
    "petechiae": ("PETEQUIA_N", "Petechiae (small red skin spots)"),
    "tourniquet_test": ("LACO", "Positive tourniquet test"),
    "leukopenia": ("LEUCOPENIA", "Leukopenia (low white blood cells)"),
}
COMORBIDITIES = {
    "diabetes": ("DIABETES", "Diabetes"),
    "hypertension": ("HIPERTENSA", "Hypertension"),
    "kidney_disease": ("RENAL", "Chronic kidney disease"),
    "liver_disease": ("HEPATOPAT", "Liver disease"),
    "hematologic_disease": ("HEMATOLOG", "Blood disorder"),
    "peptic_ulcer": ("ACIDO_PEPT", "Peptic ulcer"),
    "autoimmune_disease": ("AUTO_IMUNE", "Autoimmune disease"),
}
BINARY_GROUPS = {"Symptoms": SYMPTOMS, "Clinical and laboratory findings": CLINICAL_FINDINGS,
                 "Comorbidities": COMORBIDITIES}
BINARY = [key for group in BINARY_GROUPS.values() for key in group]
NUMERIC = ["age", "days_since_onset", "sex_female"]

# ---------------------------------------------------------------- Epidemiological context
# Where and when the patient consults. Clinicians use local circulation as the pre-test probability:
# the same symptoms point to chikungunya where it circulates and to dengue elsewhere. The state is the
# patient's Brazilian state of residence (IBGE code, SINAN column SG_UF) and the month that of the consultation.
STATES = {
    "11": "Rondônia", "12": "Acre", "13": "Amazonas", "14": "Roraima", "15": "Pará", "16": "Amapá",
    "17": "Tocantins", "21": "Maranhão", "22": "Piauí", "23": "Ceará", "24": "Rio Grande do Norte",
    "25": "Paraíba", "26": "Pernambuco", "27": "Alagoas", "28": "Sergipe", "29": "Bahia",
    "31": "Minas Gerais", "32": "Espírito Santo", "33": "Rio de Janeiro", "35": "São Paulo",
    "41": "Paraná", "42": "Santa Catarina", "43": "Rio Grande do Sul", "50": "Mato Grosso do Sul",
    "51": "Mato Grosso", "52": "Goiás", "53": "Distrito Federal",
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
CONTEXT = ["state", "month"]

FEATURES = NUMERIC + CONTEXT + BINARY

FEATURE_LABELS = {
    "age": "Age",
    "days_since_onset": "Days since symptom onset",
    "sex_female": "Female sex",
    "state": "State of residence",
    "month": "Month of consultation",
    **{key: label for group in BINARY_GROUPS.values() for key, (_, label) in group.items()},
}
SINAN_COLUMNS = {key: col for group in BINARY_GROUPS.values() for key, (col, _) in group.items()}

MAX_AGE = 110
MAX_DAYS_SINCE_ONSET = 30


def to_binary(series):
    """SINAN codes yes/no fields as 1 = Yes, 2 = No (9 = unknown). Returns 1.0 / 0.0 / NaN."""
    return series.astype(str).str.strip().map({"1": 1.0, "2": 0.0}).astype(float)


def age_in_years(series):
    """Decodes SINAN's NU_IDADE_N: the first digit is the unit (1 hours, 2 days, 3 months, 4 years)."""
    text = series.astype(str).str.strip()
    unit, value = text.str[:1], pd.to_numeric(text.str[1:], errors="coerce")
    years = value.where(unit == "4", (value / 12).where(unit == "3", (value / 365).where(unit == "2", 0)))
    return years.where(unit.isin(["1", "2", "3", "4"])).where(lambda a: a.between(0, MAX_AGE))


def days_between(start, end):
    """Whole days from start to end (YYYYMMDD strings); values outside 0..30 become NaN."""
    days = (pd.to_datetime(end, format="%Y%m%d", errors="coerce")
            - pd.to_datetime(start, format="%Y%m%d", errors="coerce")).dt.days
    return days.where(days.between(0, MAX_DAYS_SINCE_ONSET))


def feature_value_text(feature, value):
    """Readable value of a feature for the explanation (e.g. state code 31 -> Minas Gerais)."""
    if pd.isna(value):
        return "unknown"
    if feature == "state":
        return STATES.get(str(int(value)), str(int(value)))
    if feature == "month":
        return MONTHS[int(value) - 1]
    if feature in ("age", "days_since_onset"):
        return str(int(value))
    return "yes" if value else "no"


def patient_frame(age, sex, days_since_onset, findings, state, month):
    """One-row DataFrame for a patient entered in the web form.

    findings: dict {binary feature: bool}. Anything not ticked is "No" (0), as in the SINAN form.
    state: IBGE state code (e.g. "31"); month: 1-12.
    """
    row = {
        "age": float(age),
        "days_since_onset": float(min(max(days_since_onset, 0), MAX_DAYS_SINCE_ONSET)),
        "sex_female": 1.0 if sex == "F" else 0.0,
        "state": float(state),
        "month": float(month),
        **{key: 1.0 if findings.get(key) else 0.0 for key in BINARY},
    }
    return pd.DataFrame([row])[FEATURES]
