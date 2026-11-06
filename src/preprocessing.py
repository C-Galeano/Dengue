"""Cleaning SIVIGILA dengue data (Tuluá) and defining the model variables.

This module is used by both the training process (src/train.py) and the web application,
so that a patient from the form is transformed EXACTLY like the training data.

Note: raw column names (``ajuste_``, ``tip_cas_``, ``grup_etario``...) and some raw values
(``"si"``, ``"no"``) come from the official SIVIGILA dataset, which is published in Spanish.
They are kept as-is because they must match the source file.
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "dengue_tulua.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"

# Symptoms that a patient can report in a form.
# key = clean name used by the model and app; value = original SIVIGILA column.
# Excluded: fever, laboratory/examination findings and variables with data leakage.
SYMPTOMS = {
    "headache": "cefalea",
    "retro_orbital_pain": "dolrretroo",
    "myalgia": "malgias",
    "arthralgia": "artralgia",
    "rash": "erupcionr",
    "abdominal_pain": "dolor_abdo",
    "vomiting": "vomito",
    "diarrhea": "diarrea",
    "drowsiness": "somnolenci",
    "mucosal_bleeding": "hem_mucosa",
}

# Readable labels for the web app and charts.
SYMPTOM_LABELS = {
    "headache": "Headache",
    "retro_orbital_pain": "Pain behind the eyes",
    "myalgia": "Muscle pain",
    "arthralgia": "Joint pain",
    "rash": "Skin rash or spots",
    "abdominal_pain": "Abdominal pain",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "drowsiness": "Drowsiness or fatigue",
    "mucosal_bleeding": "Bleeding gums or nose",
}

# SIVIGILA age ranges (grup_etario column).
# The dataset does not include the exact age.
AGE_GROUPS = ["0-5", "6-11", "12-17", "18-28", "29-59", "60+"]
SEXES = ["F", "M"]

FEATURES = ["age_group", "sex"] + list(SYMPTOMS)
CATEGORICAL = ["age_group", "sex"]
BINARY = list(SYMPTOMS)
TARGET = "dengue"


def age_to_group(age):
    """Converts an age in years to the SIVIGILA range."""
    age = float(age)

    if age < 0:
        raise ValueError("Age cannot be negative")

    for limit, group in [
        (5, "0-5"),
        (11, "6-11"),
        (17, "12-17"),
        (28, "18-28"),
        (59, "29-59"),
    ]:
        if age <= limit:
            return group

    return "60+"


def to_binary(series):
    """Normalizes a symptom to 1 (yes) / 0 (no).

    SIVIGILA uses 1 = Yes, 2 = No. Other values are converted to NaN.
    The Spanish words ("si", "sí") are kept because they appear in the raw data.
    """
    mapping = {
        "1": 1,
        "2": 0,
        "0": 0,
        "si": 1,
        "sí": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )

    return text.map(mapping).astype(float)


def define_target(df):
    """Labels each case:
    1 = confirmed dengue
    0 = discarded
    NaN = probable but unconfirmed.
    """
    adjustment = pd.to_numeric(df["ajuste_"], errors="coerce")
    case_type = pd.to_numeric(df["tip_cas_"], errors="coerce")

    discarded = adjustment == 6
    confirmed = (~discarded) & (
        case_type.isin([3, 5]) | adjustment.isin([3, 5])
    )

    y = pd.Series(float("nan"), index=df.index)
    y[confirmed] = 1
    y[discarded] = 0

    return y


def prepare_data(path=DATA_PATH):
    """Reads the raw CSV and returns (X, y, dates) ready for training."""
    raw = pd.read_csv(path, low_memory=False).drop_duplicates()

    y = define_target(raw)

    raw = raw[y.notna()]
    y = y[y.notna()].astype(int).rename(TARGET)

    X = pd.DataFrame(index=raw.index)

    X["age_group"] = raw["grup_etario"].where(raw["grup_etario"].isin(AGE_GROUPS))

    X["sex"] = raw["sexo_"].str.upper().where(raw["sexo_"].str.upper().isin(SEXES))

    for clean, original in SYMPTOMS.items():
        X[clean] = to_binary(raw[original])

    dates = pd.to_datetime(raw["fec_not"], errors="coerce")

    return X[FEATURES], y, dates
