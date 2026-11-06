
"""Cleaning SIVIGILA dengue data (Tuluá) and defining the model variables.

This module is used by both the training process (src/train.py) and the web application,
so that a patient from the form is transformed EXACTLY like the training data.
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "dengue_tulua.csv"
MODEL_PATH = ROOT_DIR / "models" / "modelo.pkl"

# Symptoms that a patient can report in a form.
# key = clean name used by the model and app; value = original SIVIGILA column.
# Excluded: fever, laboratory/examination findings and variables with data leakage.

SINTOMAS = {
    "cefalea": "cefalea",
    "dolor_retroocular": "dolrretroo",
    "mialgias": "malgias",
    "artralgia": "artralgia",
    "erupcion": "erupcionr",
    "dolor_abdominal": "dolor_abdo",
    "vomito": "vomito",
    "diarrea": "diarrea",
    "somnolencia": "somnolenci",
    "sangrado_mucosas": "hem_mucosa",
}

# Readable labels for the web app and charts.
NOMBRES_SINTOMAS = {
    "cefalea": "Headache",
    "dolor_retroocular": "Pain behind the eyes",
    "mialgias": "Muscle pain",
    "artralgia": "Joint pain",
    "erupcion": "Skin rash or spots",
    "dolor_abdominal": "Abdominal pain",
    "vomito": "Vomiting",
    "diarrea": "Diarrhea",
    "somnolencia": "Drowsiness or fatigue",
    "sangrado_mucosas": "Bleeding gums or nose",
}

# SIVIGILA age ranges (grup_etario column).
# The dataset does not include the exact age.

GRUPOS_EDAD = ["0-5", "6-11", "12-17", "18-28", "29-59", "60+"]
SEXOS = ["F", "M"]

FEATURES = ["grupo_edad", "sexo"] + list(SINTOMAS)
CATEGORICAS = ["grupo_edad", "sexo"]
BINARIAS = list(SINTOMAS)
TARGET = "dengue"


def edad_a_grupo(edad):
    """Converts an age in years to the SIVIGILA range."""
    edad = float(edad)

    if edad < 0:
        raise ValueError("Age cannot be negative")

    for limite, grupo in [
        (5, "0-5"),
        (11, "6-11"),
        (17, "12-17"),
        (28, "18-28"),
        (59, "29-59")
    ]:
        if edad <= limite:
            return grupo

    return "60+"


def a_binario(serie):
    """Normalizes a symptom to 1 (yes) / 0 (no).

    SIVIGILA uses 1 = Yes, 2 = No.
    Other values are converted to NaN.
    """

    mapa = {
        "1": 1,
        "2": 0,
        "0": 0,
        "si": 1,
        "sí": 1,
        "no": 0,
        "true": 1,
        "false": 0
    }

    texto = (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )

    return texto.map(mapa).astype(float)


def definir_objetivo(df):
    """Labels each case:
    1 = confirmed dengue
    0 = discarded
    NaN = probable but unconfirmed.
    """

    ajuste = pd.to_numeric(df["ajuste_"], errors="coerce")
    tipo = pd.to_numeric(df["tip_cas_"], errors="coerce")

    descartado = ajuste == 6
    confirmado = (~descartado) & (
        tipo.isin([3, 5]) | ajuste.isin([3, 5])
    )

    y = pd.Series(float("nan"), index=df.index)
    y[confirmado] = 1
    y[descartado] = 0

    return y


def preparar_datos(path=DATA_PATH):
    """Reads the raw CSV and returns (X, y, dates) ready for training."""

    crudo = pd.read_csv(path, low_memory=False).drop_duplicates()

    y = definir_objetivo(crudo)

    crudo = crudo[y.notna()]
    y = y[y.notna()].astype(int).rename(TARGET)

    X = pd.DataFrame(index=crudo.index)

    X["grupo_edad"] = crudo["grup_etario"].where(
        crudo["grup_etario"].isin(GRUPOS_EDAD)
    )

    X["sexo"] = crudo["sexo_"].str.upper().where(
        crudo["sexo_"].str.upper().isin(SEXOS)
    )

    for limpio, original in SINTOMAS.items():
        X[limpio] = a_binario(crudo[original])

    fechas = pd.to_datetime(crudo["fec_not"], errors="coerce")

    return X[FEATURES], y, fechas
