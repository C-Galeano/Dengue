"""Limpieza de los datos SIVIGILA de dengue (Tuluá) y definición de las variables del modelo.

Este módulo lo usan tanto el entrenamiento (src/train.py) como la aplicación web,
para que un paciente del formulario se transforme EXACTAMENTE igual que los datos de entrenamiento.
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "dengue_tulua.csv"
MODEL_PATH = ROOT_DIR / "models" / "modelo.pkl"

# Síntomas que un paciente puede reportar en un formulario.
# clave = nombre limpio usado por el modelo y la app; valor = columna original SIVIGILA.
# Se excluyen: fiebre (constante, está en el 100 % de los casos), hallazgos de laboratorio/examen
# (caida_plaq, aum_hemato, hipotensio, hepatomeg, acum_liqui, hipotermia) y variables con fuga de
# información (choque, da_o_organ, extravasac, clasfinal, conducta, pac_hos_, ajuste_, tip_cas_...).
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

# Etiquetas legibles para la app web y los gráficos.
NOMBRES_SINTOMAS = {
    "cefalea": "Dolor de cabeza",
    "dolor_retroocular": "Dolor detrás de los ojos",
    "mialgias": "Dolor muscular",
    "artralgia": "Dolor en las articulaciones",
    "erupcion": "Erupción o manchas en la piel",
    "dolor_abdominal": "Dolor abdominal",
    "vomito": "Vómito",
    "diarrea": "Diarrea",
    "somnolencia": "Somnolencia o decaimiento",
    "sangrado_mucosas": "Sangrado de encías o nariz",
}

# Rangos de edad de SIVIGILA (columna grup_etario). El dataset no trae la edad exacta.
GRUPOS_EDAD = ["0-5", "6-11", "12-17", "18-28", "29-59", "60+"]
SEXOS = ["F", "M"]

FEATURES = ["grupo_edad", "sexo"] + list(SINTOMAS)
CATEGORICAS = ["grupo_edad", "sexo"]
BINARIAS = list(SINTOMAS)
TARGET = "dengue"


def edad_a_grupo(edad):
    """Convierte una edad en años al rango de SIVIGILA (el formulario pide la edad exacta)."""
    edad = float(edad)
    if edad < 0:
        raise ValueError("La edad no puede ser negativa")
    for limite, grupo in [(5, "0-5"), (11, "6-11"), (17, "12-17"), (28, "18-28"), (59, "29-59")]:
        if edad <= limite:
            return grupo
    return "60+"


def a_binario(serie):
    """Normaliza un síntoma a 1 (sí) / 0 (no). SIVIGILA codifica 1 = Sí, 2 = No.

    Cualquier otro valor (vacío, 'desconocido', texto raro) queda como NaN y lo resuelve
    el imputador del pipeline. Acepta también True/False y 'si'/'no' (útil para la app web).
    """
    mapa = {"1": 1, "2": 0, "0": 0, "si": 1, "sí": 1, "no": 0, "true": 1, "false": 0}
    texto = serie.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    return texto.map(mapa).astype(float)


def definir_objetivo(df):
    """Etiqueta cada caso: 1 = dengue confirmado, 0 = descartado, NaN = probable sin confirmar.

    - Descartado: ajuste_ = 6 (el ajuste posterior manda sobre el tipo de caso inicial).
    - Confirmado: tip_cas_ en {3 laboratorio, 5 nexo} o ajuste_ en {3, 5}.
    - Los "probables" que nunca se confirmaron ni descartaron se excluyen (etiqueta incierta).
    """
    ajuste = pd.to_numeric(df["ajuste_"], errors="coerce")
    tipo = pd.to_numeric(df["tip_cas_"], errors="coerce")
    descartado = ajuste == 6
    confirmado = ~descartado & (tipo.isin([3, 5]) | ajuste.isin([3, 5]))
    y = pd.Series(float("nan"), index=df.index)
    y[confirmado] = 1
    y[descartado] = 0
    return y


def preparar_datos(path=DATA_PATH):
    """Lee el CSV crudo y devuelve (X, y, fechas) listos para entrenar."""
    crudo = pd.read_csv(path, low_memory=False).drop_duplicates()
    y = definir_objetivo(crudo)
    crudo = crudo[y.notna()]
    y = y[y.notna()].astype(int).rename(TARGET)

    X = pd.DataFrame(index=crudo.index)
    X["grupo_edad"] = crudo["grup_etario"].where(crudo["grup_etario"].isin(GRUPOS_EDAD))
    X["sexo"] = crudo["sexo_"].str.upper().where(crudo["sexo_"].str.upper().isin(SEXOS))
    for limpio, original in SINTOMAS.items():
        X[limpio] = a_binario(crudo[original])

    fechas = pd.to_datetime(crudo["fec_not"], errors="coerce")
    return X[FEATURES], y, fechas
