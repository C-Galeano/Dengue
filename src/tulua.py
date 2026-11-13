"""External validation in Colombia: SIVIGILA dengue records of Tuluá (Valle del Cauca), datos.gov.co dq2k-gub9.

The arbovirus model is trained on Brazilian data. This module checks whether its dengue probability still
separates confirmed from discarded dengue cases in a Colombian municipality.

The Colombian form (SIVIGILA event 210) records fewer fields than SINAN. Fields with an equivalent are
mapped; fields the Colombian form does not record (arthritis, back pain, conjunctivitis, nausea,
leukopenia, tourniquet test, comorbidities…) are set to 0 = "not reported". Only age RANGES are
published, so the middle of each range is used as the age. The Brazilian state of residence does not
apply in Colombia, so it is left missing.

Raw SIVIGILA column names are in Spanish because that is how the dataset is published.
"""
import pandas as pd

from src.preprocessing import BINARY, FEATURES, MAX_DAYS_SINCE_ONSET, ROOT_DIR

DATA_PATH = ROOT_DIR / "data" / "raw" / "dengue_tulua.csv"
SOURCE = "SIVIGILA Tuluá, Colombia - datos.gov.co (dq2k-gub9)"

# model feature -> SIVIGILA column
EQUIVALENT_FIELDS = {
    "fever": "fiebre",
    "headache": "cefalea",
    "retro_orbital_pain": "dolrretroo",
    "myalgia": "malgias",
    "arthralgia": "artralgia",
    "rash": "erupcionr",
    "vomiting": "vomito",
}
AGE_GROUP_MIDPOINTS = {"0-5": 3, "6-11": 9, "12-17": 15, "18-28": 23, "29-59": 44, "60+": 70}


def define_target(raw):
    """1 = confirmed dengue (laboratory or epidemiological link), 0 = discarded (ajuste_ = 6), NaN = probable."""
    adjustment = pd.to_numeric(raw["ajuste_"], errors="coerce")
    case_type = pd.to_numeric(raw["tip_cas_"], errors="coerce")
    discarded = adjustment == 6
    confirmed = ~discarded & (case_type.isin([3, 5]) | adjustment.isin([3, 5]))
    y = pd.Series(float("nan"), index=raw.index)
    y[confirmed], y[discarded] = 1, 0
    return y


def load_tulua(path=DATA_PATH):
    """Returns (X with the arbovirus model features, y: 1 = dengue / 0 = discarded)."""
    raw = pd.read_csv(path, low_memory=False).drop_duplicates()
    y = define_target(raw)
    raw, y = raw[y.notna()], y[y.notna()].astype(int)

    X = pd.DataFrame(0.0, index=raw.index, columns=FEATURES)
    X["age"] = raw["grup_etario"].map(AGE_GROUP_MIDPOINTS)
    X["sex_female"] = raw["sexo_"].str.upper().map({"F": 1.0, "M": 0.0})
    days = (pd.to_datetime(raw["fec_not"], errors="coerce") - pd.to_datetime(raw["ini_sin_"], errors="coerce")).dt.days
    X["days_since_onset"] = days.where(days.between(0, MAX_DAYS_SINCE_ONSET))
    # Epidemiological context: Tuluá has no Brazilian state (left missing) but the month is known.
    X["state"] = float("nan")
    X["month"] = pd.to_datetime(raw["fec_not"], errors="coerce").dt.month
    for feature, column in EQUIVALENT_FIELDS.items():
        X[feature] = pd.to_numeric(raw[column], errors="coerce").map({1: 1.0, 2: 0.0}).fillna(0.0)
    keep = X[["age", "sex_female"]].notna().all(axis=1)
    assert set(EQUIVALENT_FIELDS) <= set(BINARY)
    return X[keep], y[keep]
