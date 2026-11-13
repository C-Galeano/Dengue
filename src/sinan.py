"""CRISP-DM Phases 2–3: download Brazil's SINAN arbovirus microdata and build the modeling dataset.

Source: DATASUS (Brazilian Ministry of Health), SINAN final files, one per disease and year:
    DENGBR<yy>.dbc  dengue notifications        CHIKBR<yy>.dbc  chikungunya notifications
Both diseases share the same notification form (same symptoms, findings and comorbidities), so their
records can be combined. Zika (ZIKABR<yy>.dbc) uses a short form WITHOUT clinical fields, so it cannot be
used to learn from symptoms.

Labels (only cases closed by LABORATORY criterion, CRITERIO = 1, so that the label does not depend on the
same symptoms the model uses):
    dengue         CLASSI_FIN 10, 11, 12 (dengue, dengue with warning signs, severe dengue)
    chikungunya    CLASSI_FIN 13
    other_febrile  CLASSI_FIN 5 (discarded). The public dengue files contain NO discarded cases, so this
                   class comes from chikungunya suspects that the laboratory ruled out.

Usage (downloads ~150 MB per year the first time, then reads ~2 M records; takes several minutes):
    python -m src.sinan --years 2023 2025
"""
import argparse
import json
import urllib.request

import pandas as pd

from src.dbc import dbc_columns, read_dbc
from src.preprocessing import (BINARY, CLASSES, FEATURES, PROCESSED_DIR, ROOT_DIR, SINAN_COLUMNS,
                               age_in_years, days_between, to_binary)

FTP_URL = "ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/FINAIS/{name}.dbc"
RAW_DIR = ROOT_DIR / "data" / "raw" / "sinan"
SEED = 42
TRAIN_YEAR, VALIDATION_YEAR = 2023, 2025
SAMPLE_PER_CLASS = {TRAIN_YEAR: 40_000, VALIDATION_YEAR: 15_000}

ID_COLUMNS = ["CLASSI_FIN", "CRITERIO", "SG_UF", "NU_IDADE_N", "CS_SEXO", "DT_NOTIFIC", "DT_SIN_PRI"]
LABEL_CODES = {"10": "dengue", "11": "dengue", "12": "dengue", "13": "chikungunya", "5": "other_febrile"}
LABORATORY = "1"


def dataset_path(year):
    return PROCESSED_DIR / f"arbovirus_{year}.parquet"


def download(name):
    """Download one SINAN file to data/raw/sinan/ (skipped if it is already there)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.dbc"
    if not path.exists():
        print(f"Downloading {name}.dbc from DATASUS…")
        urllib.request.urlretrieve(FTP_URL.format(name=name), path)
    return path


def read_file(name):
    """Read the columns used by the project from one SINAN file."""
    columns = ID_COLUMNS + list(SINAN_COLUMNS.values())
    print(f"Reading {name}.dbc…")
    return pd.DataFrame(read_dbc(download(name), columns=columns))


def to_features(raw):
    """Raw SINAN records -> model features (see src/preprocessing.py) + label and context columns."""
    X = pd.DataFrame(index=raw.index)
    X["age"] = age_in_years(raw["NU_IDADE_N"])
    X["days_since_onset"] = days_between(raw["DT_SIN_PRI"], raw["DT_NOTIFIC"])
    X["sex_female"] = raw["CS_SEXO"].map({"F": 1.0, "M": 0.0})
    X["notified"] = pd.to_datetime(raw["DT_NOTIFIC"], format="%Y%m%d", errors="coerce")
    X["state"] = pd.to_numeric(raw["SG_UF"], errors="coerce")  # epidemiological context: where
    X["month"] = X["notified"].dt.month                        # epidemiological context: when
    for key in BINARY:
        X[key] = to_binary(raw[SINAN_COLUMNS[key]])
    X["diagnosis"] = raw["CLASSI_FIN"].map(LABEL_CODES)
    return X


def summarize(raw, source):
    """Counts of the raw file by final classification and criterion (for the Data Understanding phase)."""
    table = pd.crosstab(raw["CLASSI_FIN"].replace("", "blank"), raw["CRITERIO"].replace("", "blank"))
    return {"source": source, "records": int(len(raw)),
            "classification_by_criterion": {str(k): {str(c): int(v) for c, v in row.items()}
                                            for k, row in table.iterrows()}}


def build_year(year, per_class=None):
    """Lab-confirmed, class-balanced sample of one year. Saves data/processed/arbovirus_<year>.parquet."""
    per_class = per_class or SAMPLE_PER_CLASS.get(year, 15_000)
    yy = str(year)[2:]
    frames, summary = [], {"year": year, "files": []}
    for name in (f"DENGBR{yy}", f"CHIKBR{yy}"):
        raw = read_file(name)
        summary["files"].append(summarize(raw, name))
        lab = raw[(raw["CRITERIO"] == LABORATORY) & raw["CLASSI_FIN"].isin(LABEL_CODES)]
        frames.append(to_features(lab))
    data = pd.concat(frames, ignore_index=True)
    complete = data.dropna(subset=["age", "sex_female", "state", "month", *BINARY])
    summary["lab_confirmed"] = {c: int((data["diagnosis"] == c).sum()) for c in CLASSES}
    summary["dropped_incomplete"] = int(len(data) - len(complete))

    sample = (pd.concat([group.sample(min(per_class, len(group)), random_state=SEED)
                         for _, group in complete.groupby("diagnosis")])
              .sample(frac=1, random_state=SEED).reset_index(drop=True))
    summary["sample"] = {c: int((sample["diagnosis"] == c).sum()) for c in CLASSES}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample[FEATURES + ["diagnosis", "notified"]].to_parquet(dataset_path(year), index=False)
    zika = download(f"ZIKABR{yy}")
    summary["zika_columns"] = dbc_columns(zika)
    (PROCESSED_DIR / f"arbovirus_{year}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{year}: {len(sample):,} records saved to {dataset_path(year).relative_to(ROOT_DIR)}  {summary['sample']}")
    return sample


def load_dataset(year=TRAIN_YEAR):
    """Read a processed year, building it first if needed. Returns (X, y, context)."""
    path = dataset_path(year)
    data = pd.read_parquet(path) if path.exists() else build_year(year)
    # Files built before the context features were added store the state as text: derive them here.
    data["state"] = pd.to_numeric(data["state"], errors="coerce")
    data["month"] = pd.to_datetime(data["notified"]).dt.month
    y = data["diagnosis"].map({c: i for i, c in enumerate(CLASSES)}).astype(int).rename("diagnosis")
    return data[FEATURES], y, data[["state", "notified"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[TRAIN_YEAR, VALIDATION_YEAR])
    for year in parser.parse_args().years:
        build_year(year)


if __name__ == "__main__":
    main()
