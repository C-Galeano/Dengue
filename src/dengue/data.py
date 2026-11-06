"""Loading and cleaning SIVIGILA dengue case records (one row per reported case)."""
import unicodedata

import pandas as pd
import requests

from dengue.config import COLUMN_ALIASES, DATOS_GOV_DOMAIN, RAW_DIR


def download_socrata(dataset_id, filename=None, limit=500_000, domain=DATOS_GOV_DOMAIN):
    """Download a datos.gov.co dataset (Socrata API) as CSV into data/raw/.

    dataset_id is the 9-character code in the dataset URL, e.g. 'abcd-1234'.
    """
    url = f"https://{domain}/resource/{dataset_id}.csv"
    response = requests.get(url, params={"$limit": limit}, timeout=120)
    response.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / (filename or f"{dataset_id}.csv")
    path.write_bytes(response.content)
    return path


def _normalize(name):
    """Lowercase, strip accents and spaces so column variants can be matched."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return name.strip().lower().replace(" ", "_")


def standardize_columns(df):
    """Rename municipality-specific columns to the standard names in COLUMN_ALIASES."""
    lookup = {_normalize(alias): std for std, aliases in COLUMN_ALIASES.items() for alias in aliases}
    renamed = {col: lookup[_normalize(col)] for col in df.columns if _normalize(col) in lookup}
    return df.rename(columns=renamed)


def load_cases(path, municipality=None):
    """Read one SIVIGILA CSV (path or file-like) and return it with standard columns and valid year/week."""
    df = standardize_columns(pd.read_csv(path, low_memory=False))
    missing = {"year", "week"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}; add aliases in config.py")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["year", "week"])
    df = df[df["week"].between(1, 53)].astype({"year": int, "week": int})
    if municipality is not None:
        df["municipality"] = municipality
    return df


def load_many(sources):
    """Concatenate several municipal datasets: sources = {municipality: csv_path}."""
    return pd.concat([load_cases(p, municipality=m) for m, p in sources.items()], ignore_index=True)
