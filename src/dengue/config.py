"""Project paths and column conventions shared across modules."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # Untouched downloads (SIVIGILA, IDEAM)
INTERIM_DIR = DATA_DIR / "interim"    # Cleaned case records with standard columns
PROCESSED_DIR = DATA_DIR / "processed"  # Weekly series with features, ready to model
EXTERNAL_DIR = DATA_DIR / "external"  # Climate and other auxiliary sources
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

DATOS_GOV_DOMAIN = "www.datos.gov.co"

# Each municipality publishes SIVIGILA data with its own column names.
# Map every known variant to one standard name (lowercase, no accents).
COLUMN_ALIASES = {
    "year": ["year", "ano", "año", "anio", "ano_", "año_"],
    "week": ["week", "semana", "semana_", "semana_epidemiologica", "sem_epi"],
    "age": ["age", "edad", "edad_"],
    "sex": ["sex", "sexo", "sexo_"],
    "area": ["area", "area_", "área"],
    "commune": ["commune", "comuna", "nom_comuna", "comuna_"],
    "municipality": ["municipality", "municipio", "nmun_resi", "nmun_proce"],
}

TARGET = "cases"
