"""Main page: project structure and weekly baseline forecast.

Reproduces the original Streamlit app (legacy/streamlit_app.py) with the same logic.
"""
import json
from functools import lru_cache

import markdown
import numpy as np
import pandas as pd

from src.preprocesamiento import ROOT_DIR

DOC_PROYECTO = ROOT_DIR / "docs" / "estructura_proyecto.md"
SEMANAS_TEST_MIN, SEMANAS_TEST_MAX, SEMANAS_TEST_DEFECTO = 8, 52, 26


@lru_cache
def documento_proyecto():
    """Converts docs/estructura_proyecto.md to HTML (only once) with Bootstrap styles."""
    html = markdown.markdown(DOC_PROYECTO.read_text(encoding="utf-8"), extensions=["tables"])
    return html.replace("<table>", '<table class="table table-sm table-bordered">')


def serie_demo(semanas=156):
    """Synthetic series with annual seasonality (same as the original app)."""
    rng = np.random.default_rng(42)
    t = np.arange(semanas)
    casos = 40 + 30 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 6, semanas)
    return pd.DataFrame({"semana": t + 1, "casos": np.clip(casos, 0, None).round()})


def leer_serie(archivo=None, serie_json=None):
    """Reads the series from an uploaded CSV (columns `semana` and `casos`) or from the JSON
    sent back by the page when moving the slider. Raises ValueError with a readable message if invalid."""
    try:
        df = pd.read_csv(archivo) if archivo else pd.DataFrame(json.loads(serie_json))
    except Exception as exc:
        raise ValueError("The file could not be read as CSV.") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"semana", "casos"} <= set(df.columns):
        raise ValueError("The CSV must have the columns `semana` and `casos`.")
    df = df[["semana", "casos"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < SEMANAS_TEST_MIN + 2:
        raise ValueError(f"The CSV needs at least {SEMANAS_TEST_MIN + 2} valid weeks.")
    return df.reset_index(drop=True)


def pronostico_baseline(df, semanas_test=SEMANAS_TEST_DEFECTO):
    """Baseline: next week = this week. Time-based split: the last weeks are used for testing."""
    semanas_test = int(np.clip(semanas_test, SEMANAS_TEST_MIN, min(SEMANAS_TEST_MAX, len(df) - 1)))
    test = df.iloc[-semanas_test:]
    pred = df["casos"].shift(1).iloc[-semanas_test:]
    error = test["casos"] - pred
    return {
        "semanas_test": semanas_test,
        "n_train": len(df) - semanas_test,
        "n_test": semanas_test,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "semanas": df["semana"].tolist(),
        "real": df["casos"].tolist(),
        "predicho": [None] * (len(df) - semanas_test) + pred.tolist(),
        "serie_json": df.to_json(orient="records"),
    }