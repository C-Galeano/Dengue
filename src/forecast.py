"""Main page: project structure and weekly baseline forecast.

Reproduces the original Streamlit app (legacy/streamlit_app.py) with the same logic.
"""
import json
from functools import lru_cache

import markdown
import numpy as np
import pandas as pd

from src.preprocessing import ROOT_DIR

PROJECT_DOC = ROOT_DIR / "docs" / "project_structure.md"
TEST_WEEKS_MIN, TEST_WEEKS_MAX, TEST_WEEKS_DEFAULT = 8, 52, 26

# Spanish column names are still accepted so CSV files made for the old version keep working.
COLUMN_ALIASES = {"semana": "week", "casos": "cases"}


@lru_cache
def project_document():
    """Converts docs/project_structure.md to HTML (only once) with Bootstrap styles."""
    html = markdown.markdown(PROJECT_DOC.read_text(encoding="utf-8"), extensions=["tables"])
    return html.replace("<table>", '<table class="table table-sm table-bordered">')


def demo_series(weeks=156):
    """Synthetic series with annual seasonality (same as the original app)."""
    rng = np.random.default_rng(42)
    t = np.arange(weeks)
    cases = 40 + 30 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 6, weeks)
    return pd.DataFrame({"week": t + 1, "cases": np.clip(cases, 0, None).round()})


def read_series(file=None, series_json=None):
    """Reads the series from an uploaded CSV (columns `week` and `cases`) or from the JSON
    sent back by the page when moving the slider. Raises ValueError with a readable message if invalid."""
    try:
        df = pd.read_csv(file) if file else pd.DataFrame(json.loads(series_json))
    except Exception as exc:
        raise ValueError("The file could not be read as CSV.") from exc
    df.columns = [COLUMN_ALIASES.get(c, c) for c in (str(c).strip().lower() for c in df.columns)]
    if not {"week", "cases"} <= set(df.columns):
        raise ValueError("The CSV must have the columns `week` and `cases`.")
    df = df[["week", "cases"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < TEST_WEEKS_MIN + 2:
        raise ValueError(f"The CSV needs at least {TEST_WEEKS_MIN + 2} valid weeks.")
    return df.reset_index(drop=True)


def baseline_forecast(df, test_weeks=TEST_WEEKS_DEFAULT):
    """Baseline: next week = this week. Time-based split: the last weeks are used for testing."""
    test_weeks = int(np.clip(test_weeks, TEST_WEEKS_MIN, min(TEST_WEEKS_MAX, len(df) - 1)))
    test = df.iloc[-test_weeks:]
    pred = df["cases"].shift(1).iloc[-test_weeks:]
    error = test["cases"] - pred
    return {
        "test_weeks": test_weeks,
        "n_train": len(df) - test_weeks,
        "n_test": test_weeks,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "weeks": df["week"].tolist(),
        "actual": df["cases"].tolist(),
        "predicted": [None] * (len(df) - test_weeks) + pred.tolist(),
        "series_json": df.to_json(orient="records"),
    }
