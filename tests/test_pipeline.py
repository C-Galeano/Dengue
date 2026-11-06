"""Smoke test of the full pipeline on synthetic SIVIGILA-like records."""
import numpy as np
import pandas as pd

from dengue.evaluation import evaluate, time_split
from dengue.features import build_features, build_weekly_series, climate_to_epiweeks
from dengue.data import standardize_columns
from dengue.models import NaiveBaseline, SeasonalBaseline, ArimaModel, LagMLModel


def synthetic_cases(years=(2021, 2022, 2023), seed=0):
    """Seasonal weekly counts expanded into one row per case, with Spanish column names."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for week in range(1, 53):
            n = rng.poisson(20 + 15 * np.sin(2 * np.pi * week / 52))
            rows += [{"AÑO": year, "SEMANA": week, "comuna": rng.integers(1, 4)}] * n
    return pd.DataFrame(rows)


def test_standardize_columns():
    df = standardize_columns(pd.DataFrame(columns=["AÑO", "SEMANA", "Edad_", "comuna"]))
    assert list(df.columns) == ["year", "week", "age", "commune"]


def test_weekly_series_has_no_gaps():
    weekly = build_weekly_series(standardize_columns(synthetic_cases()))
    assert (weekly.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all()
    by_commune = build_weekly_series(standardize_columns(synthetic_cases()), by="commune")
    assert by_commune["commune"].nunique() == 3


def test_models_run_end_to_end():
    weekly = build_weekly_series(standardize_columns(synthetic_cases()))
    days = pd.date_range(weekly.index.min(), weekly.index.max(), freq="D")
    climate = climate_to_epiweeks(pd.DataFrame({
        "date": days, "rainfall": np.random.default_rng(1).gamma(2, 3, len(days)), "temperature": 26.0,
    }))
    features = build_features(weekly, climate)
    assert "rainfall_lag3" in features and "rainfall" not in features

    train, test = time_split(features, test_weeks=20)
    metrics, predictions = evaluate([NaiveBaseline(), SeasonalBaseline(), ArimaModel(), LagMLModel()], train, test)
    assert len(metrics) == 4 and metrics["MAE"].notna().all()
    assert len(predictions) == 20
