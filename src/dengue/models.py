"""Models to compare. Every model has fit(train) and predict(test) on the feature frame.

train / test are DataFrames indexed by week with the 'cases' column plus the features from
features.build_features. Baseline, ARIMA and lag-based ML forecast one step ahead (each week
uses the real cases up to the previous week); Prophet forecasts the whole test horizon.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from dengue.config import TARGET


class NaiveBaseline:
    """Next week = this week."""
    name = "Baseline (last week)"

    def fit(self, train):
        return self

    def predict(self, test):
        return test[f"{TARGET}_lag1"].to_numpy()


class SeasonalBaseline:
    """Next week = historical average for that epidemiological week."""
    name = "Baseline (week average)"

    def fit(self, train):
        self.by_week = train.groupby("epi_week")[TARGET].mean()
        self.overall = train[TARGET].mean()
        return self

    def predict(self, test):
        return test["epi_week"].map(self.by_week).fillna(self.overall).to_numpy()


class ArimaModel:
    """ARIMA learned from the case series itself."""
    name = "ARIMA"

    def __init__(self, order=(2, 1, 2)):
        self.order = order

    def fit(self, train):
        from statsmodels.tsa.arima.model import ARIMA
        self.result = ARIMA(train[TARGET].to_numpy(dtype=float), order=self.order).fit()
        return self

    def predict(self, test):
        # Keep the trained parameters and feed the test weeks one by one
        extended = self.result.append(test[TARGET].to_numpy(dtype=float), refit=False)
        start = len(extended.data.endog) - len(test)
        return np.clip(extended.predict(start=start), 0, None)


class ProphetModel:
    """Prophet: yearly seasonality handled automatically."""
    name = "Prophet"

    def fit(self, train):
        from prophet import Prophet
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(train[[TARGET]].rename_axis("ds").reset_index().rename(columns={TARGET: "y"}))
        return self

    def predict(self, test):
        future = test.rename_axis("ds").reset_index()[["ds"]]
        return np.clip(self.model.predict(future)["yhat"].to_numpy(), 0, None)


class LagMLModel:
    """Random Forest (or XGBoost) on past cases, climate lags and epidemiological week."""

    def __init__(self, estimator=None, name="Random Forest"):
        self.estimator = estimator or RandomForestRegressor(n_estimators=300, random_state=42)
        self.name = name

    def fit(self, train):
        self.features = [c for c in train.columns if c != TARGET]
        self.estimator.fit(train[self.features], train[TARGET])
        return self

    def predict(self, test):
        return self.estimator.predict(test[self.features])

    def feature_importance(self):
        """Which lags matter most (e.g. rainfall 3 weeks ago)."""
        import pandas as pd
        return pd.Series(self.estimator.feature_importances_, index=self.features).sort_values(ascending=False)


def xgboost_model():
    from xgboost import XGBRegressor
    return LagMLModel(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42), name="XGBoost")


def default_models():
    """All models in the comparison; Prophet/XGBoost are skipped if not installed."""
    models = [NaiveBaseline(), SeasonalBaseline(), ArimaModel(), LagMLModel()]
    try:
        models.append(xgboost_model())
    except ImportError:
        pass
    try:
        import prophet  # noqa: F401
        models.append(ProphetModel())
    except ImportError:
        pass
    return models
