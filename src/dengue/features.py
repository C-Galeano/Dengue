"""From case records to a weekly time series with lag and climate features."""
import pandas as pd
from epiweeks import Week

from dengue.config import TARGET


def epiweek_start(year, week):
    """Start date (Sunday) of an epidemiological week, as used by SIVIGILA."""
    return pd.Timestamp(Week(int(year), int(week)).startdate())


def build_weekly_series(cases, by=None):
    """Count cases per epidemiological week (optionally per group, e.g. 'commune').

    Weeks with no reported cases are filled with 0 so the series has no gaps.
    """
    group_cols = [by] if isinstance(by, str) else list(by or [])
    counts = cases.groupby(group_cols + ["year", "week"]).size().rename(TARGET).reset_index()
    counts["date"] = [epiweek_start(y, w) for y, w in zip(counts["year"], counts["week"])]

    full_dates = pd.date_range(counts["date"].min(), counts["date"].max(), freq="7D")

    def fill(frame):
        return frame.set_index("date")[[TARGET]].reindex(full_dates, fill_value=0)

    if not group_cols:
        series = fill(counts)
    else:
        series = counts.groupby(group_cols).apply(fill, include_groups=False)
        series = series.reset_index(level=group_cols)
    series.index.name = "date"
    return series


def climate_to_epiweeks(daily, date_col="date", agg=None):
    """Convert daily IDEAM records (one row per day and station) to epidemiological weeks.

    Stations are averaged per day first; then rainfall is summed and temperature averaged
    per week. Pass agg to override, e.g. {'rainfall': 'sum', 'temperature': 'mean'}.
    """
    daily = daily.copy()
    daily[date_col] = pd.to_datetime(daily[date_col])
    value_cols = [c for c in daily.select_dtypes("number").columns]
    per_day = daily.groupby(date_col)[value_cols].mean()

    weeks = [Week.fromdate(d) for d in per_day.index]
    per_day.index = [pd.Timestamp(w.startdate()) for w in weeks]
    agg = agg or {c: ("sum" if "rain" in c or "precip" in c else "mean") for c in value_cols}
    weekly = per_day.groupby(level=0).agg(agg)
    weekly.index.name = "date"
    return weekly


def add_lags(series, columns, lags):
    """Add lagged copies of columns, e.g. rainfall_lag3 = rainfall from 3 weeks ago."""
    out = series.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_calendar(series):
    """Epidemiological week number, so models can learn the yearly seasonality."""
    out = series.copy()
    out["epi_week"] = [Week.fromdate(d).week for d in out.index]
    return out


def build_features(weekly, climate=None, case_lags=(1, 2, 3, 4), climate_lags=(1, 2, 3, 4, 6, 8)):
    """Join climate, add case and climate lags plus calendar features; drop incomplete rows."""
    df = weekly.copy()
    lag_climate = []
    if climate is not None:
        df = df.join(climate, how="left")
        lag_climate = list(climate.columns)
    df = add_lags(df, [TARGET], case_lags)
    df = add_lags(df, lag_climate, climate_lags)
    # Only past climate is known at forecast time, so the same-week values are dropped
    df = add_calendar(df.drop(columns=lag_climate))
    return df.dropna()
