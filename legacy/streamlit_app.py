"""Weekly dengue forecasting dashboard (Streamlit), deployed on Render.

Tab "Project structure": shows docs/project_structure.md.
Tab "Forecast": without uploaded data it uses a synthetic demo series. With a
CSV (columns `week`, `cases`) it computes the baseline and its error on the latest weeks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DOC = Path(__file__).resolve().parent.parent / "docs" / "project_structure.md"

st.set_page_config(page_title="Dengue forecasting", page_icon="🦟", layout="wide")
st.title("Weekly Dengue Case Forecasting in Colombia")
st.caption("Machine Learning final project · SIVIGILA (datos.gov.co) + IDEAM weather")


@st.cache_data
def demo_series(weeks: int = 156) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.arange(weeks)
    cases = 40 + 30 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 6, weeks)
    return pd.DataFrame({"week": t + 1, "cases": np.clip(cases, 0, None).round()})


tab_project, tab_forecast = st.tabs(["Project structure", "Forecast"])

with tab_project:
    st.markdown(DOC.read_text(encoding="utf-8"))

with tab_forecast:
    file = st.file_uploader("CSV with columns `week` and `cases`", type="csv")
    test_weeks = st.slider("Test weeks (time-based split)", 8, 52, 26)

    df = pd.read_csv(file) if file else demo_series()
    if not file:
        st.info("Showing synthetic demo data. Upload a CSV to use real data.")

    train, test = df.iloc[:-test_weeks], df.iloc[-test_weeks:]

    # Baseline: next week = this week.
    pred = df["cases"].shift(1).iloc[-test_weeks:]
    mae = (test["cases"] - pred).abs().mean()
    rmse = float(np.sqrt(((test["cases"] - pred) ** 2).mean()))

    c1, c2 = st.columns(2)
    c1.metric("MAE (baseline)", f"{mae:.1f} cases")
    c2.metric("RMSE (baseline)", f"{rmse:.1f} cases")

    chart = df.set_index("week")[["cases"]].rename(columns={"cases": "Actual"})
    chart["Predicted (baseline)"] = np.nan
    chart.loc[test["week"].values, "Predicted (baseline)"] = pred.values
    st.line_chart(chart)
    st.caption(f"Train: {len(train)} weeks · Test: {len(test)} weeks (latest).")