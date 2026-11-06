"""Weekly dengue forecasting dashboard (Streamlit), deployed on Render.

Tab "Project structure": shows docs/estructura_proyecto.md.
Tab "Forecast": without uploaded data it uses a synthetic demo series. With a
CSV (columns `semana`, `casos`) it computes the baseline and its error on the latest weeks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DOC = Path(__file__).parent / "docs" / "estructura_proyecto.md"

st.set_page_config(page_title="Dengue forecasting", page_icon="🦟", layout="wide")
st.title("Weekly Dengue Case Forecasting in Colombia")
st.caption("Machine Learning final project · SIVIGILA (datos.gov.co) + clima IDEAM")


@st.cache_data
def demo_series(weeks: int = 156) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.arange(weeks)
    casos = 40 + 30 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 6, weeks)
    return pd.DataFrame({"semana": t + 1, "casos": np.clip(casos, 0, None).round()})


tab_proyecto, tab_pronostico = st.tabs(["Project structure", "Forecast"])

with tab_proyecto:
    st.markdown(DOC.read_text(encoding="utf-8"))

with tab_pronostico:
    archivo = st.file_uploader("CSV with columns `semana` and `casos`", type="csv")
    test_weeks = st.slider("Test weeks (time-based split)", 8, 52, 26)

    df = pd.read_csv(archivo) if archivo else demo_series()
    if not archivo:
        st.info("Showing synthetic demo data. Upload a CSV to use real data.")

    train, test = df.iloc[:-test_weeks], df.iloc[-test_weeks:]

    # Baseline: next week = this week.
    pred = df["casos"].shift(1).iloc[-test_weeks:]
    mae = (test["casos"] - pred).abs().mean()
    rmse = float(np.sqrt(((test["casos"] - pred) ** 2).mean()))

    c1, c2 = st.columns(2)
    c1.metric("MAE (baseline)", f"{mae:.1f} cases")
    c2.metric("RMSE (baseline)", f"{rmse:.1f} cases")

    chart = df.set_index("semana")[["casos"]].rename(columns={"casos": "Actual"})
    chart["Predicted (baseline)"] = np.nan
    chart.loc[test["semana"].values, "Predicted (baseline)"] = pred.values
    st.line_chart(chart)
    st.caption(f"Train: {len(train)} weeks · Test: {len(test)} weeks (latest).")
