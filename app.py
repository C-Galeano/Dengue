"""Dashboard de pronóstico semanal de dengue (Streamlit) para desplegar en Render.

Pestaña "Estructura del proyecto": muestra docs/estructura_proyecto.md.
Pestaña "Pronóstico": sin datos cargados usa una serie sintética de demostración. Con un
CSV (columnas `semana`, `casos`) se calcula el modelo base y el error en las últimas semanas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DOC = Path(__file__).parent / "docs" / "estructura_proyecto.md"

st.set_page_config(page_title="Pronóstico de dengue", page_icon="🦟", layout="wide")
st.title("Pronóstico semanal de casos de dengue en Colombia")
st.caption("Proyecto final de Machine Learning · SIVIGILA (datos.gov.co) + clima IDEAM")


@st.cache_data
def demo_series(weeks: int = 156) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.arange(weeks)
    casos = 40 + 30 * np.sin(2 * np.pi * t / 52) + 0.1 * t + rng.normal(0, 6, weeks)
    return pd.DataFrame({"semana": t + 1, "casos": np.clip(casos, 0, None).round()})


tab_proyecto, tab_pronostico = st.tabs(["Estructura del proyecto", "Pronóstico"])

with tab_proyecto:
    st.markdown(DOC.read_text(encoding="utf-8"))

with tab_pronostico:
    archivo = st.file_uploader("CSV con columnas `semana` y `casos`", type="csv")
    test_weeks = st.slider("Semanas de prueba (partición temporal)", 8, 52, 26)

    df = pd.read_csv(archivo) if archivo else demo_series()
    if not archivo:
        st.info("Mostrando datos sintéticos de demostración. Sube un CSV para usar datos reales.")

    train, test = df.iloc[:-test_weeks], df.iloc[-test_weeks:]

    # Modelo base: la semana siguiente = la semana actual.
    pred = df["casos"].shift(1).iloc[-test_weeks:]
    mae = (test["casos"] - pred).abs().mean()
    rmse = float(np.sqrt(((test["casos"] - pred) ** 2).mean()))

    c1, c2 = st.columns(2)
    c1.metric("MAE (modelo base)", f"{mae:.1f} casos")
    c2.metric("RMSE (modelo base)", f"{rmse:.1f} casos")

    chart = df.set_index("semana")[["casos"]].rename(columns={"casos": "Real"})
    chart["Predicho (base)"] = np.nan
    chart.loc[test["semana"].values, "Predicho (base)"] = pred.values
    st.line_chart(chart)
    st.caption(f"Entrenamiento: {len(train)} semanas · Prueba: {len(test)} semanas (últimas).")
