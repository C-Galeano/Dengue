"""Web dashboard: weekly dengue series and model comparison.

Run locally:
    streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

from dengue.config import TARGET
from dengue.data import load_cases, standardize_columns
from dengue.evaluation import evaluate, peak_threshold, time_split
from dengue.features import build_features, build_weekly_series, climate_to_epiweeks
from dengue.models import ArimaModel, LagMLModel, NaiveBaseline, SeasonalBaseline

MODELS = {
    "Baseline (semana anterior)": NaiveBaseline,
    "Baseline (promedio de la semana)": SeasonalBaseline,
    "ARIMA": ArimaModel,
    "Random Forest": LagMLModel,
}


def demo_cases(years=(2020, 2021, 2022, 2023), seed=0):
    """Synthetic SIVIGILA-like records with a yearly outbreak, used when no file is uploaded."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for week in range(1, 53):
            n = rng.poisson(25 + 20 * np.sin(2 * np.pi * (week - 27) / 52))  # peak near week 40
            rows += [{"AÑO": year, "SEMANA": week}] * n
    return standardize_columns(pd.DataFrame(rows))


@st.cache_data(show_spinner=False)
def prepare(cases_bytes, climate_bytes):
    cases = load_cases(io.BytesIO(cases_bytes)) if cases_bytes else demo_cases()
    weekly = build_weekly_series(cases)
    climate = climate_to_epiweeks(pd.read_csv(io.BytesIO(climate_bytes))) if climate_bytes else None
    return weekly, build_features(weekly, climate)


@st.cache_data(show_spinner=False)
def compare(features, model_names, test_weeks):
    train, test = time_split(features, test_weeks)
    models = [MODELS[name]() for name in model_names]
    metrics, predictions = evaluate(models, train, test)
    importance = next((m.feature_importance() for m in models if isinstance(m, LagMLModel)), None)
    return metrics, predictions, importance, peak_threshold(train)


st.set_page_config(page_title="Pronóstico de dengue", layout="wide")
st.title("Pronóstico semanal de casos de dengue")
st.caption("Casos SIVIGILA (datos.gov.co) agregados por semana epidemiológica · clima IDEAM opcional")

with st.sidebar:
    st.header("Datos")
    cases_file = st.file_uploader("CSV de casos SIVIGILA (una fila por caso)", type="csv")
    climate_file = st.file_uploader("CSV diario IDEAM (opcional)", type="csv",
                                    help="Columna 'date' y variables numéricas, p. ej. rainfall, temperature")
    if cases_file is None:
        st.info("Sin archivo: se usan datos sintéticos de ejemplo.")

    st.header("Modelos")
    model_names = st.multiselect("Modelos a comparar", list(MODELS), default=list(MODELS))
    test_weeks = st.slider("Semanas de prueba (las más recientes)", 8, 52, 26)

try:
    weekly, features = prepare(cases_file.getvalue() if cases_file else None,
                               climate_file.getvalue() if climate_file else None)
except ValueError as err:
    st.error(f"No se pudo leer el archivo: {err}")
    st.stop()

peak_date = weekly[TARGET].idxmax()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Casos totales", f"{int(weekly[TARGET].sum()):,}")
col2.metric("Semanas", len(weekly))
col3.metric("Pico semanal", int(weekly[TARGET].max()))
col4.metric("Semana del pico", peak_date.strftime("%Y-%m-%d"))

st.subheader("Casos por semana epidemiológica")
st.line_chart(weekly[TARGET].rename("Casos"))

st.subheader("Comparación de modelos")
if not model_names:
    st.warning("Selecciona al menos un modelo.")
    st.stop()
if test_weeks >= len(features):
    st.error(f"Hay {len(features)} semanas con variables completas; reduce las semanas de prueba.")
    st.stop()

with st.spinner("Entrenando modelos..."):
    metrics, predictions, importance, threshold = compare(features, tuple(model_names), test_weeks)

st.markdown(
    f"Entrenamiento con las semanas anteriores y prueba con las últimas **{test_weeks}**. "
    f"Una semana de brote es la que supera **{threshold:.0f} casos** (percentil 90 del entrenamiento)."
)
st.dataframe(
    metrics.rename(columns={"peak_recall": "Brotes detectados", "peak_precision": "Precisión en brotes"}),
    column_config={
        "MAE": st.column_config.NumberColumn(format="%.1f"),
        "RMSE": st.column_config.NumberColumn(format="%.1f"),
        "Brotes detectados": st.column_config.NumberColumn(format="percent"),
        "Precisión en brotes": st.column_config.NumberColumn(format="percent"),
    },
)
st.caption("MAE / RMSE: error promedio en casos por semana (menor es mejor). "
           "Todo modelo debe superar a los baselines.")

st.subheader("Real vs pronosticado (semanas de prueba)")
st.line_chart(predictions.rename(columns={"real": "Real"}))

if importance is not None:
    st.subheader("Variables más importantes (Random Forest)")
    st.bar_chart(importance.head(10).rename("Importancia"), horizontal=True)
