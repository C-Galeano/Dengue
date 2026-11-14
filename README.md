<div align="center">

# 🦟 Arbovirus Differential Diagnosis with Explainable AI

**Machine Learning final project** · a "smart hospital" module that suggests, for a febrile patient, whether the illness is
**dengue**, **chikungunya** or **another febrile illness** — and explains why

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-1f6feb)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2c6e49)
![SHAP](https://img.shields.io/badge/explainability-SHAP-8a3ffc)
![Methodology](https://img.shields.io/badge/methodology-CRISP--ML(Q)-2c6e49)

</div>

Dengue, chikungunya and Zika share the same mosquito and almost the same first symptoms. At the first consultation a
clinician must decide what to suspect, which test to order and which warning signs to watch. This project:

1. **Differential diagnosis** (main deliverable): Random Forest, XGBoost and LightGBM trained on **laboratory-confirmed**
   surveillance cases estimate the probability of each diagnosis from age, sex, days since onset, 11 symptoms, 3 clinical and
   laboratory findings, 7 comorbidities and the epidemiological context (state of residence and month).
2. **Explainability**: every result shows the **SHAP** contribution of each finding, so the clinician sees *why*.
3. **Weekly forecasting** (baseline module): weekly dengue case series and a baseline forecast on the home page.

> ⚠️ Academic project. Clinical **decision support** only; it does **not** replace medical diagnosis.

## Methodology: CRISP-ML(Q)

The project follows **CRISP-ML(Q)** (Studer et al., 2021), the machine-learning version of CRISP-DM: business and data
understanding form one phase, every phase checks its risks (quality assurance), and a final phase monitors the deployed model.
Each phase has a notebook and a page in the web app (`/methodology/<n>`) explaining the phase and its implementation.
Full description: [`docs/crisp_ml.md`](docs/crisp_ml.md).

| # | Phase | Notebook | Code / artifacts |
|---|-------|----------|------------------|
| 1 | **Business and Data Understanding** | [`01_business_and_data_understanding`](notebooks/01_business_and_data_understanding.ipynb) | [`docs/project_structure.md`](docs/project_structure.md) (original proposal), `src/dbc.py`, `src/sinan.py` |
| 2 | **Data Engineering** | [`02_data_engineering`](notebooks/02_data_engineering.ipynb) | `src/sinan.py`, `src/preprocessing.py`, `src/tulua.py` |
| 3 | **Machine Learning Model Engineering** | [`03_model_engineering`](notebooks/03_model_engineering.ipynb) | `src/train.py` |
| 4 | **Quality Assurance for ML Applications** | [`04_quality_assurance`](notebooks/04_quality_assurance.ipynb) | `src/explain.py`, `models/metrics.json` |
| 5 | **Deployment** | [`05_deployment`](notebooks/05_deployment.ipynb) | `app.py`, `src/prediction.py`, `src/database.py`, `app/`, `render.yaml` |
| 6 | **Monitoring and Maintenance** | [`06_monitoring_maintenance`](notebooks/06_monitoring_maintenance.ipynb) | `src/monitoring.py`, `models/monitoring.json` |

### Results at a glance

| | |
|---|---|
| Data | 120,000 laboratory-confirmed cases, SINAN Brazil 2023 (40,000 per class) |
| Features | 26: age, sex, days since onset, 11 symptoms, leukopenia, tourniquet test, petechiae, 7 comorbidities + epidemiological context (state of residence, month) |
| Model comparison (CV macro ROC-AUC) | Logistic Regression 0.704 · Random Forest 0.839 · XGBoost 0.857 · LightGBM 0.860 |
| Selected model | LightGBM, dengue alert when P(dengue) ≥ 0.34 |
| **Test set** | **macro ROC-AUC 0.86** (target ≥ 0.80) · balanced accuracy 0.69 (chance 0.33) |
| Recall / ROC-AUC per diagnosis | Dengue 0.80 / 0.90 · Chikungunya 0.72 / 0.87 · Other febrile illness 0.54 / 0.81 |
| Effect of the context | clinical features only: macro ROC-AUC 0.71 → with state and month: above 0.85 |
| Top SHAP features | State of residence, Joint pain, Days since symptom onset, Age, Month of consultation, Joint swelling (arthritis) |
| Temporal validation | trained 2023 → 2025: macro ROC-AUC 0.76 (circulation changes between years) |
| External validation (Colombia) | Tuluá dengue vs discarded: ROC-AUC 0.56 → does not transfer; retrain with local data |
| Monitoring 2023 → 2025 | State of residence PSI 1.39 (significant drift), symptoms stable; 3 of 3 retraining rules fire |

## Data

| Source | Use |
|---|---|
| **SINAN, Brazil** (DATASUS FTP, `DENGBR` / `CHIKBR` final files) | Training (2023) and temporal validation (2025). Dengue and chikungunya share the same notification form. |
| **SIVIGILA Tuluá, Colombia** (datos.gov.co `dq2k-gub9`) | External validation of the dengue signal |
| datos.gov.co Medellín Zika / Chikungunya | Reviewed and discarded: no clinical fields |

Key data decisions (details in Phase 1):
- Only cases closed by **laboratory** criterion, so the label does not depend on the symptoms the model uses.
- **Zika is not modeled**: its public form has no clinical fields.
- **Other febrile illness** = chikungunya suspects ruled out by the laboratory (Brazil's public dengue files contain no
  discarded cases).
- No public source has **numeric** lab values; leukopenia and the tourniquet test enter as yes/no.

Raw SINAN (Portuguese) and SIVIGILA (Spanish) column names appear only in the mappings that read the raw files.

## Web App (Phase 5)

| Route | Page |
|-------|------|
| `/login` | Sign in for clinical staff (`APP_USER` / `APP_PASSWORD` environment variables) |
| `/` | Home: project structure and baseline weekly forecast |
| `/register` | New consultation: patient data, symptoms, findings, comorbidities |
| `/result/<id>` | Probability of each diagnosis, suggested diagnosis, SHAP explanation and next steps |
| `/patients` | Consultations, filterable by suggested diagnosis |
| `/about` | Model card: metrics per disease, SHAP importance, temporal and Colombian validation, limitations |
| `/methodology`, `/methodology/1` … `/6` | One page per CRISP-ML(Q) phase |

## Getting Started

```bash
python -m venv venv
venv\Scripts\activate            # macOS / Linux: source venv/bin/activate
pip install -r requirements-app.txt
pip install -e .
pytest
flask run --debug                # http://127.0.0.1:5000
```

The trained model (`models/model.pkl`) and the processed datasets (`data/processed/`) are versioned, so the app and the
notebooks run without downloading anything. To rebuild everything from the original sources:

```bash
python -m src.sinan --years 2023 2025   # downloads SINAN from DATASUS (~300 MB) and builds the datasets (~15 min)
python -m src.train                     # trains, evaluates and saves models/model.pkl + models/metrics.json
python -m src.monitoring                # drift report 2023 vs 2025 -> models/monitoring.json
```

The external validation needs `data/raw/dengue_tulua.csv` (datos.gov.co `dq2k-gub9`); without it, it is skipped.

## Forecasting Module (baseline)

```text
Case records ──► Aggregate per epi week (and commune) ──► Weekly series ──► Lags of cases + climate
```

| Step | Module |
|------|--------|
| Download / load / align columns across municipalities | `src/dengue/data.py` |
| Weekly series, IDEAM daily → epi weeks, lag features | `src/dengue/features.py` |
| Baseline, ARIMA, Prophet, Random Forest, XGBoost | `src/dengue/models.py` |
| Time split, MAE / RMSE, peak detection, real vs predicted plot | `src/dengue/evaluation.py` |
| End-to-end run | `src/dengue/pipeline.py` |

```bash
python -m dengue.pipeline data/raw/bucaramanga.csv --test-weeks 26
```

## Project Structure

```text
dengue-forecasting/
├── app.py                 # Flask routes (Phase 5)
├── app/
│   ├── static/styles.css  # Visual theme
│   └── templates/         # Jinja templates (phases/ = one page per CRISP-ML(Q) phase)
├── data/
│   ├── raw/               # Untouched downloads (not versioned): sinan/*.dbc, dengue_tulua.csv
│   ├── processed/         # arbovirus_2023/2025.parquet + raw-count summaries
│   └── external/          # IDEAM climate
├── docs/
│   ├── crisp_ml.md        # The six CRISP-ML(Q) phases of the project
│   └── project_structure.md  # Original project proposal
├── models/                # model.pkl + metrics.json + monitoring.json
├── notebooks/             # 01_business_and_data_understanding … 06_monitoring_maintenance (one per CRISP-ML(Q) phase)
├── src/
│   ├── dbc.py             # Phase 1: pure-Python reader of DATASUS .dbc files
│   ├── sinan.py           # Phases 1–2: download SINAN, labels, features, balanced samples
│   ├── preprocessing.py   # Phase 2: feature definitions shared by training and the app
│   ├── tulua.py           # Phase 4: Colombian external validation data
│   ├── train.py           # Phases 3–4: models, decision rule, evaluation
│   ├── explain.py         # Phases 4–5: SHAP values (TreeSHAP)
│   ├── prediction.py      # Phase 5: probabilities, dengue alert, explanation, next steps
│   ├── validation.py      # Phase 5: server-side form validation
│   ├── database.py        # Phase 5: consultations table (SQLAlchemy)
│   ├── app_factory.py     # Phase 5: Flask application factory
│   ├── monitoring.py      # Phase 6: drift (PSI), retraining rules
│   ├── methodology.py     # CRISP-ML(Q) phase pages: metadata and real code excerpts
│   ├── forecast.py        # Baseline weekly forecast for the home page
│   └── dengue/            # Forecasting package
└── tests/                 # App, model and pipeline tests
```

## Next Iteration (CRISP-ML(Q) cycle)

1. Numeric laboratory values (platelets, white cells, hematocrit) from hospital records.
2. Retrain with Colombian data; add Zika once a source with clinical fields is available.
3. Store the laboratory result of each consultation to monitor drift and build a local labeled dataset.
