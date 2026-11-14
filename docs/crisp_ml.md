# CRISP-ML(Q) Methodology

The project follows **CRISP-ML(Q)** (*CRoss-Industry Standard Process for the development of Machine Learning applications
with Quality assurance methodology*; Studer et al., 2021). It adapts CRISP-DM to machine learning:

- **Business and data understanding are one phase**: whether the data can support the goal decides whether the project is
  feasible.
- **Every phase identifies its risks** and how they are checked (the "Q", quality assurance).
- A final phase **monitors and maintains** the model after deployment.

The process is **iterative**: the results of one phase can send the work back to an earlier one. Each phase has a notebook in
`notebooks/` and a page in the web app (`/methodology/<n>`).

```text
 1. Business & Data Understanding → 2. Data Engineering → 3. Model Engineering
            ↑                                                      ↓
 6. Monitoring & Maintenance ← 5. Deployment ← 4. Quality Assurance
```

| # | Phase | Notebook | Code / artifacts |
|---|-------|----------|------------------|
| 1 | Business and Data Understanding | `01_business_and_data_understanding.ipynb` | `docs/project_structure.md` (original proposal), `src/dbc.py`, `src/sinan.py` |
| 2 | Data Engineering | `02_data_engineering.ipynb` | `src/sinan.py`, `src/preprocessing.py`, `src/tulua.py` |
| 3 | Machine Learning Model Engineering | `03_model_engineering.ipynb` | `src/train.py` |
| 4 | Quality Assurance for ML Applications | `04_quality_assurance.ipynb` | `src/train.py`, `src/explain.py`, `models/metrics.json` |
| 5 | Deployment | `05_deployment.ipynb` | `app.py`, `src/prediction.py`, `src/database.py`, `app/`, `render.yaml` |
| 6 | Monitoring and Maintenance | `06_monitoring_maintenance.ipynb` | `src/monitoring.py`, `models/monitoring.json` |

## Iterations

| Iteration | Objective | Data | Outcome |
|---|---|---|---|
| 0 | Forecast weekly dengue cases | SIVIGILA case series | Kept as the baseline forecast on the home page |
| 1 | Binary screening: dengue yes/no | SIVIGILA Tuluá (Colombia) | ROC-AUC ≈ 0.63; judged too narrow |
| **2** | **Arbovirus differential diagnosis + explainable AI ("smart hospital")** | **SINAN (Brazil)**; Tuluá as external validation | Current version |

---

## 1. Business and Data Understanding

**Business understanding**
- **Problem:** dengue, chikungunya and Zika share the mosquito vector and the initial symptoms. At the first consultation the
  clinician must decide what to suspect and which test to order. Missing dengue is dangerous (severe dengue; NSAIDs used for
  chikungunya are contraindicated in dengue).
- **Business objective:** decision support for the first consultation of a febrile patient, with a transparent explanation.
- **ML objective:** 3-class classifier (dengue / chikungunya / other febrile illness) with per-patient SHAP explanations.
- **Success criteria:** dengue recall ≥ 80 %; macro ROC-AUC ≥ 0.80 (course target); better than a linear baseline; an
  explanation for every prediction; no data leakage.
- **Risks:** overlapping symptoms; a model trained in Brazil may not transfer to Colombia; the class mix changes between years.

**Data understanding**

| Source | Diseases | Symptoms | Lab | Decision |
|---|---|---|---|---|
| SIVIGILA Tuluá, Colombia (`dq2k-gub9`) | Dengue only | Yes | Yes/no | External validation |
| Medellín Zika / Chikungunya (datos.gov.co) | Zika, chikungunya | No | No | Not usable |
| **SINAN, Brazil (DATASUS FTP)** | Dengue + chikungunya (same form); Zika | Yes | Yes/no | **Training** |

Findings:
- Dengue and chikungunya share the SINAN notification form, so their records can be combined.
- The **Zika** form has 38 administrative columns and **no clinical fields** → Zika cannot be modeled from symptoms.
- About half of the cases are closed by **clinical-epidemiological** criterion (from the symptoms themselves) → only
  **laboratory**-closed cases are used.
- Brazil's public **dengue files contain no discarded cases** → "other febrile illness" = chikungunya suspects ruled out by
  the laboratory.
- **Numeric lab values** (lowest platelet count) are empty in 100 % of records → lab information only as yes/no
  (leukopenia, tourniquet test).
- Warning-sign and severity fields are filled only for dengue → excluded (leakage).
- Clear signal: joint pain in 52 % of chikungunya vs 16 % of dengue; leukopenia twice as frequent in dengue.

## 2. Data Engineering

- **Reader:** `src/dbc.py`, a pure-Python reader of DATASUS `.dbc` files (no C compiler needed).
- **Labels:** laboratory criterion only — dengue (`CLASSI_FIN` 10–12), chikungunya (13), other febrile illness (5).
- **26 features:** age, days since onset, sex, 11 symptoms, 3 clinical/laboratory findings, 7 comorbidities and the
  **epidemiological context** (state of residence and month of consultation).
- **Second iteration of this phase:** with clinical features only the macro ROC-AUC was 0.71, below the 0.80 target. Adding
  where and when the patient consults (local circulation, the clinician's pre-test probability) raised it to 0.86. Caveat:
  part of the state effect may reflect each state's laboratory practices.
- **Balanced samples:** 40,000 cases per class for 2023 (training/test) and 15,000 per class for 2025 (temporal validation).
- **Colombian mapping:** `src/tulua.py` maps the 7 equivalent SIVIGILA symptoms; the rest are "not reported".

## 3. Machine Learning Model Engineering

- 80 % train / 20 % test, stratified; 5-fold stratified cross-validation.
- Logistic Regression (baseline) vs **Random Forest, XGBoost, LightGBM**.
- Selection by macro ROC-AUC (one-vs-rest); ties broken by log-loss.
- **Dengue alert:** dengue is suggested whenever P(dengue) ≥ a threshold chosen for dengue recall ≥ 80 % on out-of-fold
  predictions.
- **Reproducibility:** fixed seeds; scikit-learn, XGBoost and LightGBM versions pinned and saved with the model.

## 4. Quality Assurance for ML Applications

- Test metrics per diagnosis, confusion matrix and one-vs-rest ROC curves.
- **SHAP (TreeSHAP)** global importance by diagnosis and per-patient explanations.
- **Temporal validation:** trained on 2023, evaluated on 2025.
- **External validation:** P(dengue) on Colombian SIVIGILA data (Tuluá).
- Results: LightGBM, test macro ROC-AUC 0.86 (dengue 0.90, chikungunya 0.87, other 0.81), dengue recall 0.80; temporal
  validation 2025: 0.76; Colombia (Tuluá): 0.56, the model does not transfer.
- All results in `models/metrics.json` and on the `/about` page.

## 5. Deployment

- Flask "smart hospital" module: consultation form → probabilities, suggested diagnosis, SHAP explanation and next steps.
- Login for clinical staff; patient clinical history can be updated and the model re-run.
- XGBoost/LightGBM native TreeSHAP: no `shap` package on the server.
- SQLAlchemy `consultations` table (findings, probabilities and explanation as JSON); SQLite locally, PostgreSQL on Render.

## 6. Monitoring and Maintenance

`python -m src.monitoring` compares the training year (2023) with a newer year (2025) and saves `models/monitoring.json`:

| Signal | Measure | Retraining rule | 2023 → 2025 |
|---|---|---|---|
| Data drift | Population Stability Index per feature | PSI > 0.25 in a top-5 SHAP feature | State of residence PSI 1.39 (significant); month 0.11; symptoms < 0.02 |
| Prior drift | Share of each lab-confirmed diagnosis | Change > 5 percentage points | Dengue 85.1 % → 77.4 %; other febrile 7.6 % → 12.9 % |
| Performance drift | Macro ROC-AUC on new labeled cases | Below 0.80 | 0.86 → 0.76 |

Decision: **retrain with recent data**. Patients present the same way (symptoms are stable); what changed is where and when
lab-confirmed cases occur, which is the model's most important feature.

Maintenance plan: retrain when a rule fires and at least yearly (`python -m src.sinan`, `python -m src.train`,
`python -m src.monitoring`); store the laboratory result of each consultation to build a local labeled dataset; keep the
decision-support notice.

## Next iteration

Numeric laboratory values (platelets, white cells, hematocrit) from hospital records, retraining with recent and Colombian
data, and a Zika class once a source with clinical fields is available.

## Reference

Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K.-R. (2021). Towards CRISP-ML(Q):
a machine learning process model with quality assurance methodology. *Machine Learning and Knowledge Extraction, 3*(2),
392–413.
