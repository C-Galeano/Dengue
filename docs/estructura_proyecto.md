# PROJECT STRUCTURE

## PROJECT TITLE

Weekly Dengue Case Forecasting in Colombia Using Machine Learning

## PROBLEM STATEMENT AND JUSTIFICATION

Dengue is one of Colombia's most pressing public health problems: outbreaks return year after year and overwhelm hospitals and health services, especially in warm, low-altitude municipalities. Today the response is **reactive**: cases are counted after they happen, resources arrive once the peak has started, and prevention campaigns are launched late, when the outbreak is much harder to contain.

**Research question:** can we anticipate, several weeks ahead, the number of weekly dengue cases using historical SIVIGILA records and climate variables (rainfall and temperature from IDEAM)?

**Justification:** a weekly forecast makes it possible to plan staff, beds and supplies in advance and to launch vector control and prevention in time. The data are open (datos.gov.co), weather directly drives mosquito reproduction and transmission, and time-series and machine learning techniques can turn these records into a simple, reproducible early-warning tool.

## PROJECT OBJECTIVES

### GENERAL OBJECTIVE

Develop and compare machine learning and time-series models that forecast weekly dengue cases in Colombian municipalities, and make them available through an early-warning web dashboard.

### SPECIFIC OBJECTIVES

1. Download, clean and unify dengue case records from SIVIGILA (datos.gov.co) for municipalities such as Bucaramanga, Tuluá and Pereira, aligning columns across sources.
2. Build the weekly series (per epidemiological week and commune) and add lagged features for cases, rainfall and temperature (IDEAM).
3. Train and compare a baseline, ARIMA, Prophet and lag-based ML models (Random Forest / XGBoost), requiring every model to beat the baseline.
4. Evaluate with a time-based split (train: earlier weeks; test: latest weeks) using MAE, RMSE, peak detection and real vs. predicted plots.
5. Identify which climate lags matter most and deploy a web dashboard (Render) showing forecasts and alerts.

## REFERENCE FRAMEWORK (Optional)

- **Epidemiological:** dengue as a vector-borne disease (*Aedes aegypti*); public health surveillance with SIVIGILA (INS) and epidemiological weeks.
- **Climate:** rainfall and temperature affect the mosquito life cycle; their effects appear with a delay (e.g., rainfall from 3 weeks ago can explain today's cases).
- **Technical:** time series (ARIMA, Prophet), supervised learning with lags (Random Forest, XGBoost), time-based validation and MAE/RMSE metrics.
- **Data:** datos.gov.co (SIVIGILA, municipal datasets) and IDEAM (daily data per station).

## METHODOLOGY AND DEVELOPMENT TIMELINE

**Methodology**

1. **Data:** download through the Socrata API, column standardization per municipality, and conversion of daily climate data to epidemiological weeks.
2. **Series and features:** case counts per week (and commune) and lagged variables for cases, rainfall and temperature.
3. **Modeling:** baseline (next week = this week, or the historical average for that week), ARIMA, Prophet and Random Forest/XGBoost.
4. **Evaluation:** time-based split (never random) to avoid optimistic results; MAE, RMSE, peak detection and real vs. predicted plots.
5. **Delivery:** web dashboard deployed on Render, with reproducible code (`src/dengue`, tests with `pytest`).

**Timeline (16 weeks)**

| Weeks | Activity |
|-------|----------|
| 1–3 | Download and clean the datasets |
| 4–6 | Build the weekly series and lag features |
| 7–11 | Train and compare models |
| 12–14 | Evaluation, analysis of climate lags and peaks |
| 15–16 | Web dashboard on Render, synthesis document and final presentation |

**Risks:** short series (little seasonal history), different formats across municipalities, joining daily per-station climate data with epidemiological weeks, and careful validation to avoid overestimating performance.

## BIBLIOGRAPHIC REFERENCES (optional)

- Instituto Nacional de Salud. *Sistema de Vigilancia en Salud Pública – SIVIGILA*. Colombia.
- Datos Abiertos Colombia (datos.gov.co). Dengue surveillance datasets.
- IDEAM. Hydrometeorological data (rainfall and temperature).
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1).
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

---

## EXPECTED PRODUCTS (mark with an X)

| X | Product |
|---|---------|
| X | PGC synthesis document |
| X | Software |
|   | Information system |
| X | Prototype |
|   | Scientific article |
|   | Poster / Presentation |
|   | Community of practice and knowledge networks |
|   | Scientific or academic event |
|   | Journal editions or outreach book |
|   | Digital content production |
|   | Non-specialized editorial publications |
| X | Web development |
|   | Course creation |
|   | Entrepreneurship and innovation |
|   | Other. Which? |
