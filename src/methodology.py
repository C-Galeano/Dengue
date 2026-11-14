"""CRISP-ML(Q) methodology pages: phase metadata and the real code excerpts shown on each page.

The explanatory text of each phase lives in app/templates/phases/<number>.html.
Code excerpts are read from the source with `inspect`, so the pages always show the code that actually runs.
"""
import inspect
import textwrap
from functools import lru_cache

from src import dbc, explain, monitoring, prediction, preprocessing, sinan, train, validation
from src.preprocessing import ROOT_DIR

PHASES = [
    {"number": 1, "name": "Business and Data Understanding", "short": "Business & Data", "icon": "target",
     "question": "What problem do we solve, how do we measure success, and can the data support it?",
     "summary": "Problem, objectives and success criteria; public arbovirus sources, label quality and clinical signal.",
     "notebook": "01_business_and_data_understanding.ipynb"},
    {"number": 2, "name": "Data Engineering", "short": "Data Engineering", "icon": "filter",
     "question": "How do we turn raw records into a reliable modeling table?",
     "summary": "Laboratory-confirmed labels, clinical features plus epidemiological context, balanced samples.",
     "notebook": "02_data_engineering.ipynb"},
    {"number": 3, "name": "Machine Learning Model Engineering", "short": "Model Engineering", "icon": "cpu",
     "question": "Which algorithm, and which decision rule?",
     "summary": "Random Forest, XGBoost and LightGBM vs. a linear baseline, and the dengue alert.",
     "notebook": "03_model_engineering.ipynb"},
    {"number": 4, "name": "Quality Assurance for ML Applications", "short": "Quality Assurance", "icon": "shield",
     "question": "Does the model meet the success criteria, and is it robust?",
     "summary": "Test metrics, SHAP explainability, temporal and Colombian external validation.",
     "notebook": "04_quality_assurance.ipynb"},
    {"number": 5, "name": "Deployment", "short": "Deployment", "icon": "server",
     "question": "How do clinicians get access to the model?",
     "summary": "Smart-hospital web app with per-patient SHAP explanations, database and hosting.",
     "notebook": "05_deployment.ipynb"},
    {"number": 6, "name": "Monitoring and Maintenance", "short": "Monitoring", "icon": "activity",
     "question": "When is the deployed model getting worse, and when do we retrain?",
     "summary": "Data, prior and performance drift (2023 vs 2025) and the retraining rules.",
     "notebook": "06_monitoring_maintenance.ipynb"},
]

CODE = {
    1: [dbc.read_dbc, sinan.summarize],
    2: [sinan.build_year, sinan.to_features, preprocessing.age_in_years],
    3: [train.build_models, train.choose_best, train.decide, train.choose_dengue_threshold],
    4: [train.compute_metrics, explain.shap_values, train.external_validation],
    5: [prediction.Predictor.predict, explain.explain_patient, validation.validate_patient],
    6: [monitoring.psi, monitoring.retraining_decision],
}


def get_phase(number):
    """Phase metadata plus its neighbors, or None if the number does not exist."""
    if not 1 <= number <= len(PHASES):
        return None
    return {**PHASES[number - 1],
            "previous": PHASES[number - 2] if number > 1 else None,
            "next": PHASES[number] if number < len(PHASES) else None}


@lru_cache
def code_excerpts(number):
    """List of (title, file, source code) for a phase."""
    excerpts = []
    for item in CODE.get(number, []):
        if isinstance(item, tuple):
            excerpts.append(item)
            continue
        path = inspect.getsourcefile(item)
        rel = path[len(str(ROOT_DIR)) + 1:].replace("\\", "/") if path.startswith(str(ROOT_DIR)) else path
        excerpts.append((item.__qualname__ + "()", rel, textwrap.dedent(inspect.getsource(item)).rstrip()))
    return excerpts
