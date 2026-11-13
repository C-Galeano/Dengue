"""CRISP-DM methodology pages: phase metadata and the real code excerpts shown on each page.

The explanatory text of each phase lives in app/templates/phases/<number>.html.
Code excerpts are read from the source with `inspect`, so the pages always show the code that actually runs.
"""
import inspect
import textwrap
from functools import lru_cache

from src import dbc, explain, prediction, preprocessing, sinan, train, validation
from src.preprocessing import ROOT_DIR

PHASES = [
    {"number": 1, "name": "Business Understanding", "icon": "target",
     "question": "What problem do we solve, and how do we measure success?",
     "summary": "Problem, objectives, success criteria and project plan.",
     "notebook": "01_business_understanding.ipynb"},
    {"number": 2, "name": "Data Understanding", "icon": "search",
     "question": "What do the data contain, and how reliable are they?",
     "summary": "Public arbovirus sources, SINAN records, label quality and clinical signal by disease.",
     "notebook": "02_data_understanding.ipynb"},
    {"number": 3, "name": "Data Preparation", "icon": "filter",
     "question": "How do we turn raw records into a modeling table?",
     "summary": "Laboratory-confirmed labels, clinical features plus epidemiological context, balanced samples.",
     "notebook": "03_data_preparation.ipynb"},
    {"number": 4, "name": "Modeling", "icon": "cpu",
     "question": "Which algorithm, and which decision rule?",
     "summary": "Random Forest, XGBoost and LightGBM vs. a linear baseline, and the dengue alert.",
     "notebook": "04_modeling.ipynb"},
    {"number": 5, "name": "Evaluation", "icon": "chart",
     "question": "Does the model meet the clinical success criteria?",
     "summary": "Test metrics, SHAP explainability, temporal and Colombian external validation.",
     "notebook": "05_evaluation.ipynb"},
    {"number": 6, "name": "Deployment", "icon": "server",
     "question": "How do clinicians get access to the model?",
     "summary": "Smart-hospital web app with per-patient SHAP explanations, database and hosting.",
     "notebook": "06_deployment.ipynb"},
]

CODE = {
    2: [dbc.read_dbc, sinan.summarize],
    3: [sinan.build_year, sinan.to_features, preprocessing.age_in_years],
    4: [train.build_models, train.choose_best, train.decide, train.choose_dengue_threshold],
    5: [train.compute_metrics, explain.shap_values, train.external_validation],
    6: [prediction.Predictor.predict, explain.explain_patient, validation.validate_patient],
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
