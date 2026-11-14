"""Smoke tests for the Flask web app (routes, form validation, patient registration)."""
import io
import os

import pytest

import pandas as pd

from src.preprocessing import CLASSES, FEATURES, age_in_years, patient_frame, to_binary


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The routes live in app.py, so the module is imported with a throw-away SQLite database."""
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    # Consultation pages require login (demo credentials, overridable with APP_USER / APP_PASSWORD).
    response = client.post("/login", data={"username": app_module.AUTH_USER, "password": app_module.AUTH_PASSWORD})
    assert response.status_code == 302
    return client


def test_consultation_pages_require_login(client):
    anonymous = client.application.test_client()
    response = anonymous.get("/register")
    assert response.status_code == 302 and "/login" in response.headers["Location"]
    assert anonymous.post("/login", data={"username": "x", "password": "y"}).status_code == 401


def test_sinan_decoding():
    """SINAN ages: first digit = unit (4 years, 3 months, 2 days, 1 hours); yes/no fields: 1 = yes, 2 = no."""
    ages = age_in_years(pd.Series(["4035", "3006", "2073", "1005", "4999", ""])).tolist()
    assert ages[:4] == [35, 0.5, 0.2, 0] and all(pd.isna(a) for a in ages[4:])
    assert to_binary(pd.Series(["1", "2", "9", ""])).fillna(-1).tolist() == [1, 0, -1, -1]


def test_patient_frame_has_model_columns():
    X = patient_frame(30, "F", 3, {"fever": True, "arthralgia": True}, "31", 4)
    assert list(X.columns) == FEATURES and X.loc[0, "arthralgia"] == 1 and X.loc[0, "rash"] == 0
    assert X.loc[0, "state"] == 31 and X.loc[0, "month"] == 4


def test_predictor_returns_three_probabilities_and_explanation(client):
    predictor = client.application.extensions["predictor"]
    out = predictor.predict(40, "M", 2, {"fever": True, "arthralgia": True, "arthritis": True}, "31", 3)
    assert set(out["probabilities"]) == set(CLASSES)
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6
    assert out["prediction"] in CLASSES and len(out["explanation"]) > 0


def test_pages_load(client):
    for url in ("/", "/register", "/patients", "/about"):
        assert client.get(url).status_code == 200


def test_methodology_pages(client):
    """The overview and one page per CRISP-ML(Q) phase, each with its explanation and real code."""
    overview = client.get("/methodology")
    assert overview.status_code == 200 and b"CRISP-ML(Q)" in overview.data
    for number, name in enumerate(["Business and Data Understanding", "Data Engineering",
                                   "Machine Learning Model Engineering", "Quality Assurance for ML Applications",
                                   "Deployment", "Monitoring and Maintenance"], start=1):
        page = client.get(f"/methodology/{number}")
        assert page.status_code == 200
        assert name.encode() in page.data and b"What we did in this project" in page.data
        assert b"CRISP-DM" not in page.data
    assert b"def build_models" in client.get("/methodology/3").data
    monitoring = client.get("/methodology/6").data
    assert b"def psi" in monitoring and b"Population Stability Index" in monitoring
    assert client.get("/methodology/7").status_code == 404


def test_register_patient_and_see_result(client):
    form = {"name": "Maria Perez", "document": "123456789", "age": "25", "sex": "F", "days_since_onset": "3",
            "state": "35", "month": "4", "fever": "1", "headache": "1", "rash": "1", "leukopenia": "1"}
    response = client.post("/register", data=form)
    assert response.status_code == 302 and "/result/" in response.headers["Location"]
    page = client.get(response.headers["Location"])
    assert page.status_code == 200 and b"Maria Perez" in page.data and b"Why this suggestion?" in page.data
    assert b"Maria Perez" in client.get("/patients").data
    for diagnosis in CLASSES:
        assert client.get(f"/patients?result={diagnosis}").status_code == 200


def test_invalid_form_is_rejected(client):
    response = client.post("/register", data={"name": "x", "document": "1", "age": "-1", "days_since_onset": "99",
                                              "state": "99", "month": "13"})
    assert response.status_code == 400 and b"is-invalid" in response.data
    assert b"Select the state of residence." in response.data


def test_forecast_accepts_csv_with_english_or_spanish_columns(client):
    for header in ("week,cases", "semana,casos"):
        body = header + "\n" + "\n".join(f"{i},{20 + i % 7}" for i in range(1, 61))
        response = client.post("/", data={"file": (io.BytesIO(body.encode()), "series.csv")},
                               content_type="multipart/form-data")
        assert response.status_code == 200 and b"Showing synthetic demo data" not in response.data


def test_psi_detects_drift():
    from src.monitoring import psi, psi_status
    same = pd.Series([0, 1] * 500)
    assert psi(same, same) < 0.01 and psi_status(psi(same, same)) == "stable"
    shifted = pd.Series([1] * 900 + [0] * 100)
    assert psi_status(psi(same, shifted)) == "significant"
    ages = pd.Series(range(1000))
    assert psi(ages, ages, numeric=True) < 0.01 and psi(ages, ages + 500, numeric=True) > 0.25
