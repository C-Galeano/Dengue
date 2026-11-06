"""Smoke tests for the Flask web app (routes, form validation, patient registration)."""
import io
import os

import pytest

from src.preprocessing import age_to_group


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The routes live in app.py, so the module is imported with a throw-away SQLite database."""
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_age_to_group():
    assert [age_to_group(a) for a in (0, 5, 6, 17, 28, 59, 60)] == ["0-5", "0-5", "6-11", "12-17", "18-28", "29-59", "60+"]


def test_pages_load(client):
    for url in ("/", "/register", "/patients", "/about"):
        assert client.get(url).status_code == 200


def test_register_patient_and_see_result(client):
    form = {"name": "Maria Perez", "document": "123456789", "age": "25", "sex": "F", "headache": "1", "rash": "1"}
    response = client.post("/register", data=form)
    assert response.status_code == 302 and "/result/" in response.headers["Location"]
    page = client.get(response.headers["Location"])
    assert page.status_code == 200 and b"Maria Perez" in page.data
    assert b"Maria Perez" in client.get("/patients").data


def test_invalid_form_is_rejected(client):
    response = client.post("/register", data={"name": "x", "document": "1", "age": "-1"})
    assert response.status_code == 400 and b"is-invalid" in response.data


def test_forecast_accepts_csv_with_english_or_spanish_columns(client):
    for header in ("week,cases", "semana,casos"):
        body = header + "\n" + "\n".join(f"{i},{20 + i % 7}" for i in range(1, 61))
        response = client.post("/", data={"file": (io.BytesIO(body.encode()), "series.csv")},
                               content_type="multipart/form-data")
        assert response.status_code == 200 and b"Showing synthetic demo data" not in response.data
