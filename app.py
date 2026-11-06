"""Web application routes. The logic is in src/ and the templates are in app/templates/.

Local:       flask run --debug          (http://127.0.0.1:5000)
Production:  gunicorn app:app
"""
from flask import abort, current_app, redirect, render_template, request, url_for

from src.app_factory import create_app
from src.database import Patient, db
from src.forecast import (TEST_WEEKS_DEFAULT, baseline_forecast, demo_series,
                          project_document, read_series)
from src.prediction import PROBABLE, UNLIKELY
from src.preprocessing import SYMPTOM_LABELS
from src.validation import validate_patient

app = create_app()


@app.route("/", methods=["GET", "POST"])
def home():
    """Home page: project structure + baseline forecast (tabs)."""
    error, own_data, series = None, False, demo_series()
    if request.method == "POST":
        file = request.files.get("file")
        try:
            if file and file.filename:
                series, own_data = read_series(file=file), True
            elif request.form.get("series"):
                series, own_data = read_series(series_json=request.form["series"]), True
        except ValueError as exc:
            error = str(exc)
    test_weeks = request.form.get("test_weeks", TEST_WEEKS_DEFAULT, type=int)
    return render_template("home.html", document=project_document(), error=error,
                           forecast=baseline_forecast(series, test_weeks),
                           own_data=own_data, forecast_tab=request.method == "POST")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Patient registration and prediction form."""
    if request.method == "GET":
        return render_template("register.html", values={}, errors={})

    data, errors = validate_patient(request.form)
    if errors:
        return render_template("register.html", values=request.form, errors=errors), 400

    symptoms = {s: data[s] for s in SYMPTOM_LABELS}
    proba, result = current_app.extensions["predictor"].predict(data["age"], data["sex"], symptoms)
    patient = Patient(**data, probability=proba, result=result)
    db.session.add(patient)
    db.session.commit()
    # Post/Redirect/Get: reloading the results page does not register the patient again.
    return redirect(url_for("result", patient_id=patient.id))


@app.route("/result/<int:patient_id>")
def result(patient_id):
    patient = db.session.get(Patient, patient_id) or abort(404)
    return render_template("result.html", p=patient, threshold=current_app.extensions["predictor"].threshold)


@app.route("/patients")
def patients():
    """Patient list filtered by result: ?result=probable | unlikely | all."""
    filters = {"probable": PROBABLE, "unlikely": UNLIKELY}
    current_filter = request.args.get("result", "all")
    query = db.select(Patient).order_by(Patient.registered_at.desc())
    if current_filter in filters:
        query = query.where(Patient.result == filters[current_filter])
    else:
        current_filter = "all"
    return render_template("patients.html", patients=db.session.scalars(query).all(), current_filter=current_filter)


@app.route("/about")
def about():
    return render_template("about.html", info=current_app.extensions["predictor"].info)
