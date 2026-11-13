"""Web application routes. The logic is in src/ and the templates are in app/templates/.

Local:       flask run --debug          (http://127.0.0.1:5000)
Production:  gunicorn app:app
"""
from flask import abort, current_app, redirect, render_template, request, url_for

from src.app_factory import create_app
from src.database import Patient, db
from src.methodology import PHASES, code_excerpts, get_phase
from src.forecast import (TEST_WEEKS_DEFAULT, baseline_forecast, demo_series,
                          project_document, read_series)
from src.preprocessing import CLASSES
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
    """Consultation form: patient data and clinical findings -> differential diagnosis."""
    if request.method == "GET":
        return render_template("register.html", values={}, errors={})

    data, errors = validate_patient(request.form)
    if errors:
        return render_template("register.html", values=request.form, errors=errors), 400

    findings = data["findings"]
    output = current_app.extensions["predictor"].predict(
        data["age"], data["sex"], data["days_since_onset"], findings, findings["state"], findings["month"])
    patient = Patient(**data, **output)
    db.session.add(patient)
    db.session.commit()
    # Post/Redirect/Get: reloading the results page does not register the patient again.
    return redirect(url_for("result", patient_id=patient.id))


@app.route("/result/<int:patient_id>")
def result(patient_id):
    patient = db.session.get(Patient, patient_id) or abort(404)
    return render_template("result.html", p=patient,
                           threshold=current_app.extensions["predictor"].dengue_threshold)


@app.route("/patients")
def patients():
    """Consultation list filtered by suggested diagnosis: ?result=dengue | chikungunya | other_febrile | all."""
    current_filter = request.args.get("result", "all")
    query = db.select(Patient).order_by(Patient.registered_at.desc())
    if current_filter in CLASSES:
        query = query.where(Patient.prediction == current_filter)
    else:
        current_filter = "all"
    return render_template("patients.html", patients=db.session.scalars(query).all(), current_filter=current_filter)


@app.route("/about")
def about():
    return render_template("about.html", info=current_app.extensions["predictor"].info)


@app.route("/methodology")
def methodology():
    """Overview of the six CRISP-DM phases."""
    return render_template("methodology.html", phases=PHASES)


@app.route("/methodology/<int:number>")
def phase(number):
    """One page per CRISP-DM phase: what the phase is and how this project implements it."""
    current = get_phase(number) or abort(404)
    return render_template(f"phases/{number}.html", phase=current, phases=PHASES,
                           code=code_excerpts(number), info=current_app.extensions["predictor"].info)
