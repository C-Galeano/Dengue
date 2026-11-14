"""Web application routes. The logic is in src/ and the templates are in app/templates/.

Local: flask run --debug (http://127.0.0.1:5000)

Production: gunicorn app:app
"""

import hmac
import os
from functools import wraps

from flask import (abort, current_app, redirect, render_template, request,
                   session, url_for)

from src.app_factory import create_app
from src.database import Patient, db
from src.methodology import PHASES, code_excerpts, get_phase
from src.monitoring import load_report
from src.forecast import (TEST_WEEKS_DEFAULT, baseline_forecast, demo_series,
                          project_document, read_series)
from src.preprocessing import CLASSES
from src.validation import validate_patient

app = create_app()

# --- Simple login (demo level) ---------------------------------------------
# Change these with environment variables before deploying (e.g. in Render).
app.config["SECRET_KEY"] = app.config.get("SECRET_KEY") or os.environ.get("SECRET_KEY", "dev-change-me")
AUTH_USER = os.environ.get("APP_USER", "admin")
AUTH_PASSWORD = os.environ.get("APP_PASSWORD", "dengue2026")


def login_required(view):
    """Redirect to /login when nobody is signed in."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


def safe_next(value):
    """Only allow redirects to paths inside this site."""
    return value if value.startswith("/") and not value.startswith("//") else ""


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = safe_next(request.values.get("next", ""))
    if request.method == "GET":
        return render_template("login.html", error=None, next_url=next_url, username="")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user_ok = hmac.compare_digest(username.encode(), AUTH_USER.encode())
    pass_ok = hmac.compare_digest(password.encode(), AUTH_PASSWORD.encode())
    if user_ok and pass_ok:
        session.clear()
        session["user"] = username
        return redirect(next_url or url_for("home"))
    return render_template("login.html", error="Incorrect username or password.",
                           next_url=next_url, username=username), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


# --- Pages -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    """Home page: model snapshot + baseline forecast (tabs)."""
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
                           own_data=own_data, forecast_tab=request.method == "POST",
                           info=current_app.extensions["predictor"].info)


@app.route("/register", methods=["GET", "POST"])
@login_required
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
@login_required
def result(patient_id):
    patient = db.session.get(Patient, patient_id) or abort(404)
    return render_template("result.html", p=patient,
                           threshold=current_app.extensions["predictor"].dengue_threshold)
@app.route("/patient/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    """Update a patient's clinical history (new symptoms, findings, etc.) and re-run the model."""
    patient = db.session.get(Patient, patient_id) or abort(404)

    if request.method == "GET":
        # Findings are stored with the same keys the form uses, so they pre-fill the checkboxes, state and month.
        values = {
            "name": patient.name, "document": patient.document, "age": patient.age,
            "sex": patient.sex, "days_since_onset": patient.days_since_onset,
            **(patient.findings or {}),
        }
        return render_template("register.html", values=values, errors={}, editing=True, patient=patient)

    data, errors = validate_patient(request.form)
    if errors:
        return render_template("register.html", values=request.form, errors=errors,
                               editing=True, patient=patient), 400

    findings = data["findings"]
    output = current_app.extensions["predictor"].predict(
        data["age"], data["sex"], data["days_since_onset"], findings, findings["state"], findings["month"])

    # Overwrite the stored data and the new prediction
    for field, value in {**data, **output}.items():
        setattr(patient, field, value)
    db.session.commit()
    return redirect(url_for("result", patient_id=patient.id))

@app.route("/patients")
@login_required
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
    """Overview of the six CRISP-ML(Q) phases."""
    return render_template("methodology.html", phases=PHASES)


@app.route("/methodology/<int:number>")
def phase(number):
    """One page per CRISP-ML(Q) phase: what the phase is and how this project implements it."""
    current = get_phase(number) or abort(404)
    return render_template(f"phases/{number}.html", phase=current, phases=PHASES,
                           code=code_excerpts(number), info=current_app.extensions["predictor"].info,
                           monitoring=load_report() if number == 6 else None)