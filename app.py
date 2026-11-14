"""Web application routes. The logic is in src/ and the templates are in app/templates/.

Local: flask run --debug (http://127.0.0.1:5000)

Production: gunicorn app:app
"""

from urllib.parse import urlsplit

from flask import abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from src.app_factory import create_app
from src.auth import HomeInfo, Worker, init_auth
from src.database import Patient, db
from src.methodology import PHASES, code_excerpts, get_phase
from src.forecast import (TEST_WEEKS_DEFAULT, baseline_forecast, demo_series,
                          project_document, read_series)
from src.preprocessing import CLASSES
from src.validation import validate_patient

app = create_app()
init_auth(app)
with app.app_context():
    db.create_all()  # crea las tablas nuevas (Worker, HomeInfo) si no existen


# ----------------------------------------------------------------------------
# Autenticación
# ----------------------------------------------------------------------------
def _is_safe_next(target):
    """Solo se permite redirigir a rutas internas del sitio."""
    return bool(target) and target.startswith("/") and not target.startswith("//") \
        and not urlsplit(target).netloc


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        worker = db.session.scalar(db.select(Worker).where(Worker.username == username))
        if worker and worker.check_password(request.form.get("password", "")):
            login_user(worker)
            nxt = request.args.get("next", "")
            return redirect(nxt if _is_safe_next(nxt) else url_for("home"))
        error = "Usuario o contraseña incorrectos."
    return render_template("login.html", error=error), (401 if error else 200)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# ----------------------------------------------------------------------------
# Inicio
# ----------------------------------------------------------------------------
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
                           own_data=own_data, forecast_tab=request.method == "POST",
                           home_info=db.session.get(HomeInfo, 1))


@app.route("/home/edit", methods=["GET", "POST"])
@login_required
def edit_home():
    """El trabajador edita el título y el texto informativo del inicio."""
    info = db.session.get(HomeInfo, 1)
    if request.method == "POST":
        if info is None:
            info = HomeInfo(id=1)
            db.session.add(info)
        info.title = request.form.get("title", "").strip()
        info.body = request.form.get("body", "").strip()
        db.session.commit()
        flash("Información del inicio actualizada.")
        return redirect(url_for("home"))
    return render_template("edit_home.html", info=info)


# ----------------------------------------------------------------------------
# Pacientes (solo trabajadores con sesión iniciada)
# ----------------------------------------------------------------------------
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


@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    """Actualiza los datos/hallazgos del paciente y recalcula el diagnóstico sugerido."""
    patient = db.session.get(Patient, patient_id) or abort(404)

    if request.method == "GET":
        values = {"age": patient.age, "sex": patient.sex,
                  "days_since_onset": patient.days_since_onset,
                  **(patient.findings or {})}
        return render_template("register.html", values=values, errors={}, editing=True)

    data, errors = validate_patient(request.form)
    if errors:
        return render_template("register.html", values=request.form, errors=errors, editing=True), 400

    findings = data["findings"]
    output = current_app.extensions["predictor"].predict(
        data["age"], data["sex"], data["days_since_onset"], findings, findings["state"], findings["month"])
    for key, value in {**data, **output}.items():
        setattr(patient, key, value)
    db.session.commit()
    flash("Paciente actualizado.")
    return redirect(url_for("result", patient_id=patient.id))


@app.route("/result/<int:patient_id>")
@login_required
def result(patient_id):
    patient = db.session.get(Patient, patient_id) or abort(404)
    return render_template("result.html", p=patient,
                           threshold=current_app.extensions["predictor"].dengue_threshold)


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


# ----------------------------------------------------------------------------
# Páginas públicas
# ----------------------------------------------------------------------------
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