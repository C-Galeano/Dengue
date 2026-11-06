"""Web application routes. The logic is in src/ and the templates are in app/templates/.

Local:       flask run --debug          (http://127.0.0.1:5000)
Production:  gunicorn app:app
"""
from flask import abort, current_app, redirect, render_template, request, url_for

from src.base_datos import Paciente, db
from src.configuracion import crear_app
from src.prediccion import POCO_PROBABLE, PROBABLE
from src.preprocesamiento import NOMBRES_SINTOMAS
from src.pronostico import (SEMANAS_TEST_DEFECTO, documento_proyecto, leer_serie,
                            pronostico_baseline, serie_demo)
from src.validacion import validar_paciente

app = crear_app()


@app.route("/", methods=["GET", "POST"])
def inicio():
    """Home page: project structure + baseline forecast (tabs)."""
    error, datos_propios, serie = None, False, serie_demo()
    if request.method == "POST":
        archivo = request.files.get("archivo")
        try:
            if archivo and archivo.filename:
                serie, datos_propios = leer_serie(archivo=archivo), True
            elif request.form.get("serie"):
                serie, datos_propios = leer_serie(serie_json=request.form["serie"]), True
        except ValueError as exc:
            error = str(exc)
    semanas_test = request.form.get("semanas_test", SEMANAS_TEST_DEFECTO, type=int)
    return render_template("home.html", documento=documento_proyecto(), error=error,
                           pronostico=pronostico_baseline(serie, semanas_test),
                           datos_propios=datos_propios, pestana_pronostico=request.method == "POST")


@app.route("/registro", methods=["GET", "POST"])
def registro():
    """Patient registration and prediction form."""
    if request.method == "GET":
        return render_template("register.html", valores={}, errores={})

    datos, errores = validar_paciente(request.form)
    if errores:
        return render_template("register.html", valores=request.form, errores=errores), 400

    sintomas = {s: datos[s] for s in NOMBRES_SINTOMAS}
    proba, resultado = current_app.extensions["predictor"].predecir(datos["edad"], datos["sexo"], sintomas)
    paciente = Paciente(**datos, probabilidad=proba, resultado=resultado)
    db.session.add(paciente)
    db.session.commit()
    # Post/Redirect/Get: reloading the results page does not register the patient again.
    return redirect(url_for("resultado", paciente_id=paciente.id))


@app.route("/resultado/<int:paciente_id>")
def resultado(paciente_id):
    paciente = db.session.get(Paciente, paciente_id) or abort(404)
    return render_template("result.html", p=paciente, umbral=current_app.extensions["predictor"].umbral)


@app.route("/pacientes")
def pacientes():
    """Patient list filtered by result: ?resultado=probable | poco | todos."""
    filtros = {"probable": PROBABLE, "poco": POCO_PROBABLE}
    filtro = request.args.get("resultado", "todos")
    consulta = db.select(Paciente).order_by(Paciente.fecha_registro.desc())
    if filtro in filtros:
        consulta = consulta.where(Paciente.resultado == filtros[filtro])
    else:
        filtro = "todos"
    return render_template("patients.html", pacientes=db.session.scalars(consulta).all(), filtro=filtro)


@app.route("/acerca")
def acerca():
    return render_template("about.html", info=current_app.extensions["predictor"].info)