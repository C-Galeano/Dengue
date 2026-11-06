"""Server-side validation of the registration form.

Browser validation (HTML5) can be easily bypassed; this validation cannot.
"""
import re

from src.preprocesamiento import NOMBRES_SINTOMAS

RE_NOMBRE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñÜü' .-]{3,100}$")
RE_DOCUMENTO = re.compile(r"^[0-9A-Za-z]{5,15}$")


def validar_paciente(form):
    """Returns (clean_data, errors). If errors is empty, the data is valid."""
    errores = {}
    nombre = " ".join(form.get("nombre", "").split())
    documento = form.get("documento", "").strip()
    sexo = form.get("sexo", "")

    if not RE_NOMBRE.match(nombre):
        errores["nombre"] = "Enter a valid name (3 to 100 letters)."
    if not RE_DOCUMENTO.match(documento):
        errores["documento"] = "The document must contain between 5 and 15 letters or numbers, without spaces or periods."
    try:
        edad = int(form.get("edad", "").strip())
        if not 0 <= edad <= 120:
            raise ValueError
    except ValueError:
        edad = None
        errores["edad"] = "Age must be an integer between 0 and 120."
    if sexo not in ("F", "M"):
        errores["sexo"] = "Please select a sex."

    sintomas = {s: s in form for s in NOMBRES_SINTOMAS}
    return {"nombre": nombre, "documento": documento, "edad": edad, "sexo": sexo, **sintomas}, errores