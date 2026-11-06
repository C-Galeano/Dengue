"""Validación en el servidor del formulario de registro.

La validación del navegador (HTML5) se puede saltar fácilmente; esta no.
"""
import re

from src.preprocesamiento import NOMBRES_SINTOMAS

RE_NOMBRE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñÜü' .-]{3,100}$")
RE_DOCUMENTO = re.compile(r"^[0-9A-Za-z]{5,15}$")


def validar_paciente(form):
    """Devuelve (datos_limpios, errores). Si errores está vacío, los datos son válidos."""
    errores = {}
    nombre = " ".join(form.get("nombre", "").split())
    documento = form.get("documento", "").strip()
    sexo = form.get("sexo", "")

    if not RE_NOMBRE.match(nombre):
        errores["nombre"] = "Ingrese un nombre válido (3 a 100 letras)."
    if not RE_DOCUMENTO.match(documento):
        errores["documento"] = "El documento debe tener entre 5 y 15 letras o números, sin espacios ni puntos."
    try:
        edad = int(form.get("edad", "").strip())
        if not 0 <= edad <= 120:
            raise ValueError
    except ValueError:
        edad = None
        errores["edad"] = "La edad debe ser un número entero entre 0 y 120."
    if sexo not in ("F", "M"):
        errores["sexo"] = "Seleccione el sexo."

    sintomas = {s: s in form for s in NOMBRES_SINTOMAS}
    return {"nombre": nombre, "documento": documento, "edad": edad, "sexo": sexo, **sintomas}, errores
