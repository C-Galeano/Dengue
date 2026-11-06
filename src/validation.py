"""Server-side validation of the registration form.

Browser validation (HTML5) can be easily bypassed; this validation cannot.
"""
import re

from src.preprocessing import SYMPTOM_LABELS

NAME_RE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñÜü' .-]{3,100}$")  # Accented letters: names are Spanish
DOCUMENT_RE = re.compile(r"^[0-9A-Za-z]{5,15}$")


def validate_patient(form):
    """Returns (clean_data, errors). If errors is empty, the data is valid."""
    errors = {}
    name = " ".join(form.get("name", "").split())
    document = form.get("document", "").strip()
    sex = form.get("sex", "")

    if not NAME_RE.match(name):
        errors["name"] = "Enter a valid name (3 to 100 letters)."
    if not DOCUMENT_RE.match(document):
        errors["document"] = "The document must contain between 5 and 15 letters or numbers, without spaces or periods."
    try:
        age = int(form.get("age", "").strip())
        if not 0 <= age <= 120:
            raise ValueError
    except ValueError:
        age = None
        errors["age"] = "Age must be an integer between 0 and 120."
    if sex not in ("F", "M"):
        errors["sex"] = "Please select a sex."

    symptoms = {s: s in form for s in SYMPTOM_LABELS}
    return {"name": name, "document": document, "age": age, "sex": sex, **symptoms}, errors
