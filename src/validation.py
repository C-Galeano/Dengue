"""Server-side validation of the consultation form.

Browser validation (HTML5) can be easily bypassed; this validation cannot.
"""
import re

from src.preprocessing import BINARY, MAX_DAYS_SINCE_ONSET, STATES

NAME_RE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñÜü' .-]{3,100}$")  # Accented letters: names are Spanish
DOCUMENT_RE = re.compile(r"^[0-9A-Za-z]{5,15}$")


def _integer(form, field, low, high):
    try:
        value = int(form.get(field, "").strip())
        return value if low <= value <= high else None
    except ValueError:
        return None


def validate_patient(form):
    """Returns (clean_data, errors). If errors is empty, the data is valid."""
    errors = {}
    name = " ".join(form.get("name", "").split())
    document = form.get("document", "").strip()
    sex = form.get("sex", "")
    age = _integer(form, "age", 0, 110)
    days = _integer(form, "days_since_onset", 0, MAX_DAYS_SINCE_ONSET)

    if not NAME_RE.match(name):
        errors["name"] = "Enter a valid name (3 to 100 letters)."
    if not DOCUMENT_RE.match(document):
        errors["document"] = "The document must contain between 5 and 15 letters or numbers, without spaces or periods."
    if age is None:
        errors["age"] = "Age must be an integer between 0 and 110."
    if sex not in ("F", "M"):
        errors["sex"] = "Please select a sex."
    if days is None:
        errors["days_since_onset"] = f"Enter the days since the symptoms started (0 to {MAX_DAYS_SINCE_ONSET})."
    state = form.get("state", "")
    if state not in STATES:
        errors["state"] = "Select the state of residence."
    month = _integer(form, "month", 1, 12)
    if month is None:
        errors["month"] = "Select the month of the consultation."

    findings = {key: key in form for key in BINARY}
    # The epidemiological context is stored with the findings (JSON), so the table schema does not change.
    findings.update({"state": state, "month": month})
    return {"name": name, "document": document, "age": age, "sex": sex,
            "days_since_onset": days, "findings": findings}, errors
