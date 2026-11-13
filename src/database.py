"""Database model (SQLAlchemy ORM). The same class works with SQLite and PostgreSQL."""
from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
COLOMBIA_TZ = timezone(timedelta(hours=-5))  # Colombia does not observe daylight saving time


def now_colombia():
    return datetime.now(COLOMBIA_TZ).replace(tzinfo=None)


class Patient(db.Model):
    """One consultation: patient data, clinical findings and the model's differential diagnosis.

    Stored in the "consultations" table (the earlier dengue-only version used "patients").
    """
    __tablename__ = "consultations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # Not unique: the same person can be registered for multiple consultations.
    document = db.Column(db.String(15), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(1), nullable=False)  # "F" or "M"
    days_since_onset = db.Column(db.Integer, nullable=False)
    registered_at = db.Column(db.DateTime, nullable=False, default=now_colombia)

    # {feature: bool} for symptoms, clinical/laboratory findings and comorbidities (src/preprocessing.py)
    findings = db.Column(db.JSON, nullable=False, default=dict)

    # Model output
    probabilities = db.Column(db.JSON, nullable=False)                 # {class: probability}
    prediction = db.Column(db.String(20), nullable=False, index=True)  # dengue / chikungunya / other_febrile
    dengue_alert = db.Column(db.Boolean, nullable=False, default=False)
    explanation = db.Column(db.JSON, nullable=False, default=list)     # SHAP contributions

    def present(self, labels):
        """Readable names of the findings marked as present."""
        return [label for key, label in labels.items() if self.findings.get(key)]

    @property
    def probability(self):
        """Probability of the suggested diagnosis."""
        return self.probabilities[self.prediction]

    def __repr__(self):
        return f"<Consultation {self.id} {self.document} {self.prediction}>"
