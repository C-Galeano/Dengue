"""Database model (SQLAlchemy ORM). The same class works with SQLite and PostgreSQL."""
from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
COLOMBIA_TZ = timezone(timedelta(hours=-5))  # Colombia does not observe daylight saving time


def now_colombia():
    return datetime.now(COLOMBIA_TZ).replace(tzinfo=None)


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # Not unique: the same person can be registered for multiple consultations.
    document = db.Column(db.String(15), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(1), nullable=False)  # "F" or "M", as in SIVIGILA
    registered_at = db.Column(db.DateTime, nullable=False, default=now_colombia)

    # Symptoms (booleans): same names as the model variables
    headache = db.Column(db.Boolean, nullable=False, default=False)
    retro_orbital_pain = db.Column(db.Boolean, nullable=False, default=False)
    myalgia = db.Column(db.Boolean, nullable=False, default=False)
    arthralgia = db.Column(db.Boolean, nullable=False, default=False)
    rash = db.Column(db.Boolean, nullable=False, default=False)
    abdominal_pain = db.Column(db.Boolean, nullable=False, default=False)
    vomiting = db.Column(db.Boolean, nullable=False, default=False)
    diarrhea = db.Column(db.Boolean, nullable=False, default=False)
    drowsiness = db.Column(db.Boolean, nullable=False, default=False)
    mucosal_bleeding = db.Column(db.Boolean, nullable=False, default=False)

    probability = db.Column(db.Float, nullable=False)  # 0 to 1
    result = db.Column(db.String(20), nullable=False, index=True)  # "Probable dengue" / "Unlikely"

    def symptoms(self, labels):
        """List of present symptoms, with their readable names."""
        return [label for key, label in labels.items() if getattr(self, key)]

    def __repr__(self):
        return f"<Patient {self.id} {self.document} {self.result}>"
