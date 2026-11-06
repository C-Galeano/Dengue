"""Database model (SQLAlchemy ORM). The same class works with SQLite and PostgreSQL."""
from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
HORA_COLOMBIA = timezone(timedelta(hours=-5))  # Colombia does not observe daylight saving time


def ahora_colombia():
    return datetime.now(HORA_COLOMBIA).replace(tzinfo=None)


class Paciente(db.Model):
    __tablename__ = "pacientes"

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    # Not unique: the same person can be registered for multiple consultations.
    documento = db.Column(db.String(15), nullable=False, index=True)
    edad = db.Column(db.Integer, nullable=False)
    sexo = db.Column(db.String(1), nullable=False)  # "F" or "M", as in SIVIGILA
    fecha_registro = db.Column(db.DateTime, nullable=False, default=ahora_colombia)

    # Symptoms (booleans): same names as the model variables
    cefalea = db.Column(db.Boolean, nullable=False, default=False)
    dolor_retroocular = db.Column(db.Boolean, nullable=False, default=False)
    mialgias = db.Column(db.Boolean, nullable=False, default=False)
    artralgia = db.Column(db.Boolean, nullable=False, default=False)
    erupcion = db.Column(db.Boolean, nullable=False, default=False)
    dolor_abdominal = db.Column(db.Boolean, nullable=False, default=False)
    vomito = db.Column(db.Boolean, nullable=False, default=False)
    diarrea = db.Column(db.Boolean, nullable=False, default=False)
    somnolencia = db.Column(db.Boolean, nullable=False, default=False)
    sangrado_mucosas = db.Column(db.Boolean, nullable=False, default=False)

    probabilidad = db.Column(db.Float, nullable=False)  # 0 to 1
    resultado = db.Column(db.String(20), nullable=False, index=True)  # "Probable dengue" / "Unlikely"

    def sintomas(self, nombres):
        """List of present symptoms, with their readable names."""
        return [etiqueta for clave, etiqueta in nombres.items() if getattr(self, clave)]

    def __repr__(self):
        return f"<Paciente {self.id} {self.documento} {self.resultado}>"