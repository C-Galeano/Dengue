"""Modelo de base de datos (ORM SQLAlchemy). La misma clase funciona en SQLite y PostgreSQL."""
from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
HORA_COLOMBIA = timezone(timedelta(hours=-5))  # Colombia no tiene horario de verano


def ahora_colombia():
    return datetime.now(HORA_COLOMBIA).replace(tzinfo=None)


class Paciente(db.Model):
    __tablename__ = "pacientes"

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    # No es único: una misma persona puede registrarse en varias consultas.
    documento = db.Column(db.String(15), nullable=False, index=True)
    edad = db.Column(db.Integer, nullable=False)
    sexo = db.Column(db.String(1), nullable=False)  # "F" o "M", como en SIVIGILA
    fecha_registro = db.Column(db.DateTime, nullable=False, default=ahora_colombia)

    # Síntomas (booleanos): mismos nombres que las variables del modelo
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

    probabilidad = db.Column(db.Float, nullable=False)  # 0 a 1
    resultado = db.Column(db.String(20), nullable=False, index=True)  # "Probable dengue" / "Poco probable"

    def sintomas(self, nombres):
        """Lista de síntomas presentes, con su nombre legible."""
        return [etiqueta for clave, etiqueta in nombres.items() if getattr(self, clave)]

    def __repr__(self):
        return f"<Paciente {self.id} {self.documento} {self.resultado}>"
