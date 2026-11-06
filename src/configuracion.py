"""Creación y configuración de la aplicación Flask (base de datos, modelo, plantillas)."""
import os

from flask import Flask

from src.base_datos import db
from src.prediccion import PROBABLE, Predictor
from src.preprocesamiento import NOMBRES_SINTOMAS, ROOT_DIR


def url_base_datos():
    """SQLite por defecto. Para migrar a PostgreSQL basta con definir la variable DATABASE_URL:
    SQLAlchemy abstrae el motor, así que el código de los modelos no cambia."""
    url = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'instance' / 'pacientes.db'}")
    return url.replace("postgres://", "postgresql://", 1)  # Render entrega URLs "postgres://"


def crear_app():
    app = Flask(__name__, template_folder=str(ROOT_DIR / "app" / "templates"),
                static_folder=str(ROOT_DIR / "app" / "static"),
                instance_path=str(ROOT_DIR / "instance"))
    app.config.update(
        SQLALCHEMY_DATABASE_URI=url_base_datos(),
        SECRET_KEY=os.environ.get("SECRET_KEY", "solo-para-desarrollo-local"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # CSV subidos: máximo 2 MB
    )
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()  # Crea las tablas si no existen

    # El modelo se carga UNA vez al iniciar (no en cada petición): cargar el .pkl es lento.
    app.extensions["predictor"] = Predictor()

    @app.context_processor
    def variables_globales():
        """Disponibles en todas las plantillas."""
        return {"NOMBRES_SINTOMAS": NOMBRES_SINTOMAS, "PROBABLE": PROBABLE}

    return app
