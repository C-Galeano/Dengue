"""Creation and configuration of the Flask application (database, model, templates)."""
import os

from flask import Flask

from src.base_datos import db
from src.prediccion import PROBABLE, Predictor
from src.preprocesamiento import NOMBRES_SINTOMAS, ROOT_DIR


def url_base_datos():
    """SQLite by default. To migrate to PostgreSQL, simply define the DATABASE_URL variable:
    SQLAlchemy abstracts the database engine, so the model code does not change."""
    url = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'instance' / 'pacientes.db'}")
    return url.replace("postgres://", "postgresql://", 1)  # Render provides "postgres://" URLs


def crear_app():
    app = Flask(__name__, template_folder=str(ROOT_DIR / "app" / "templates"),
                static_folder=str(ROOT_DIR / "app" / "static"),
                instance_path=str(ROOT_DIR / "instance"))
    app.config.update(
        SQLALCHEMY_DATABASE_URI=url_base_datos(),
        SECRET_KEY=os.environ.get("SECRET_KEY", "solo-for-local-development"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # Uploaded CSV files: maximum 2 MB
    )
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()  # Creates the tables if they do not exist

    # The model is loaded ONCE at startup (not on every request): loading the .pkl file is slow.
    app.extensions["predictor"] = Predictor()

    @app.context_processor
    def variables_globales():
        """Available in all templates."""
        return {"NOMBRES_SINTOMAS": NOMBRES_SINTOMAS, "PROBABLE": PROBABLE}

    return app