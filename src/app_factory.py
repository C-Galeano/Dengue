"""Creation and configuration of the Flask application (database, model, templates)."""
import os

from flask import Flask

from src.database import db
from src.prediction import PROBABLE, Predictor
from src.preprocessing import ROOT_DIR, SYMPTOM_LABELS


def database_url():
    """SQLite by default. To migrate to PostgreSQL, simply define the DATABASE_URL variable:
    SQLAlchemy abstracts the database engine, so the model code does not change."""
    url = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'instance' / 'patients.db'}")
    return url.replace("postgres://", "postgresql://", 1)  # Render provides "postgres://" URLs


def create_app():
    app = Flask(__name__, template_folder=str(ROOT_DIR / "app" / "templates"),
                static_folder=str(ROOT_DIR / "app" / "static"),
                instance_path=str(ROOT_DIR / "instance"))
    app.config.update(
        SQLALCHEMY_DATABASE_URI=database_url(),
        SECRET_KEY=os.environ.get("SECRET_KEY", "only-for-local-development"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # Uploaded CSV files: maximum 2 MB
    )
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()  # Creates the tables if they do not exist

    # The model is loaded ONCE at startup (not on every request): loading the .pkl file is slow.
    app.extensions["predictor"] = Predictor()

    @app.context_processor
    def global_variables():
        """Available in all templates."""
        return {"SYMPTOM_LABELS": SYMPTOM_LABELS, "PROBABLE": PROBABLE}

    return app
