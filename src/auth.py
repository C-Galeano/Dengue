"""Login de trabajadores y texto editable del inicio."""

import os
from datetime import datetime

import click
from flask_login import LoginManager, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from src.database import db

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Inicia sesión para continuar."


class Worker(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class HomeInfo(db.Model):
    """Texto editable de la página de inicio (una sola fila, id=1)."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, default="")
    body = db.Column(db.Text, nullable=False, default="")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Worker, int(user_id))


def init_auth(app):
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cambia-esto-solo-para-desarrollo")
    login_manager.init_app(app)

    @app.cli.command("create-worker")
    @click.argument("username")
    @click.password_option()
    def create_worker(username, password):
        """Uso: flask create-worker nombre_usuario"""
        db.create_all()
        if db.session.scalar(db.select(Worker).where(Worker.username == username)):
            click.echo("Ese usuario ya existe.")
            return
        worker = Worker(username=username)
        worker.set_password(password)
        db.session.add(worker)
        db.session.commit()
        click.echo(f"Trabajador '{username}' creado.")