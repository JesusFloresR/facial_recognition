from flask import Flask
from .routes import facial_recognition

app = Flask(__name__)

def init_app():
    app.register_blueprint(facial_recognition.main, url_prefix = '/api')
    # app.register_blueprint(datasets.main, url_prefix = '/api')

    return app