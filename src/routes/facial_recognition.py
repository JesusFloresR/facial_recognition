from flask import Blueprint
from src.controllers.facial_recognition import facial_recognition

main = Blueprint('conclusions_blueprint',__name__)

@main.route('/facial_recognition')
def index():
    facial_recognition()