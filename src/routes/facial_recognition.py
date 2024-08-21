from flask import Blueprint
import cv2
from src.controllers.facial_recognition import facial_recognition
# from src.controllers.save_faces_per_user import save_faces_per_user
from src.controllers.model_training import model_training_controller
from src.middlewares.save_faces_per_user import save_faces_per_user
from joblib import load

main = Blueprint('conclusions_blueprint',__name__)

@main.route('/facial-recognition')
def index():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Cargando modelo...")
    face_recognizer.read('./src/resources/modeloLBPHFace.xml')
    print("Modelo cargado")

    print("Cargando Label Encoder...")
    label_encoder = load('./src/resources/label_encoder.pkl')
    print("Label Encoder cargado")
    facial_recognition(face_recognizer, label_encoder)

# @main.route('/save-face')
# def save_face():
#     save_faces_per_user()

@main.route('/model-training', methods = ['POST'])
@save_faces_per_user
def model_training ():
    model_training_controller()