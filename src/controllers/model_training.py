import os
import cv2
import numpy as np
from joblib import dump
from flask import g
from sklearn.preprocessing import LabelEncoder


path = 'D:\\UNMSM\\Ciclo X\\Desarrollo de proyecto de tesis II\\Proyecto\\facial_recognition\\src\\resources\\faces'
people_list = os.listdir(path)
faces = []
labels = []
label = 0

def model_training_controller ():
    print('Generando Codificador...')
    label_encoder = LabelEncoder()
    for people in people_list:
        path_user = f"{path}\\{people}"
        for name_img in os.listdir(path_user):
            img = cv2.imread(f"{path_user}\\{name_img}", 0)
            labels.append(people)
            faces.append(img)
    
    encoded_labels = label_encoder.fit_transform(labels)

    print('Creando el modelo...')
    model = cv2.face.LBPHFaceRecognizer_create()
    print('Entrenando al modelo...')
    model.train(faces, encoded_labels)
    print('Exportando el modelo...')
    model.write('./src/resources/modeloLBPHFace.xml')
    dump(label_encoder, './src/resources/label_encoder.pkl')
    return 'ok'