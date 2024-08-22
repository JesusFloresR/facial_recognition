from flask import request
from functools import wraps
from src.utils.save_face import save_face
import cv2

def save_faces_per_user (f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        body = request.json
        name = body['name']
        video = cv2.VideoCapture(0)
        path = f'./src/resources/faces/{name}'
        num_img = 1
        print('name: ', name)
        iterator = 0
        print('Extrayendo frames del rostro')
        while True:
            iterator += 1
            check,frame = video.read()

            if iterator % 5 == 0:
                frame_copy = frame.copy()
                xmin_, ymin_, xmax_, ymax_ = save_face(frame_copy, path, name + str(num_img))
                num_img += 1

            if num_img == 101:
                break
        video.release()
        cv2.destroyAllWindows()
        print('Finalizacion de la extraccion del rostro')
        return f(*args, **kwargs)
    return decorated_function