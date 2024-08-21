from src.utils.extract_face import extract_face
import cv2
import os

def save_face(img, path, name):
    face, xmin, ymin, xmax, ymax = extract_face(img)
    if not os.path.exists(path ):
        os.makedirs(path)
    cv2.imwrite(path + '/' + name + '.jpg',face)
    return xmin, ymin, xmax, ymax