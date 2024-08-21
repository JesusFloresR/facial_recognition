import cv2, time
import numpy as np
import face_detection
import imutils

def get_label(prediction, label_encoder, umbral_confianza):
    etiqueta_predicha = None

    if np.max(prediction)>umbral_confianza:
        clase_predicha = np.argmax(prediction)
        print("clase_predicha: ", clase_predicha)
        etiqueta_predicha = label_encoder.inverse_transform([prediction[0]])[0]
    else: 
        etiqueta_predicha = "desconocido"
    
    return etiqueta_predicha

def facial_recognition(model, label_encoder, umbral=0.99):
    # face_label = []
    i=0
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,frame = cap.read()
        _frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
        prediction = model.predict(gray)
        print("Prediccion: ", prediction)
        label = get_label(prediction, label_encoder, umbral)
        # face_label.append(label)

        print(label)
        # results[i]["name"] = label
        i += 1
    return results

def _facial_recognition():
    video = cv2.VideoCapture(0)
    # address = "rtsp://192.168.100.168:8080/h264_ulaw.sdp"
    # video.open(address)
    detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)

    iterator = 0
    xmin_ = None
    ymin_ = None
    xmax_ = None
    ymax_ = None

    while True:
        iterator += 1
        check,frame = video.read()

        if xmin_ is not None and ymin_ is not None and xmax_ is not None and ymax_ is not None:
            cv2.rectangle(frame, (int(xmin_),int(ymin_)), (int(xmax_), int(ymax_)), (0,255,0), 3)

        if iterator % 15 == 0:
            xmin_ = None
            ymin_ = None
            xmax_ = None
            ymax_ = None
            detections = detector.detect(frame)

            for xmin,ymin,xmax,ymax,precision in detections:
                # print(precision)
                xmin_, ymin_, xmax_, ymax_ = xmin, ymin, xmax, ymax
                cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0), 3)
        cv2.imshow("Real Time", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


