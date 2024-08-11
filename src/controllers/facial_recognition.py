import cv2, time
import numpy as np
import face_detection

def facial_recognition():
    video = cv2.VideoCapture(0)
    address = "http://192.168.100.168:8080/video"
    video.open(address)
    detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)
    # print(video.isOpened())
    
    while True:
        check,frame = video.read()
        detections = detector.detect(frame)
        # print(detections)
        for xmin,ymin,xmax,ymax,precision in detections:
            # print(precision)
            img = cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0), 3)
        cv2.imshow("Real Time", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


