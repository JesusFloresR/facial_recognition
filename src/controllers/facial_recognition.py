import cv2, time
import numpy as np
import face_detection

def facial_recognition():
    video = cv2.VideoCapture(0)
    address = "rtsp://192.168.18.120:8080/h264_ulaw.sdp"
    video.open(address)
    detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)
    # print(video.isOpened())
    
    iterator = 0
    xmin_ = None
    ymin_ = None
    xmax_ = None
    ymax_ = None

    while True:
        iterator += 1
        check,frame = video.read()
        # print(detections)

        if xmin_ is not None and ymin_ is not None and xmax_ is not None and ymax_ is not None:
            cv2.rectangle(frame, (int(xmin_),int(ymin_)), (int(xmax_), int(ymax_)), (0,255,0), 3)

        if iterator % 20 == 0:
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


