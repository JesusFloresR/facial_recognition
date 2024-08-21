import face_detection
import imutils
import cv2

detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)
def extract_face(img):
    frame = imutils.resize(img, width=640)
    auxFrame = frame.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img)
    detections = detector.detect(frame)

    xmin_ = None
    ymin_ = None
    xmax_ = None
    ymax_ = None

    for xmin,ymin,xmax,ymax,precision in detections:
        xmin_, ymin_, xmax_, ymax_ = xmin, ymin, xmax, ymax
        # print(ymin,ymax,xmin,xmax)
        face = auxFrame[int(ymin):int(ymax),int(xmin):int(xmax)]
        face = cv2.resize(face,(150,150),interpolation=cv2.INTER_CUBIC)
        return face, xmin_, ymin_, xmax_, ymax_