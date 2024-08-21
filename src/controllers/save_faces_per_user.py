from src.utils.save_face import save_face
import cv2

def save_faces_per_user():
    video = cv2.VideoCapture(0)
    path = 'D:\\UNMSM\\Ciclo X\\Desarrollo de proyecto de tesis II\\Proyecto\\facial_recognition\\src\\resources\\jesus'
    num_img = 1
    name = 'jesus'
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

        if iterator % 20 == 0:
            xmin_ = None
            ymin_ = None
            xmax_ = None
            ymax_ = None

            frame_copy = frame.copy()
            xmin_, ymin_, xmax_, ymax_ = save_face(frame_copy, path, name + str(num_img))
            num_img += 1
            
        cv2.imshow("Real Time", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()