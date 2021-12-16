import numpy as np
import cv2

CAM = cv2.VideoCapture(2)
# Defines pre-trained cascades/filters
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


while True:
    RET, FRAME = CAM.read()
    WIDTH = int(CAM.get(3))
    HEIGHT = int(CAM.get(4))
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)

    # Searches for faces in image (Source, Scale factor, Accuracy, Min Size, Max Size)
    faces = FACE_CASCADE.detectMultiScale(GRAY, 1.3, 5)
    for (x, y, w, h) in faces:
        # Draws rectangle on face
        cv2.rectangle(FRAME, (x, y), (x + w, y + h), (255, 0, 0), 5)
        # Uses face coordinates to find eyes
        roi_gray = GRAY[y : y + h, x : x + w]
        roi_color = FRAME[y : y + h, x : x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            # Draws rectangle on eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow("Live Capture", FRAME)

    if cv2.waitKey(1) == ord("q"):
        break

CAM.release()
cv2.destroyAllWindows
