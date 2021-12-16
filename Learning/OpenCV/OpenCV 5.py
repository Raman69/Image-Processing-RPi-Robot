import numpy as np
import cv2

CAM = cv2.VideoCapture(2)

while True:
    RET, FRAME = CAM.read()
    WIDTH = int(CAM.get(3))
    HEIGHT = int(CAM.get(4))

    # Converts BGR image to HSV
    HSV = cv2.cvtColor(FRAME, cv2.COLOR_BGR2HSV)
    # Defines boundaries of the color blue
    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])

    # Produces a mask with only the colour blue
    mask = cv2.inRange(HSV, LOWER_BLUE, UPPER_BLUE)
    # Converts any pixels outside the blue boundaries of black
    result = cv2.bitwise_and(FRAME, FRAME, mask=mask)

    cv2.imshow("Live Capture", result)

    if cv2.waitKey(1) == ord("q"):
        break

CAM.release()
cv2.destroyAllWindows
