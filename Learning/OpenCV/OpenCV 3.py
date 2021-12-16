import numpy as np
import cv2

# Puts camera feed into a variable
CAM = cv2.VideoCapture(2)

while True:
    RET, FRAME = CAM.read()
    # Puts the camera width and height in variables
    WIDTH = int(CAM.get(3))
    HEIGHT = int(CAM.get(4))

    # Makes 4 copies of my camera and rotates 2 of them by 180
    image = np.zeros(FRAME.shape, np.uint8)
    smaller_frame = cv2.resize(FRAME, (0, 0), fx=0.5, fy=0.5)
    image[: HEIGHT // 2, : WIDTH // 2] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180)
    image[HEIGHT // 2 :, : WIDTH // 2] = smaller_frame
    image[: HEIGHT // 2, WIDTH // 2 :] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180)
    image[HEIGHT // 2 :, WIDTH // 2 :] = smaller_frame

    # Displays a live feed of camera
    cv2.imshow("Live Capture", image)

    # Stops camera feed when "Q" is pressed
    if cv2.waitKey(1) == ord("q"):
        break

CAM.release()
cv2.destroyAllWindows
