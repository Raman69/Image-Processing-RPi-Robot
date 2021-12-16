import numpy as np
import cv2

CAM = cv2.VideoCapture(2)

while True:
    ret, FRAME = CAM.read()
    WIDTH = int(CAM.get(3))
    HEIGHT = int(CAM.get(4))

    # Defines the font used for text on the image
    FONT = cv2.FONT_HERSHEY_DUPLEX
    # Makes 2 line which cross each other (Image to draw on, Start point, End point, Color, Thickness)
    line = cv2.line(FRAME, (0, 0), (WIDTH, HEIGHT), (255, 0, 0), 10)
    line = cv2.line(FRAME, (0, HEIGHT), (WIDTH, 0), (128, 255, 0), 10)
    # Draws a rectangle (Image to draw on, Top-left corner, Bottom-right corner, Color, Thickness)
    rectangle = cv2.rectangle(FRAME, (0, 0), (WIDTH, HEIGHT), (128, 128, 128), 10)
    # Draws a circle (Image to draw on, Middle point, Radius, Color, Thickness)
    circle = cv2.circle(FRAME, (325, 400), 34, (0, 0, 255), -1)
    # Writes text (Image to write on, Text, Bottom-left corner, Font, Text scale, Color, Thickness, Line type)
    text = cv2.putText(
        FRAME, "wagwan", (200, 45), FONT, 2, (255, 255, 255), 5, cv2.LINE_AA
    )

    cv2.imshow("Live Capture", FRAME)

    if cv2.waitKey(1) == ord("q"):
        break

CAM.release()
cv2.destroyAllWindows
