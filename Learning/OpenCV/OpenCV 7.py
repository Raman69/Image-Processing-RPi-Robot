import numpy as np
import cv2

SOURCE = cv2.resize(
    cv2.imread("Learning/OpenCV/Assets/Football Practice.jpg", 0),
    (0, 0),
    fx=0.8,
    fy=0.8,
)
TEMPLATE = cv2.resize(
    cv2.imread("Learning/OpenCV/Assets/Ball.PNG", 0), (0, 0), fx=0.8, fy=0.8
)
H, W = TEMPLATE.shape

# A list of all template matching methods
METHODS = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED,
]

for method in METHODS:
    SOURCE_2 = SOURCE.copy()
    # Attempts to match the template to the source image (Source image, Template, Method)
    result = cv2.matchTemplate(SOURCE_2, TEMPLATE, method)
    # Draws rectangle on matched item
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + W, location[1] + H)
    cv2.rectangle(SOURCE_2, location, bottom_right, 255, 5)
    cv2.imshow("Match", SOURCE_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows
