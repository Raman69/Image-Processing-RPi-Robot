import numpy as np
import cv2

chessboard = cv2.imread("Learning/OpenCV/Assets/Chess Board.jpg")
chessboard = cv2.resize(chessboard, (0, 0), fx=0.25, fy=0.25)
# Makes a greyscale image
gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)

# Detects corners on an image (Source image, Number of corners to return, Minimum certainty, Minimum distance between corners)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# Turns floating points returned by corner detection to integers
corners = np.int0(corners)

# Changes the numpy arrays to draw circles on each corner
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(chessboard, (x, y), 5, (255, 0, 0), -1)

# Draws lines of random color between each corner
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
        cv2.line(chessboard, corner1, corner2, color, 1)

cv2.imwrite("Learning/OpenCV/Assets/Edited Chessboard1.jpg", chessboard)

cv2.imshow("Image", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
