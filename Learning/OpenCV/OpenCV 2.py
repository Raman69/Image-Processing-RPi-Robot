import cv2
import random

ME_WITH_TEXT = cv2.imread("Learning/OpenCV/Assets/Me With Text.jpg", -1)

# Changes the First 100 Rows to Random Colours
for i in range(100):
    for j in range(ME_WITH_TEXT.shape[1]):
        ME_WITH_TEXT[i][j] = [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ]

print(ME_WITH_TEXT.shape)

# Copies Part of the Image and Pastes it Elsewhere (My face in this Case)
face = ME_WITH_TEXT[100:350, 300:500]
ME_WITH_TEXT[230:480, 0:200] = face

# Saves an Image of the Copied Segment (My face)
cv2.imwrite("OpenCV Tutorial/Assets/Cropped face.jpg", face)

cv2.imwrite("OpenCV Tutorial/Assets/Edited Image2.jpg", ME_WITH_TEXT)

# Shows the Image in a New Window (Needs X11 Forwarding When Using SSH)
cv2.imshow("Image", ME_WITH_TEXT)
cv2.waitKey(0)
cv2.destroyAllWindows()
