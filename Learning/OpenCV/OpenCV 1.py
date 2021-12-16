# Imports Library
import cv2

# Loads an Image (-1 = Unchanged , 0 = Greyscale , 1 = Colour/Ignore Alpha Channels)
me_with_text = cv2.imread("Learning/OpenCV/Assets/Me With Text.jpg", 1)
# Resizes the Image (Halves Image Size in this Case)
me_with_text = cv2.resize(me_with_text, (0, 0), fx=0.5, fy=0.5)
# Rotates the image (By 180 Degrees)
me_with_text = cv2.rotate(me_with_text, cv2.cv2.ROTATE_180)

# Saves the Image
cv2.imwrite("OpenCV Tutorial/Assets/Edited Image.jpg", me_with_text)

# Shows the Image in a New Window (Needs X11 Forwarding When Using SSH)
cv2.imshow("Image", me_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
