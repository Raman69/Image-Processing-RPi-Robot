#This program takes a picture and a 5 second video

from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.rotation = 180

camera.start_preview()
camera.start_recording("/home/pi/Videos/Video.h264")
sleep(5)
camera.capture("/home/pi/Pictures/Image.jpg")
camera.stop_recording()
camera.stop_preview()