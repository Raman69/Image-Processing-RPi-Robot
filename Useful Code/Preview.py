#This program Previews the camera

from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.rotation = 180

camera.start_preview()

stop = input("Enter 1 to stop preview: ")

if stop == "1":
    camera.stop_preview()