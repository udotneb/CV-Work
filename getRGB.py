import cv2
import numpy as np

lowerRed = np.uint8([[[130,100,100]]])
upperRed = np.uint8([[[200, 218, 186]]])

def hsvCon(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2HSV)[0][0]

print(hsvCon(lowerRed))
print(hsvCon(upperRed))
