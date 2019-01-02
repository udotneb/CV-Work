import numpy as np
import cv2
from colormap import rgb2hex
from scipy.linalg import circulant

def nothing(x):
    pass

def pyramid():
    cap = cv2.VideoCapture(0)
    cap.set(3, 512)
    cap.set(4, 1024)
    cv2.namedWindow('image')
    cv2.createTrackbar('thresh','image',0,70,nothing)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        lower_reso = cv2.pyrDown(frame)
        higher_reso = cv2.pyrUp(lower_reso)

        frame = cv2.subtract(frame, higher_reso) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.getTrackbarPos('thresh','image')
        frame = np.where(frame > thresh, frame * 10, 0)
        cv2.imshow('image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

pyramid()