import numpy as np
import cv2
from colormap import rgb2hex
from scipy.linalg import circulant

def edgeDetection():
    cap = cv2.VideoCapture(0)
    cap.set(3, 512)
    cap.set(4, 1024)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # gaussian kernel
        gaussian = np.array([
            [1,2,1],
            [2,4,2],
            [1,2,1]]) * (1/16)

        smoothed = cv2.filter2D(frame, -1, gaussian)

        # sobel kernels
        horizontal = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]])

        vertical = np.array([
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]])


        horizontalMapped = cv2.filter2D(smoothed, -1, horizontal)
        verticalMapped = cv2.filter2D(smoothed, -1, vertical)
        both = cv2.bitwise_or(horizontalMapped, verticalMapped)
        both = cv2.cvtColor(both, cv2.COLOR_BGR2GRAY)
        both = np.where(both > 50, 50, 0).astype(both.dtype)
        cv2.imshow('vert', verticalMapped)
        cv2.imshow('horiz', horizontalMapped)
        cv2.imshow('both', both)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

edgeDetection()