import numpy as np
import cv2
from colormap import rgb2hex
from scipy.linalg import circulant

def checkTennisBall():
    cap = cv2.VideoCapture(0)
    cap.set(3, 512)
    cap.set(4, 1024)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        smoothed = cv2.GaussianBlur(frame, (5, 5), 0)
        
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

        lowerHSV = (0, 0, 150)
        higherHSV = (50, 100, 255)
        
        mask = cv2.inRange(hsv, lowerHSV, higherHSV)
        
        ones = np.ones(frame.shape)
        res = cv2.bitwise_and(ones,ones, mask= mask)

        kernelFive = np.ones((5,5)) / 25
        kernelThree = np.ones((3,3)) / 9
        
        res = cv2.erode(res, kernelThree, iterations = 1)
        res = cv2.dilate(res, kernelThree, iterations = 1)
        #cv2.drawContours(res, contours, -1, (0,255,0), 3)
        
        cv2.imshow('frame', res)
        cv2.imshow('before', lab)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def edgeDetection():
    cap = cv2.VideoCapture(0)
    cap.set(3, 512)
    cap.set(4, 1024)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        horizontal = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]])

        vertical = np.array([
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]])
        horizontalMapped = cv2.filter2D(frame, -1, horizontal)
        verticalMapped = cv2.filter2D(frame, -1, vertical)
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
