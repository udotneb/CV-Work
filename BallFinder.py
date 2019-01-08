import numpy as np
import cv2
from colormap import rgb2hex
from scipy.linalg import circulant


class BallFinder:
    # [ 80, 20, 100] [100, 255, 255]
    def __init__(self):
        self.refPt = [0,0]
        self.lowerHSV = np.array([80, 20, 100]).astype(np.uint8)
        self.higherHSV = np.array([100, 255, 255]).astype(np.uint8)
        self.clicked = False
        self.locked = False
        self.currentStep = 0
        self.WIDTH = 512
        self.HEIGHT = 1024

    def nothing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[0] = int(y)
            refPt[1] = int(x)
            self.clicked = True
        pass

    def checkTennisBall(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.WIDTH)
        cap.set(4, self.HEIGHT)
        trackingLength = 5
        previousLocations = np.full((trackingLength,2), 0).astype(np.uint32) # holds the previous coordinates, 
        cv2.namedWindow('hsv')
        cv2.setMouseCallback("hsv", self.nothing)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            smoothed = cv2.GaussianBlur(frame, (5, 5), 0)
            
            hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
            '''
            if not self.clicked and not self.locked:
                referenceColor = hsv[self.refPt[0]][self.refPt[1]]
                self.lowerHSV = np.array([referenceColor[0] - 10, 20, 100]).astype(np.uint8)
                self.higherHSV = np.array([referenceColor[0] + 10, 255, 255]).astype(np.uint8)
            if self.clicked:
                referenceColor = hsv[self.refPt[0]][self.refPt[1]]
                self.lowerHSV = np.array([referenceColor[0] - 10, 20, 100]).astype(np.uint8)
                self.higherHSV = np.array([referenceColor[0] + 10, 255, 255]).astype(np.uint8)
                self.clicked = False
                self.locked = True
            '''
            #print(self.lowerHSV, self.higherHSV)

            mask = cv2.inRange(hsv, self.lowerHSV, self.higherHSV) # filters out colors that arent close to tennis ball

            ones = np.full(frame.shape, 255)

            res = cv2.bitwise_and(ones,ones, mask= mask).astype(frame.dtype) # turns the mask into a bw colored image

            threeKernel = np.ones((3,3), np.uint8)
            res = cv2.cv2.morphologyEx(res, cv2.MORPH_OPEN, threeKernel)
            res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, threeKernel)

            edges = cv2.Canny(res, 0, 1) # gets the edges of the mask

            res2, cnts, hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            largestArea = sorted(cnts, key = cv2.contourArea, reverse = True) # gets the largest contour area

            if len(largestArea) > 0:

                (x,y),radius = cv2.minEnclosingCircle(largestArea[0])
                if radius > 40:
                    center = (int(x),int(y))

                    previousCoordinate = previousLocations[self.currentStep - trackingLength + 1]
                    radius = int(radius)
                    cv2.circle(frame, center,radius,(0,255,0),2)

                    # find difference between current and next coordinate 
                    nextCoordinate = np.add(np.subtract(center, previousCoordinate),center)
                    nextCoordinate[0] = min(self.WIDTH - 1, nextCoordinate[0])
                    nextCoordinate[1] = min(self.HEIGHT - 1, nextCoordinate[1])
                    print(nextCoordinate.dtype)
                    cv2.line(frame, center, tuple(nextCoordinate),(0,0,255),5)
                    previousLocations[(self.currentStep - trackingLength) + 1] = center

                    self.currentStep = (self.currentStep + 1) % trackingLength
            
            cv2.imshow('edges', res)
            cv2.imshow('drawn', frame)
            cv2.imshow('hsv', hsv)
            
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


x = BallFinder()
x.checkTennisBall()
