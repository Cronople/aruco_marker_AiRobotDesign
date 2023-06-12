###############################
# Camera Distortion 보정      #
# 이미 보정된 상태라면 넘어가기 #
###############################

import numpy as np
import cv2 as cv
import pickle

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

capture = cv.VideoCapture(0)
while True:
    ret, frame = capture.read()
    # cv.imshow("VideoFrame", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (8,5), corners2, ret)
        cv.imshow('VideoFrame', frame)
    else:
        cv.imshow("VideoFrame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

with open('objpoints.pk', 'wb') as f:
    pickle.dump(objpoints, f)
with open('imgpoints.pk', 'wb') as f:
    pickle.dump(imgpoints, f)
with open('gray.pk', 'wb') as f:
    pickle.dump(gray, f)

cv.destroyAllWindows()