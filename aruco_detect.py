import numpy as np
import imutils
import cv2 as cv
import sys
import pickle
from math import *


# 초기 설정
aruco_type = "DICT_4X4_50"

camera_number = 0

base_x = 0
base_y = 0
base_z = 0
base_angle_x = 0
base_angle_y = 0
base_angle_z = 0



ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
}
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

with open('objpoints.pk', 'rb') as f:
    objpoints = pickle.load(f)
with open('imgpoints.pk', 'rb') as f:
    imgpoints = pickle.load(f)
with open('gray.pk', 'rb') as f:
    gray = pickle.load(f)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera (objpoints, imgpoints, gray.shape[::-1], None , None )

# 카메라 - 아루코마커 동차행렬 변환 반환 코드
def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.aruco_dict = cv.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv.aruco.DetectorParameters_create()
    # MARKER 검출
    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)

    # 하나 이상의 MARKER가 검출됐을 때
    if len(corners) > 0:
        font=cv.FONT_HERSHEY_SIMPLEX
        for i in range(0, len(ids)):
            # 각 MARKER의 Pose를 측정 하고 rvec 과 tvec 으로 반환
            # 10cm라서 미터 단위로 0.1로 하였지만 정확도를 위해 10으로 변경하여 cm단위로 해도 좋을 듯
            rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.1, matrix_coefficients,
                                                                       distortion_coefficients)
            # MARKER 테두리 그리기
            cv.aruco.drawDetectedMarkers(frame, corners) 

            # MARKER 축 그리기
            cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            # 동차 행렬 변환 진행
            rvec_trans = np.zeros((3, 3), np.float32)
            rvec_trans,_ = cv.Rodrigues(rvec)
            tvec_T = tvec.reshape(3, 1)

            tform = np.concatenate((rvec_trans, tvec_T), axis=1)

            a = np.array([0, 0, 0, 1])
            tform = np.vstack([tform, a])
            tform = tform.astype(np.float32)
            # 소수점 반올림
            tform=np.round(tform,3)

            print(tform)

            # 화면에 행렬 띄우기
            for i in range(4):
                text = '%2.3f, %2.3f, %2.3f, %2.3f' %(tform[i][0], tform[i][1], tform[i][2], tform[i][3])
                cv.putText(frame,text,(10, 60 + 30 * i),font,1,(0,255,0),2)

    else:
        tform = []
    return frame, tform

# 카메라 - 베이스 변환
def base_cam_matrix(x,y,z,x_angle=0,y_angle=0,z_angle=0):
    # 베이스와 카메라의 차이를 입력
    # x, y, z 거리 (거리는 위 인식 단계에서 설정한 단위로 됨. 0.1의 경우 미터 단위, 10일 경우 cm단위)
    # 뒤 3개는 각도, 360도 단위

    base_cam_trans=np.identity(n=4,dtype=np.float32)
    base_cam_trans[0][3] = x
    base_cam_trans[1][3] = y
    base_cam_trans[2][3] = z
    R=angle_rotation(x_angle,y_angle,z_angle)
    base_cam=np.dot(base_cam_trans,R)
    base_cam=np.round(base_cam,3)
    return base_cam

def angle_rotation(x_angle,y_angle,z_angle):
    x=x_angle*pi/180
    y = y_angle * pi / 180
    z = z_angle * pi / 180

    R_x=np.array([[1,0,0,0],[0,cos(x),-sin(x),0],[0,sin(x),cos(x),0],[0,0,0,1]])
    R_y=np.array([[cos(y),0,sin(y),0],[0,1,0,0],[-sin(y),0,cos(y),0],[0,0,0,1]])
    R_z=np.array([[cos(z),-sin(z),0,0],[sin(z),cos(z),0,0],[0,0,1,0],[0,0,0,1]])
    R=np.dot(R_x,R_y)
    R=np.dot(R,R_z)
    return R



base_cam = base_cam_matrix(base_x, base_y, base_z,
                           base_angle_z, base_angle_y, base_angle_z)
print(base_cam)

capture = cv.VideoCapture(camera_number)
# capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

tform = []

while True:
    ret, frame = capture.read()
    # cv.imshow("VideoFrame", frame)
    
    det_frame, raw_tform = pose_esitmation(frame, ARUCO_DICT[aruco_type], mtx, dist)
    if raw_tform != []:
        tform = raw_tform

    cv.imshow("VideoFrame", det_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
capture.release()
print(tform)

targetf=np.identity(n=4,dtype=np.float32)
while True:

    num_list = list(map(float, input().split()))
    targetf[0][3]=num_list[0]
    targetf[1][3]=num_list[1]
    targetf[2][3]=num_list[2]
    print(targetf)
    real_target = np.dot(base_cam, tform)
    real_target = np.dot(real_target, targetf)
    real_target=np.round(real_target,3)

    print(real_target)
    if real_target[2][3]<2:
        real_target[2][3]=0
    elif real_target[2][3]>3:
        real_target[0][3]=real_target[0][3]+3
    x, y, z = real_target[0][3], real_target[1][3], real_target[2][3]

    print(x, y, z)