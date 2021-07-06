# python drowsiness_test_graph.py -p shape_predictor_68_face_landmarks.dat
import cv2
import dlib
import numpy as np
import time 
import argparse
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

# 두 세트 사이의 유클리드 거리 계산
def eye_aspect_ratio(eye):
   
   # 수직 landmark (x,y) 거리
   A = dist.euclidean(eye[1], eye[5])
   B = dist.euclidean(eye[2], eye[4])

   # 수평 eye landmark (x, y) 거리
   C = dist.euclidean(eye[0], eye[3])

   # compute the eye aspect ratio
   ear = (A + B) / (2.0 * C)


   # return the eye aspect ratio
   return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
args = vars(ap.parse_args()) 

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 4

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# left, right eye landmark index
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# video stream
print("[INFO] starting video stream thread...")

cam = cv2.VideoCapture(0) #카메라 생성
if cam.isOpened() == False: #카메라 생성 확인
    exit()

#윈도우 생성 및 사이즈 변경
cv2.namedWindow('CAM_Window')

prevTime = 0 #이전 시간을 저장할 변수
FPS = 10

earlist = []
count = []

while(True):

    #카메라에서 이미지 얻기
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #현재 시간 가져오기 (초단위로 가져옴)
    curTime = time.time()

    #한번 돌아온 시간!!
    sec = curTime - prevTime

    #이전 시간을 현재시간으로 다시 저장시킴
    prevTime = curTime

    # 프레임 수를 문자열에 저장
    str = "FPS : %0.1f" % FPS

    # 표시
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    rects = detector(gray, 0)
    
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

      # coordinates to compute the eye aspect ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        earlist.append(ear)

        # eyehull
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER +=1
            
            if COUNTER >=EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Drowsy".format(TOTAL), (300,30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
        else:
            # reset the eye frame counter
            COUNTER = 0

        cv2.putText(frame, "Counter : {}".format(COUNTER), (10, 30),
         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        count.append(COUNTER)
   
    #얻어온 이미지 윈도우에 표시
    cv2.imshow('CAM_Window', frame)


    #10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
       break;


cam.release()

# ear 그래프
plt.plot(earlist)
plt.show()

plt.plot(count)
plt.show()

#윈도우 종려
cv2.destroyWindow('CAM_Window')