# python dlib68_aspect_time.py -p shape_predictor_68_face_landmarks.dat
import cv2
import dlib
import numpy as np
import time 
import argparse
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from headpose_estimation import head_ROI

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
 

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 4

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

ROI = head_ROI()
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('./dlib_/shape_predictor_68_face_landmarks.dat')

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

space_img = np.full((640, 640, 3),255, dtype = np.uint8)
prev_time = 0 #이전 시간을 저장할 변수
FPS = 10
detect_exist = False
detect_drowsy = False;

earlist = []
exist_list = []
drowsy_list = []
check_exist = 0
leaving_time = 0
while(True):
    GAZE = "Face Not Found"
    ret, frame = cam.read()

    current_time = time.time() - prev_time
    print(time.time())
    print("FPS:",1./current_time);

    if(ret is True) and (current_time >1./FPS):
        prev_time = time.time()
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        if not rects:
            detect_exist = False;
            if check_exist == 0 :
                start_time = int(time.time())
            else:
                leaving_time = int(time.time()) - start_time 

            check_exist += 1    
            

        for rect in rects:

            detect_exist = True
            check_exist = 0
            shape = predictor(gray, rect)

            GAZE, frame, space_img = ROI.predictor(shape, frame)
            
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
                    detect_drowsy = True;
           
            else:
                # reset the eye frame counter
                COUNTER = 0
                detect_drowsy = False;

               

        drowsy_list.append(detect_drowsy)
        exist_list.append(detect_exist)  

    
        msg_exist = "leaving " + str(leaving_time) + "s"
        msg_drowsy= "Not drowsy"

        if detect_exist == True:
            msg_exist = "exist"
            
            if detect_drowsy == True:
                msg_drowsy = "Drowsy"
            cv2.putText(frame, msg_drowsy, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, msg_exist, (400,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
   
        

        cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("test", space_img)
        #얻어온 이미지 윈도우에 표시
        cv2.imshow('CAM_Window', frame)

    #10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
       break;

cam.release()

# 종횡비 그래프

plt.plot(exist_list)
plt.plot(drowsy_list)
plt.show()

#윈도우 종료
cv2.destroyWindow('CAM_Window')