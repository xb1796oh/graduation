# 패키지 설치
# pip install dlib opencv-python
#
# 학습 모델 다운로드 
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np
import time

  
detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv.VideoCapture(0)

prev_time = 0
FPS = 10

# range는 끝값이 포함안됨   
ALL = list(range(0, 68)) 
RIGHT_EYEBROW = list(range(17, 22))  
LEFT_EYEBROW = list(range(22, 27))  
RIGHT_EYE = list(range(36, 42))  
LEFT_EYE = list(range(42, 48))  
NOSE = list(range(27, 36))  
MOUTH_OUTLINE = list(range(48, 61))  
MOUTH_INNER = list(range(61, 68)) 
JAWLINE = list(range(0, 17)) 

index = RIGHT_EYE + LEFT_EYE + NOSE + MOUTH_OUTLINE

while True:

    ret, img_frame = cap.read()

    current_time = time.time() - prev_time

    if (ret is True) and (current_time > 1./ FPS) :
    	
        prev_time = time.time()
        img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)


        dets = detector(img_gray, 1)


        for face in dets:

            shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기

            list_points = []
            for p in shape.parts():
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)


            for i,pt in enumerate(list_points[index]):

                pt_pos = (pt[0], pt[1])
                cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

            
            cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                (0, 0, 255), 3)

            print("nose : x %d, y %d"%(list_points[30][0],list_points[30][1]))


        str = " FPS %d" % FPS 
        cv.putText(img_frame, str, (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv.imshow('result', img_frame)
 
    
    key = cv.waitKey(1)

    if key == 27:      #esc키를 눌러야 
        break
    
    elif key == ord('1'):
        FPS = 7
    elif key == ord('2'):
        FPS = 5
    elif key == ord('3'):
        FPS = 1
    elif key == ord('4'):
        FPS = 10
    elif key == ord('5'):
        FPS = 15
    elif key == ord('6'):
        FPS = 20    


cap.release()