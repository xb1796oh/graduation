# 패키지 설치
# pip install dlib opencv-python
#
# 학습 모델 다운로드 
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np

  
detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv.VideoCapture('Image/test.mp4')

#재생할 파일의 넓이와 높이
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (int(width), int(height)))

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



while (cap.isOpened()):

    ret, img_frame = cap.read()

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


    #cv.imshow('result', img_frame)
    
    out.write(img_frame)
    
    key = cv.waitKey(1)

    if key == 27:      #esc키를 눌러야 
        break
    

print("종료")
cap.release()
out.release()
cv.destroyAllWindows()