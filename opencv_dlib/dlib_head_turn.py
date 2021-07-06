import numpy as np
import cv2 as cv
import dlib
from math import *#import math

# face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def getAngle3P(p1, p2, p3):#세점 사이의 각도 1->2->3
   
    tan1 = (p1[0] - p2[0])/(p1[1]-p2[1])
    tan2 = (p3[0] - p2[0])/(p3[1]-p2[1])

    tanf = (tan1 - tan2) / (1 + tan1*tan2)

    degree = atan(tanf)

    return abs(degree)

    """
    ang1 = atan2(*pt1[::-1])
    ang2 = atan2(*pt2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))

    res = (res + 360) % 360
    if direction == "CCW":    #반시계방향
        res = (360 - res) % 360
    
    return res
    """

video = cv.VideoCapture(0, cv.CAP_DSHOW)

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

index = ALL

INCLINE1 = list(range(27,28))
INCLINE2 = list(range(30, 31))


while True:

    ret, img_frame = video.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2RGB)

    dets = detector(img_gray, 1)

    for face in dets:
        shape = predictor(img_frame, face)

        list_points = []
        i=0
        for p in shape.parts():
            list_points.append([p.x, p.y])
        
        list_points = np.array(list_points)

        for i,pt in enumerate(list_points[INCLINE1]):
            pt_pos = (pt[0], pt[1])
            fpx = pt[0]
            fpy = pt[1]
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        for i,pt in enumerate(list_points[INCLINE2]):
            pt_pos = (pt[0], pt[1])
            npx = pt[0]
            npy = pt[1]
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        cv.rectangle(img_frame,(face.left(), face.top()), (face.right(), face.bottom()),
        (0, 0, 255), 3)
        cv.imshow('capture', img_frame)

    key = cv.waitKey(1)

    if key==49:
        cv.imwrite('image/norm.jpg', img_frame)
        p1x = fpx
        p1y = fpy
        p2x = npx
        p2y = npy
        print(p1x, p1y, p2x, p2y)
        print('----------------------')
        break

video.release()
cv.destroyAllWindows()


print('%d %d', p1x, p1y)
print('%d %d', p2x, p2y)



video = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    ret, img_frame = video.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2RGB)

    dets = detector(img_gray, 1)

    for face in dets:
        shape = predictor(img_frame, face)

        list_points = []
        #i=0
        for p in shape.parts():
            list_points.append([p.x, p.y])
        
        list_points = np.array(list_points)
        
        for i,pt in enumerate(list_points[INCLINE1]):
            pt_pos = (pt[0], pt[1])
            fpx = pt[0]
            fpy = pt[1]
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        for i,pt in enumerate(list_points[INCLINE2]):
            pt_pos = (pt[0], pt[1])
            npx = pt[0]
            npy = pt[1]
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        dx = fpx - p1x
        dy = fpy- p1y
        print(fpx, fpy, npx, npy)
        print(dx, dy)

        fpx = fpx - dx
        fpy = fpy - dy
        npx = npx - dx
        npy = npy - dy

        print(fpx, fpy, npx, npy)
        print('--------------------------')

        cv.circle(img_frame, (fpx, fpy), 2, (0, 0, 255), -1)
        cv.circle(img_frame, (npx, npy), 2, (0, 0, 255), -1)

        cv.circle(img_frame, (p1x, p1y), 2, (255, 0, 0), -1)
        cv.circle(img_frame, (p2x, p2y), 2, (255, 0, 0), -1)

        degree = getAngle3P((p2x, p2y), (fpx, fpy), (npx, npy))
        
        alarm = tan(degree)
        print(alarm)

        if abs(alarm) > 0.2:
            cv.imshow('capture', img_gray)
        
        else:
            cv.imshow('capture', img_frame)

        #cv.rectangle(img_frame,(face.left(), face.top()), (face.right(), face.bottom()),
        #(0, 0, 255), 3)
        #cv.imshow('capture', img_frame)

        

    key = cv.waitKey(1)

    if key == 27:
        break



video.release()
cv.destroyAllWindows()
