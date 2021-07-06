#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import argparse
import numpy as np
import math

#import Face Recognition
import face_recognition

# helper modules
from drawFace import draw
import reference_world as world

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal",
                    type=int, default=1,
                    help="Callibrated Focal Length of the camera")
parser.add_argument("-s", "--camsource", type=int, default=0,
	help="Enter the camera source")

args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()


def set_veiwBox (distance,center):
    
    w = math.tan(math.radians(62)) * distance
    h_up = math.tan(math.radians(50)) * distance
    h_down = math.tan(math.radians(70)) * distance
    x = center[0] 
    y = center[1]
    view_box= [(x-w,y+h_down),(x-w,y-h_up),(x+w,y-h_up),(x+w,y+h_down)]


    return view_box


def set_focusBox (distance,center):
    
    w = math.tan(math.radians(30)) * distance
    h_up = math.tan(math.radians(25)) * distance
    h_down = math.tan(math.radians(30)) * distance
    x = center[0] 
    y = center[1]
    focus_box= [(x-w,y+h_down),(x-w,y-h_up),(x+w,y-h_up),(x+w,y+h_down)]


    return focus_box

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def box_rotation(box,angle_x,angle_y,angle_z,distance):
    
    sum_x=0
    sum_y=0

    for x,y in box[:]:
        sum_x += x  
        sum_y += y

    center=(sum_x/4,sum_y/4)
  
    a = math.tan(angle_y) * distance
    b = math.tan(angle_x) * distance
    print("a", a)
    print("b", b)
    ro_box= []
    i=0
    for x,y in box:
        
        i,j=rotate(center,(x,y),angle_z)
        i += a
        j += b
        ro_box.append([int(i),int(j)])
    
    return ro_box

pixel = 2
w = 640
h = 640
center = [w/2,h/2]
Cam_w = 36*pixel/2
Cam_h = 24*pixel/2

distance = 50 * pixel

Cam_space = [(w/2-Cam_w, h/2+Cam_h),(w/2-Cam_w, h/2-Cam_h),(w/2+Cam_w, h/2 -Cam_h),(w/2+Cam_w, h/2+Cam_h)]

focus_space = set_focusBox(distance, center)

Roi_space = set_veiwBox(distance, center)

def main():
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(args["camsource"])

    while True:
        GAZE = "Face Not Found"
        ret, img = cap.read()
        if not ret:
            print(f'[ERROR - System]Cannot read from source: {args["camsource"]}')
            break
       # print("pass 1")
        #faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        faces = face_recognition.face_locations(img)
        space_img = np.full((640, 640, 3),255, dtype = np.uint8) 
       # print("pass 2")
        for face in faces:
           # print("pass 3")
           
            #Extracting the co cordinates to convert them into dlib rectangle object
            x = int(face[3])
            y = int(face[0])
            w = int(abs(face[1]-x))
            h = int(abs(face[2]-y))
            u=int(face[1])
            v=int(face[2])

            newrect = dlib.rectangle(x,y,u,v)
            cv2.rectangle(img, (x, y), (x+w, y+h),
            (0, 255, 0), 2)

            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(img, shape)
            #print("pass 4")
            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            focalLength = args["focal"] * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)
           # print("pass 5")
            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
           # print("pass 6")
            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(img, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)
            # print("pass 7")
            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            print("ThetaX: ", x)
            print("ThetaY: ", y)
            print("ThetaZ: ", z)
            
            view_box = box_rotation(Roi_space,math.radians(180)-x,-y,math.radians(90)-z,distance)
            focus_box = box_rotation(focus_space,math.radians(180)-x,-y,math.radians(90)-z,distance)
            cv2.polylines(space_img, [np.array(view_box)], True,(0,0,255),1)
            cv2.polylines(space_img, [np.array(focus_box)], True,(0,255,0),1)
            print('*' * 80)
            if angles[1] < -25:
                GAZE = "Looking: Left"
            elif angles[1] > 25:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"
                
       # print("pass 8")
        cv2.polylines(space_img, [np.array(Cam_space, dtype=np.int64)], True,(0,0,0),1)
        #cv2.polylines(space_img, [np.array(Roi_space, dtype = np.int64)], True,(0,255,0),1)
        

        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", img)
        cv2.imshow("test", space_img)
       # print("pass 9")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("pass 10")



    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # path to your video file or camera serial
    main()
