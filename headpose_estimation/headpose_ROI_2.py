#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import numpy as np
import math

# helper modules
from .drawFace import draw
from . import reference_world as world
from . import IOU

face3Dmodel = world.ref3DModel()

class head_ROI(object):
    """
    predicttion Roi based on headpose and Field of view
    """

    def __init__(self):
        self.frame = None
        self.shape = None
        self.pixel = 2
        w = 640
        h = 640
        self.center = [w/2,h/2]
        Cam_w = 36*self.pixel/2
        Cam_h = 24*self.pixel/2
        self.real_distance = 50
        self.distance = self.real_distance * self.pixel
        self.Cam_space = [(w/2-Cam_w, h/2+Cam_h),(w/2-Cam_w, h/2-Cam_h),(w/2+Cam_w, h/2 -Cam_h),(w/2+Cam_w, h/2+Cam_h)]
        self.focus_space = self.set_focusBox( self.distance, self.center)
        self.Roi_space = self.set_veiwBox(self.distance, self.center)
        self.space_img = np.full((640, 640, 3),255, dtype = np.uint8) 

    def my_distance(self,point1,point2):
        dist = math.sqrt(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2))
        return dist
    
    
    def cul_distance(self, shape):
        check_point= [27,31,36,45]

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        h = self.my_distance(list_points[check_point[0]],list_points[check_point[1]])
        #print("distance of h :",int(h));
        d = self.my_distance(list_points[check_point[2]],list_points[check_point[3]])
        #print("distance of d :",int(d));

        """
        10cm 눈 D 거리 기준
        6cm 코 H 거리 기준 
        60cm D pixel내 87

        """
        tan_theta_com = 5/60

        L_c = 43.5/tan_theta_com

        L_d = 6/((d/2)/L_c)

        L_h = 4/((h/2)/L_c)
        #print("distance of L :",int(L_h));
        #print("distance of L :",int(L_d));

        self.real_distance = (L_h +L_d) /2
        self.distance = self.real_distance * self.pixel

        return h, d




    def set_veiwBox (self, distance,center):
        
        w = math.tan(math.radians(62)) * distance
        h_up = math.tan(math.radians(50)) * distance
        h_down = math.tan(math.radians(70)) * distance
        x = center[0] 
        y = center[1]
        view_box= [(x-w,y+h_down),(x-w,y-h_up),(x+w,y-h_up),(x+w,y+h_down)]

        return view_box


    def set_focusBox (self, distance,center):
        
        w = math.tan(math.radians(30)) * distance
        h_up = math.tan(math.radians(25)) * distance
        h_down = math.tan(math.radians(30)) * distance
        x = center[0] 
        y = center[1]
        focus_box= [(x-w,y+h_down),(x-w,y-h_up),(x+w,y-h_up),(x+w,y+h_down)]


        return focus_box
    

    def set_center(self, center,angle_x,angle_y):

        a = math.tan(angle_y) * self.real_distance
        b = math.tan(angle_x) * self.real_distance

        gap_a = (center[0] - a) - 320
        gap_b = (center[1] - b) - 240
        self.center[0] = 320 + gap_a
        self.center[1] = 320 + gap_b

        print( self.center)



    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def box_rotation(self, box,angle_x,angle_y,angle_z,distance):
        
        sum_x=0
        sum_y=0

        for x,y in box[:]:
            sum_x += x  
            sum_y += y

        center=(sum_x/4,sum_y/4)
    
        a = math.tan(angle_y) * distance
        b = math.tan(angle_x) * distance
        #print("a", a)
        #print("b", b)
        ro_box= []
        
        for x,y in box:
            
            i,j=self.rotate(center,(x,y),angle_z)
            i += a
            j += b
            ro_box.append([int(i),int(j)])
        
        return ro_box


    def set_eye_position(self, box, view_box,eye):
        sum_x=0
        sum_y=0

        for x,y in box[:]:
            sum_x += x  
            sum_y += y

        center=(sum_x/4,sum_y/4)


        """
        right 0 < middle 0.5 < left 1
        top 0 < middle 0.5 < bottom 1
        """
        #print("eye",eye)
        hor = (0.5 - float(eye[0]))*100
        theta_hor = 62*(abs(int(hor))/50)
        

        print("eye hor : ",(abs(int(hor))/50))
        

        hor_x = math.tan(theta_hor) * self.distance
        if hor < 0:
            hor_x *= -1
        
        ver = (0.5 - float(eye[1])) * 100
        print("eye ver : ", (abs(int(ver))/50))
        if ver <= 0 :
            theta_ver = 50*(abs(int(ver))/50)
            ver_y = -(math.tan(theta_ver) * self.distance)

        else :
            theta_ver = 70*(abs(int(ver))/50)
            ver_y = math.tan(theta_ver) * self.distance


        ro_box= []
        
        for x,y in box:
            
            i = x + hor_x
            j = y + ver_y
            ro_box.append([int(i),int(j)])
        
        for i in range(0,len(view_box)):
            if ro_box[i][0]>=view_box[i][0] or ro_box[i][1]>=view_box[i][1]:
                return box

        return ro_box



    def predictor(self,shape, frame, eye,face_center):
        
        h,d =self.cul_distance(shape)
        self.space_img = np.full((640, 640, 3),255, dtype = np.uint8) 

        refImgPts = world.ref2dImagePoints(shape)
        
        height, width, channels = frame.shape
        focalLength = 1 * width
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
        cv2.line(frame, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)
        # print("pass 7")
        # calculating euler angles
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
        '''
        print("ThetaX: ", x)
        print("ThetaY: ", y)
        print("ThetaZ: ", z)
        '''
        #self.set_center(face_center,math.radians(180)-x,-y)
        self.focus_space = self.set_focusBox( self.distance, self.center)
        self.Roi_space = self.set_veiwBox(self.distance, self.center)
        
        view_box = self.box_rotation(self.Roi_space,math.radians(180)-x,-y,math.radians(90)-z,self.distance)
        focus_box = self.box_rotation(self.focus_space,math.radians(180)-x,-y,math.radians(90)-z,self.distance)

        focus_iou = IOU.cul_IOU(self.Cam_space, focus_box)
        #print(focus_iou)
        #focus_box = self.set_eye_position(focus_box,view_box,eye)

        Cam_box = np.array(self.Cam_space,dtype=np.int64)
        cv2.polylines(self.space_img, [np.array(view_box)], True,(0,0,255),1)
        cv2.polylines(self.space_img, [np.array(focus_box)], True,(0,255,0),1)
        cv2.polylines(self.space_img, [Cam_box], True,(0,0,0),1)
        if angles[1] < -25:
            GAZE = "Looking: Left"
        elif angles[1] > 25:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"

        return GAZE, frame, self.space_img, focus_iou


    





