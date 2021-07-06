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

face3Dmodel = world.ref3DModel()

class head_ROI(object):
    """
    predicttion Roi based on headpose and Field of view
    """

    def __init__(self):
        self.frame = None
        self.shape = None
        pixel = 2
        w = 640
        h = 640
        center = [w/2,h/2]
        Cam_w = 36*pixel/2
        Cam_h = 24*pixel/2
        self.distance = 50 * pixel
        self.Cam_space = [(w/2-Cam_w, h/2+Cam_h),(w/2-Cam_w, h/2-Cam_h),(w/2+Cam_w, h/2 -Cam_h),(w/2+Cam_w, h/2+Cam_h)]
        self.focus_space = self.set_focusBox( self.distance, center)
        self.Roi_space = self.set_veiwBox(self.distance, center)
        self.space_img = np.full((640, 640, 3),255, dtype = np.uint8) 

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
        print("a", a)
        print("b", b)
        ro_box= []
        i=0
        for x,y in box:
            
            i,j=self.rotate(center,(x,y),angle_z)
            i += a
            j += b
            ro_box.append([int(i),int(j)])
        
        return ro_box


    def set_position(self, box, position, eye):
        sum_x=0
        sum_y=0

        for x,y in box[:]:
            sum_x += x  
            sum_y += y

        center=(sum_x/4,sum_y/4)


        return box


    def predictor(self,shape, frame):
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
        print("ThetaX: ", x)
        print("ThetaY: ", y)
        print("ThetaZ: ", z)
        
        view_box = self.box_rotation(self.Roi_space,math.radians(180)-x,-y,math.radians(90)-z,self.distance)
        focus_box = self.box_rotation(self.focus_space,math.radians(180)-x,-y,math.radians(90)-z,self.distance)

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

        return GAZE, frame, self.space_img


    





