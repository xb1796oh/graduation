#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import numpy as np
import math

# helper modules

from . import IOU

class head_ROI(object):
    """
    predicttion Roi based on headpose and Field of view
    """

    def __init__(self):
        self.frame = None
        self.shape = None
        self.pixel = 200
        self.real_distance = 0.5
        w = 640
        h = 480
        self.center = [w/2,h/2]
        
        self.window_w = (0.36*self.pixel)
        self.window_h = (0.24*self.pixel)
       
        self.distance = self.real_distance * self.pixel
        cw = round(self.window_w/2)
        ch = round(self.window_h/2)
        self.ROI_space = [(w/2-cw, h/2+2*ch),(w/2-cw, h/2),(w/2+cw, h/2),(w/2+cw, h/2+2*ch)]

        self.veiw_space = self.set_focusBox()
        

    def my_distance(self,dist):
        self.real_distance = dist
        self.distance = dist * self.pixel


    def set_focusBox (self):
        
        w = math.tan(math.radians(30)) * self.distance
        h_up = math.tan(math.radians(25)) * self.distance
        h_down = math.tan(math.radians(30)) * self.distance
        x = self.center[0] 
        y = self.center[1]
        focus_box= [(x-w,y+h_down),(x-w,y-h_up),(x+w,y-h_up),(x+w,y+h_down)]


        return focus_box
    

    def set_center(self, center, angle_x,angle_y):

        #degree to radians
        r_x = angle_x # * math.pi / 180
        r_y = angle_y #* math.pi / 180

        x = math.tan(r_y) * self.real_distance
        y = math.tan(r_x) * self.real_distance

        new_x =  round(640-center[0] + x)
        new_y =  round(center[1] - y)
      
        self.center = [new_x, new_y]
        



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

    def box_position(self,box,angle_x,angle_y,angle_z):
        
        sum_x=0
        sum_y=0
      
        #degree to radians
        r_x = (angle_x*1.3) * math.pi / 180
        r_y = (angle_y*1.3) * math.pi / 180
        r_z = angle_z * math.pi / 180

        for x,y in box[:]:
            sum_x += x  
            sum_y += y

        center=(sum_x/4,sum_y/4)
       
        a = math.tan(r_y) * self.distance
        b = math.tan(r_x) * self.distance
      
        ro_box= []
        
        for x,y in box:
            
            i,j=self.rotate(center,(x,y),r_z)
            i += a
            j -= b
            ro_box.append([int(i),int(j)])
        
        return ro_box


    def predictor(self,face_center,x,y,z,dist,pitch,yaw):
        
        self.my_distance(dist)
        self.space_img = np.full((480, 640, 3),255, dtype = np.uint8) 
        
        self.set_center(face_center,x,y)
    
        self.view_space = self.set_focusBox()
    
        focus_box = self.box_position(self.view_space,pitch,yaw,z)

        focus_iou = IOU.cul_IOU(self.ROI_space, focus_box)

        
        Cam_box = np.array(self.ROI_space,dtype=np.int64)
        cv2.polylines(self.space_img, [np.array(focus_box)], True,(0,255,0),1)
        cv2.polylines(self.space_img, [Cam_box], True,(0,0,0),1)
      

        return self.space_img, focus_iou
