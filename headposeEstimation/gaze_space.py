from typing import ChainMap
import numpy as np
import cv2 as cv
import math 






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



def box_rotation(box,angle):
    
    sum_x=0
    sum_y=0

    for x,y in box[:]:
        sum_x += x  
        sum_y += y

    center=(sum_x/4,sum_y/4)
   

    ro_box= []
    
    for x,y in box:
        
        i,j=rotate(center,(x,y),angle)
        ro_box.append([int(i),int(j)])
        
    return ro_box

def box_move(box,angle):
    sum_x=0
    sum_y=0

    for x,y in box[:]:
        sum_x += x  
        sum_y += y

    center=(sum_x/4,sum_y/4)

    
    len = 50
    a = math.tan(angle) * len
    
    print (a)
    mv_box= []
    for x,y in box:
        i = x + a;
        print(x)
        print(int(i)) 
        mv_box.append([int(i),int(y)])

    print(mv_box)
    return mv_box


space_img = np.full((640, 640, 3),255, dtype = np.uint8) 

w = 640
h = 640
center = [w/2,h/2]
Cam_w = 36/2
Cam_h = 24/2

distance = 50

Cam_space = [(w/2-Cam_w, h/2+Cam_h),(w/2-Cam_w, h/2-Cam_h),(w/2+Cam_w, h/2 -Cam_h),(w/2+Cam_w, h/2+Cam_h)]

focus_space = set_focusBox(distance,center)

Roi_space = set_veiwBox(distance, center)

ro_box = box_rotation(Roi_space,math.radians(0)) 
ro_box_2 = box_rotation(focus_space,math.radians(0)) 

mv_box = box_move(ro_box,math.radians(0))
mv_box_2 = box_move(ro_box_2,math.radians(0))

cv.polylines(space_img, [np.array(Cam_space, dtype=np.int64)], True,(255,0,0),1)
#cv.polylines(space_img, [np.array(Roi_space, dtype = np.int64)], True,(0,255,0),1)
cv.polylines(space_img, [np.array(mv_box,dtype = np.int64)], True,(0,0,255),1)
#cv.polylines(space_img, [np.array(focus_space, dtype = np.int64)], True,(0,255,0),1)
#cv.polylines(space_img, [np.array(mv_box_2,dtype = np.int64)], True,(0,0,255),1)


cv.imshow("test", space_img)
key = cv.waitKey(0)