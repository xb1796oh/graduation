import numpy
import copy
import ctypes


def MIN (a,b):
    return a if a<b else b

def MAX (a,b):
    return a if a >b else b


def dist(p1, p2):
    d = (p1[0] - p2[0])*(p2[0] -p1[0]) + (p1[1] - p2[1])*(p2[1] -p1[1])
    return d 

def ccw(p1, p2, p3):
    
    cross_product = (p2[0] - p1[0])*(p3[1] -p1[1]) - (p3[0] - p1[0])*( p2[1] - p1[1])


    if cross_product > 0 :
        return 1
    
    elif cross_product < 0:
        return -1
    
    else:
        return 0

#right가 left의 반시계 방향에 있으면 true이다.
#true if right is counterclockwise to left.
p =[0,0]

def Sortcomparator(left, right):

        global p
        direction = ccw(p, left, right)

        if direction == 0:
            ret = dist(p, left) < dist(p, right)
        
        elif direction == 1:
            ret = 1
        
        else:
            ret = 0
        
        return ret

cnt =0
def QuickSort (a, lo, hi):
    global cnt
    #print("cmt",cnt)
    cnt += 1
    #print("a", a)

    if hi - lo <= 0 :
        #print("return",hi-lo)
        return 

    pivot =[0,0] 
    pivot = a[int((lo + hi)/2)]

    i =lo
    j =hi

    while i <= j:
        while Sortcomparator(a[i], pivot): i += 1

        while Sortcomparator(pivot, a[j]): j -= 1

        if i <= j:
            a[i], a[j] = a[j], a[i]
            i += 1
            j -= 1

    QuickSort(a,lo,j)
    QuickSort(a, i, hi)
    # print("a_result ", a)




  

def LineComparator(left, right):
    
    if left.x == right.x : 
        ret = True if (left.y <= right.y) else False
    
    else :
        ret = True if (left.x <= right.x) else False
    
    return ret

def swap (p1, p2):
    p3 = p1.copy()
    p1 = p2.copy()
    p2 = p3.copy()
    return p1, p2


def LineIntersection(l1, l2):
    
    l1_l2 = ccw(l1[0], l1[1], l2[0]) * ccw(l1[0], l1[1], l2[1])
   
    l2_l1 = ccw(l2[0], l2[1], l1[0]) * ccw(l2[0], l2[1], l1[1])

    ret = (l1_l2 < 0) and (l2_l1 < 0)

    return ret

def PolygonInOut( p, num_vertex,  vertices):

    ret = 0

    #마지막 꼭지점과 첫번째 꼭지점이 연결되어 있지 않다면 오류를 반환한다.
    #If the last vertex and the first vertex are not connected, an error is returned.
    if (vertices[0][0] != vertices[num_vertex][0] or vertices[0][1] != vertices[num_vertex][1]) :
        print("Last vertex and first vertex are not connected.");
        return -1;
    

  
    for i in range(0, num_vertex) :
        
        # 점 p가 i번째 꼭지점과 i+1번째 꼭지점을 이은 선분 위에 있는 경우
        # If point p is on the line connecting the i and i + 1 vertices
        if( ccw(vertices[i], vertices[i+1], p) == 0 ):
            min_x = MIN(vertices[i][0], vertices[i+1][0])
            max_x = MAX(vertices[i][0], vertices[i+1][0])
            min_y = MIN(vertices[i][1], vertices[i+1][1])
            max_y = MAX(vertices[i][1], vertices[i+1][1])

            # 점 p가 선분 내부의 범위에 있는 지 확인
            # Determine if point p is in range within line segment
            if(min_x <= p[0] and p[0] <= max_x and min_y <= p[1] and p[1] <= max_y):
                return 1
            
      
    

    #다각형 외부에 임의의 점과 점 p를 연결한 선분을 만든다.
    # Create a line segment connecting a random point outside the polygon and point p.
    outside_point= []
    outside_point.append(1)
    outside_point.append(1234567)
    l1 = [[0 for j in range(2)] for i in range(2)]
    l1[0] = outside_point
    l1[1] = p

    # 앞에서 만든 선분과 다각형을 이루는 선분들이 교차되는 갯수가 센다.
    # Count the number of intersections between the previously created line segments and the polygonal segments.
 
    for i in range (0,num_vertex) :
        l2 = [[0 for j in range(2)] for i in range(2)]
        l2[0] = vertices[i]
        l2[1] = vertices[i+1]
        ret += LineIntersection(l1, l2)
        
    

    #교차한 갯수가 짝수인지 홀수인지 확인한다.
    #Check if the number of crossings is even or odd.
    ret = ret % 2;
    return ret;

# (p1, p2)를 이은 직선과 (p3, p4)를 이은 직선의 교차점을 구하는 함수
# Function to get intersection point with line connecting points (p1, p2) and another line (p3, p4).
def IntersectionPoint( p1, p2,  p3, p4):

    ret = []

    a = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])) / ( (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) )

    b = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1]- p4[1]) - (p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0]- p4[0]))

    ret.append(a)
    ret.append(b)
    return ret


# 다각형의 넓이를 구한다.
# find the area of a polygon
def GetPolygonArea( num_points, points):
        ret = 0
        #print(points)
        i = num_points - 1
        for j in range(0, num_points) :
            ret += points[i][0] * points[j][1] - points[j][0] * points[i][1];
            #print("ret: ",ret)
            i = j;
            
        ret =   -ret if ret < 0 else ret
        ret /= 2

        return ret
 
interection_num =0
intersection_point = [0 for i in range(10)]

def GetIntersection( num1, points1,num2, points2):
    global interection_num 
    global intersection_point
    global p

    ret =0
    l1 = [[0 for j in range(2)] for i in range(2)]
    l2 = [[0 for j in range(2)] for i in range(2)]

    # points1과 points2 각각을 반시계방향으로 정렬한다.
    # sort by counter clockwise point1 and points2.
    p = points1[0]
    QuickSort(points1, 1, num1-1)
    #print(points1)

    p = points2[0]
    QuickSort(points2,1, num2-1)
    #print(points2)

    #차례대로 점들을 이었을 때, 다각형이 만들어질 수 있도록 시작점을 마지막에 추가한다.
    #Add the starting point to the last in order to make polygon when connect points in order.
    points1.append(points1[0])
    points2.append(points2[0])

    #points1의 다각형 선분들과 points2의 다각형 선분들의 교차점을 구한다.
    # Find the intersection of the polygon segments of points1 and the polygon segments of points2.
    
    for i in range(0,num1):
        l1[0] = points1[i];
        l1[1] = points1[i+1];
        
        for  j in range(0, num2) :
            l2[0] = points2[j];
            l2[1]= points2[j+1];

            # 선분 l1과 l2가 교차한다면 교차점을 intersection_point에 저장한다.
            # If line segments l1 and l2 intersect, store the intersection at intersection_point.
            if LineIntersection(l1, l2):
                intersection_point[interection_num] = IntersectionPoint(l1[0], l1[1], l2[0], l2[1])
                interection_num +=1
        
   
    for i in range(0,num1):
        if(PolygonInOut(points1[i], num2, points2)):
           
            intersection_point[interection_num] = points1[i]
            interection_num +=1
        
  
    for i in range(0,num2) :
        if(PolygonInOut(points2[i], num1, points1)):
            intersection_point[interection_num] = points2[i]
            interection_num += 1
        
    
    p = intersection_point[0]
    QuickSort(intersection_point,1,interection_num-1)
    #print(intersection_point)

    ret = GetPolygonArea(interection_num, intersection_point) 
    
    points1 = list(points1)
    points2 = list(points2)
    # restore
    del points1[num1] 
    del points2[num2] 

    return ret


def GetIoU(num1,  points1,  num2,  points2):
    global interection_num
    global intersection_point
    ret = 0 
    interection_num =0
    intersection_point = [0 for i in range(10)]
    
    A = GetPolygonArea(num1, points1)
    B = GetPolygonArea(num2, points2)
    intersection_area = GetIntersection(num1, points1, num2, points2)
    #print("A: ",A)
    #print("B: ",B)
    #print("intersection_Area :",intersection_area)

    #union_area = A + B - intersection_area
    union_area = A
    ret = intersection_area / union_area       
    
    return ret

"""
if __name__ == '__main__':
    
    
    num1 = 4;
    points1  = [[0 for j in range(2)] for i in range(num1)]
    points1[0][0] = 3
    points1[0][1] = 4
    points1[1][0] = 4
    points1[1][1] = 4
    points1[2][0] = 4
    points1[2][1] = 2
    points1[3][0] = 3
    points1[3][1] = 2

    num2 = 4
    points2 = [[0 for j in range(2)] for i in range(num2)]
    points2[0][0] = 2
    points2[0][1] = 3
    points2[1][0] = 3
    points2[1][1] = 2
    points2[2][0] = 5
    points2[2][1] = 3
    points2[3][0] = 5
    points2[3][1] = 4

    print(points1)
    print(points2)
    print("IoU : ", GetIoU(num1, points1, num2, points2))
    """
    
def cul_IOU(box1, box2):

    num1 = len(box1)
    points1 =[[0 for j in range(2)] for i in range(num1)]
    points1  = box1.copy()

    num2 = len(box2)
    points2 = [[0 for j in range(2)] for i in range(num2)]
    points2 = box2.copy()
    result = GetIoU(num1, points1, num2, points2)
    print("IOU: ", result)
    return result





