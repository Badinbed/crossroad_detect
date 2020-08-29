import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


def CCDC(image_open,point):
    img_edge2=image_open
    x=int(point[0])
    y=int(point[1])
    num_4=0
    dis_min=15
    dis_max=40
    dis=dis_min
    """cv2.line(img,(x-dis_min,y-dis_min),(x+dis_min,y-dis_min),(0,0,255),2)
    cv2.line(img,(x+dis_min,y-dis_min),(x+dis_min,y+dis_min),(0,0,255),2)
    cv2.line(img,(x+dis_min,y+dis_min),(x-dis_min,y+dis_min),(0,0,255),2)
    cv2.line(img,(x-dis_min,y+dis_min),(x-dis_min,y-dis_min),(0,0,255),2)
    cv2.line(img,(x-dis_max,y-dis_max),(x+dis_max,y-dis_max),(0,255,255),2)
    cv2.line(img,(x+dis_max,y-dis_max),(x+dis_max,y+dis_max),(0,255,255),2)
    cv2.line(img,(x+dis_max,y+dis_max),(x-dis_max,y+dis_max),(0,255,255),2)
    cv2.line(img,(x-dis_max,y+dis_max),(x-dis_max,y-dis_max),(0,255,255),2)


    cv2.imshow('1',img)"""
    var_total=0
    delta=100
    while dis<dis_max:
        s1=img_edge2[y-dis,x-dis:x+dis]
        #print(s1)
        s2=img_edge2[y-dis+1:y+dis,x+dis]

        s3=img_edge2[y+dis,x-dis:x+dis-1]
        s3=s3[::-1]
        s4=img_edge2[y-dis+1:y+dis-1,x-dis]
        s4=s4[::-1]
        last=img_thresh[y-dis,x-dis]
        num=0
        lenth=[]
        length=0
        lenth_var=0
        for a in s1:
            if a==255:
                if last==0:
                    length=1
                else :
                    length=length+1
            else :
                if last==255:
                    if length>5:
                        num=num+1
                        lenth.append(length)
            last=a
        for a in s2:
            if a==255:
                if last==0:

                    length=1
                else :
                    length=length+1
            else :
                if last==255:
                    if length>5:
                        num=num+1
                        lenth.append(length)
            last=a
        for a in s3:
            if a==255:
                if last==0:

                    length=1
                else :
                    length=length+1
            else :
                if last==255:
                    if length>5:
                        num=num+1
                        lenth.append(length)
            last=a
        for a in s4:
            if a==255:
                if last==0:

                    length=1
                else :
                    length=length+1
            else :
                if last==255:
                    if length>5:
                        num=num+1
                        lenth.append(length)
            last=a
        #print(lenth)
        if num==4:
            lenth_var=np.var(lenth)
            #print(lenth_var)
            num_4=num_4+1
        dis=dis+1
        var_total=var_total+lenth_var
    if num_4>0:
        delta=var_total/num_4
    #print(delta)
    #print(num_4)
    if (num_4/(dis_max-dis_min)>0.8) & (delta<5):
        print([x,y],'is a crossroad')
        return 1
    else:
        return 0

def hierarchy_cluster(data,threshold, method='average'):
    '''层次聚类
    
    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- 文档 i 和文档 j 的距离
    
    Keyword Arguments:
        method {str} -- [linkage的方式： single、complete、average、centroid、median、ward] (default: {'average'})
        threshold {float} -- 聚类簇之间的距离
    Return:
        cluster_number int -- 聚类个数
        cluster [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    data = np.array(data)
 
    Z = linkage(data, method='average')
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    #print type(cluster_assignments)
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)
 
    return num_clusters, indices
 
 
 
def get_cluster_indices(cluster_assignments):
    '''映射每一类至原数据索引
    
    Arguments:
        cluster_assignments 层次聚类后的结果
    
    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    
    return indices



def cross_point(line1, line2):  # 计算交点函数
    #是否存在交点
    point_is_exist=False
    x=0
    y=0
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            #point_is_exist=True
    elif k2 is None:
        x=x3
        y=k1*x3+b1
    elif not k2==k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    if (x>0)&(y>0) :
        point_is_exist=True
    return point_is_exist,[x, y]


    


img = cv2.imread("3.jpg", 1)
#sp=img.shape
#print(sp)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("src", gray)

dst = cv2.equalizeHist(gray)
#cv2.imshow("dst", dst)

ret, img_thresh = cv2.threshold(dst, 203, 255, cv2.THRESH_BINARY )
#cv2.imwrite("img_thresh.jpg",img_thresh)

Matrix = np.ones((3, 3), np.uint8)
    
img_edge1 = cv2.erode(img_thresh, Matrix)
#cv2.imshow('erode.jpg',img_edge1 )

Matrix2 = np.ones((7, 7), np.uint8)
    
img_edge2 = cv2.dilate(img_edge1, Matrix2)
#cv2.imwrite('dilate.jpg',img_edge2)

edges = cv2.Canny(img_edge2,100,200,apertureSize = 7)
#cv2.imwrite('edges.jpg',edges)
"""
lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=70,maxLineGap=80)
lines1 = lines[:,0,:]#提取为二维
print(len(lines1))
for x1,y1,x2,y2 in lines1[:]: 
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    #cv2.line(edges,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('line1.jpg',img)
#cv2.imshow('line2.jpg',edges)
"""

lines = cv2.HoughLines(edges,1,np.pi/180,100)
lines2 = lines[:,0,:]#提取为为二维
print(len(lines2))
lines1=[]
for rho,theta in lines2[:]: 
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    lines1.append([x1,y1,x2,y2])
    #cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#cv2.imwrite('line2.jpg',img)


points=[]
i=0
while i<len(lines1):
    a=lines1[i]
    #print(a)
    i=i+1
    j=i
    while j<len(lines1):
        b=lines1[j]
        j=j+1

        point_is_exist, [x, y]=cross_point(a,b)
        if point_is_exist:
                print([x,y])
                points.append([x,y])
                #cv2.circle(img,(int(x),int(y)),5,(255,0,0),2)

print(len(points))            
#cv2.imwrite('points.jpg', img)
"""
for i,line in lines1:
    for j,line0 in lines1:
        if j>i:

            point_is_exist, [x, y]=cross_point(line,line0)
            if point_is_exist:
                print([x,y])
                points.append([x,y])
                cv2.circle(img,(int(x),int(y)),5,(255,0,0),2)

print(len(points))            
cv2.imshow('points.jpg', img)"""

arr = np.array(points)

num_clusters, indices = hierarchy_cluster(arr,30)
 
 
print ("%d clusters" % num_clusters)
ave=[]      
for k, ind in enumerate(indices):
    
    sumx=0
    sumy=0
    n=0
    print ("cluster", k + 1, "is", ind)
    if len(ind)>2:
        for i in ind:
            
            print(points[i])
            x,y=points[i]
            sumx=sumx+x
            sumy=sumy+y
            n=n+1
        ave.append([sumx/n,sumy/n])
print ('center point is',ave)

for point in ave:
    #print(point)
    i=CCDC(img_edge2,point)
    if i==1:
        dis_min=20
        x=int(point[0])
        y=int(point[1])
        cv2.line(img,(x-dis_min,y-dis_min),(x+dis_min,y-dis_min),(255,0,255),2)
        cv2.line(img,(x+dis_min,y-dis_min),(x+dis_min,y+dis_min),(255,0,255),2)
        cv2.line(img,(x+dis_min,y+dis_min),(x-dis_min,y+dis_min),(255,0,255),2)
        cv2.line(img,(x-dis_min,y+dis_min),(x-dis_min,y-dis_min),(255,0,255),2)
    #print(i)
cv2.imwrite('3_result.jpg',img)
