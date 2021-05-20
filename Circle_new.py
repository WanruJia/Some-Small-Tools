import cv2
import numpy as np
#from pyimagesearch import datasets
#from pyimagesearch import models
import os
import argparse
import glob


img = cv2.imread("C:/DalMasterCourses/Softwares for Hongyang's Group/CircleDetection/Circle.jpg", 0)
print(img)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,50, param1=40,param2=30,minRadius=200,maxRadius=250)
print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imwrite("C:/DalMasterCourses/Softwares for Hongyang's Group/CircleDetection/Circlefitting origin p1 40 p2 30 200 250.jpg",cimg)
cv2.destroyAllWindows()
