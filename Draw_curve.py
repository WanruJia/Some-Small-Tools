import pandas as pd
import numpy as np
from numpy import linalg as LA
import cv2
import xlwt
import csv

new = pd.read_excel ("20200516/CurveNew.xlsx", header=None)
new = new.as_matrix()
print(len(new))

outputImage = cv2.imread("C:/Dalhousie Master's Courses/Medical Image Analysis/Xray Grades/program for X-ray images/APJointCentre/1-1R AP.jpg",1)
outputImage = cv2.resize(outputImage,(1200,1200))
for i in range(int(len(new)-1)):
    image_new = cv2.line(outputImage,tuple((new[i]*10).astype(int)),tuple((new[i+1]*10).astype(int)),(0, 255, 0), thickness=2)

cv2.imshow('Curve', image_new)
cv2.waitKey(0)
cv2.imwrite('20200607/Step 1 (39,58) Tunning 20 superimpose oringin new.jpg',image_new)

outputImage = cv2.imread("C:/Dalhousie Master's Courses/Medical Image Analysis/Xray Grades/program for X-ray images/20200502/twobytwo eigenvalue.jpg",1)
outputImage = cv2.resize(outputImage,(1200,1200))
for i in range(int(len(new)-1)):
    image_new = cv2.line(outputImage,tuple((new[i]*10).astype(int)),tuple((new[i+1]*10).astype(int)),(0, 0, 255), thickness=2)

cv2.imshow('Curve', image_new)
cv2.waitKey(0)
cv2.imwrite('20200607/Step 1 (39,58) Tunning 2 superimpose vectorplots new.jpg',image_new)
