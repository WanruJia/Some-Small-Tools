import cv2
import xlwt
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#Saving Brightness
image = cv2.imread("C:/DalMasterCourses/Medical Image Analysis/Xray Grades/program for X-ray images/APJointCentre/1-1R AP.jpg",1)
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
df=pd.DataFrame(image)
B = np.array(df).transpose()
#print(B)

#Kernel Calculation
k = 9

x = []
y = []

for val in range(k):
    x.append([[-((k-1)/2-val)]*k])
    y.append([[-((k-1)/2-val)]*k])
x = np.array(x)
y = np.array(y).transpose()
#print(y)

X = np.zeros((6,k*k))
#print(x[0]**2)

for val in range(k):
    X[0][((val)*k):((val+1)*k)]=(x[val])**2
    X[1][((val)*k):((val+1)*k)]=(x[val])*y[val]
    X[2][((val)*k):((val+1)*k)]=(y[val])**2
    X[3][((val)*k):((val+1)*k)]=(x[val])
    X[4][((val)*k):((val+1)*k)]=(y[val])
    X[5][((val)*k):((val+1)*k)]=1
#print(X)
X = X.transpose()
Kernel = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
#print(Kernel)

#EigenValues/Vectors Calculation
#print(df.shape[0])
Value1 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))
Value2 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))

Vector1 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,2))
Vector2 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,2))

H = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,4))
#print(Vector2)

D = np.zeros((2,2))
for i in range(df.shape[0]-k+1):
    for j in range(df.shape[1]-k+1):
        Y = []
        for h in range(k):
            Y += B[j+h][i:(i+k)].tolist()
        Beta = np.dot(Kernel, Y)
        #print(Beta)
        D[0][0]=Beta[0]
        D[0][1]=Beta[1]/2
        D[1][0]=Beta[1]/2
        D[1][1]=Beta[2]
        Ev = np.linalg.eig(np.array(D))
        #print(Ev)
        if Ev[0][1] > Ev[0][0]:
            Value1[i][j] = Ev[0][1]
            Value2[i][j] = Ev[0][0]
            Vector1[i][j] = Ev[1][1]
            Vector2[i][j] = Ev[1][0]
        else:
            Value1[i][j] = Ev[0][0]
            Value2[i][j] = Ev[0][1]
            Vector1[i][j] = Ev[1][0]
            Vector2[i][j] = Ev[1][1]
        H[i][j] = D.flatten()
Diff = Value1 - Value2
#print(Y)
#print(Beta)
value1_new=pd.DataFrame(Value1)
value1_new.to_csv('20201018/Value1.csv',index=False,header=False)
value2_new=pd.DataFrame(Value2)
value2_new.to_csv('20201018/Value2.csv',index=False,header=False)
#vector1_new=pd.DataFrame(Vector1.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector1_new.to_csv('20200502/Vector1.csv',index=False,header=False)
#vector2_new=pd.DataFrame(Vector2.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector2_new.to_csv('20200502/Vector2.csv',index=False,header=False)

plt.figure(num=None, figsize=(120, 120), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes()
times = 10
ax.set_xlim(120)
ax.set_ylim(120)
for i in range(120):
    for j in range(120):
        if i%2 == 0 and j%2 == 0:
            ax.annotate("", xy=(i-Vector2[i][j][1]*Value2[i][j]*times, j-Vector2[i][j][0]*Value2[i][j]*times), xytext=(i, j),arrowprops=dict(arrowstyle="->"))
            #ax.annotate("", xy=(i-Vector1[i][j][1]*times, j-Vector1[i][j][0]*times), xytext=(i, j),arrowprops=dict(arrowstyle="->"))
#plt.arrow(0, 0, 0.5, 0.5)

plt.show()
