#SECTION 3.3 - SCIKIT

import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd


path1 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\train.csv"
path2 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\validation.csv"


train_data = pd.read_csv(path1, header=None)
validation_data = pd.read_csv(path2, header = None)
Y = []
for i in train_data[1]:
    Y.append([i])
Y1=[]
for i in validation_data[1]:
    Y1.append([i])

data = train_data.T
x = 0

X = []
for i in range(len(Y)):
    y = [1]
    for j in data [i]:
        x+=1 
        if(x>=3):
            y.append(j)  
    x = 0
    X.append(y)
 
X1 = []
for i in range(len(Y1)):
    y = [1]
    for j in data [i]:
        x+=1 
        if(x>=3):
            y.append(j)  
    x = 0
    X1.append(y)

X = np.array(X, dtype = float)
X1 = np.array(X1, dtype = float)
Y = np.array(Y, dtype = float)
Y1 = np.array(Y1, dtype = float)
Sample_names = np.array(train_data[0])

def mean_squared_error(W,X,Y,N):
    Y_predicted = np.dot(X,W)
    diff = Y_predicted - Y 
    c = np.dot(diff.transpose(),diff) [0,0]
    mse = c / N
    return mse

def mean_absolute_error(X,W,Y):
    sum = 0
    N = len(X)
    for i in range(N):
        y = np.dot(X[i], W)
        diff = abs(y - Y[i])
        sum += diff [0]
    return (sum/N)

reg = LinearRegression(fit_intercept = False).fit(X, Y)
print(reg.coef_)
#print(len(reg.coef_[0]))

W = []
for i in reg.coef_[0]:
    W.append([i])
    
W = np.array(W, dtype = float)
print(mean_squared_error(W, X1, Y1, len(X1)))
print(mean_absolute_error(X1, W, Y1))
















