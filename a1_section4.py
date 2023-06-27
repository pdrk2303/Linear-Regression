
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""path = r"D:\priya iitd\COL 341\assignments\prgramming assignment\train.csv"

df = pd.read_csv(path, header=None)
Y = []
Y_a = []
for i in df[1]:
    Y.append(i)
    Y_a.append([i])
data = df.T
x = 0

X = []
for i in range(144):
    y = []
    for j in data [i]:
        x+=1
        if(x>=3):
            y.append(j)
    x = 0
    X.append(y)

X = np.array(X, dtype = float)
Y = np.array(Y, dtype = float)
Y_a = np.array(Y_a, dtype = float)"""

path1 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\train.csv"
path2 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\validation.csv"


train_data = pd.read_csv(path1, header=None)
validation_data = pd.read_csv(path2, header = None)
Y = []
Y_a = []
for i in train_data[1]:
    Y.append(i)
    Y_a.append([i])
Y1=[]
Y1_a = []
for i in validation_data[1]:
    Y1.append(i)
    Y1_a.append([i])

data1 = train_data.T
x = 0

X = []
for i in range(len(Y)):
    y = [1]
    for j in data1 [i]:
        x+=1 
        if(x>=3):
            y.append(j)  
    x = 0
    X.append(y)
data2 = validation_data.T
X1 = []
for i in range(len(Y1)):
    y = [1]
    for j in data2 [i]:
        x+=1 
        if(x>=3):
            y.append(j)  
    x = 0
    X1.append(y)

X = np.array(X, dtype = float)
X1 = np.array(X1, dtype = float)
Y = np.array(Y, dtype = float)
Y1 = np.array(Y1, dtype = float)
Y_a = np.array(Y_a, dtype = float)
Y1_a = np.array(Y1_a, dtype = float)

def select_features(X, Y):
	fs = SelectKBest(score_func=f_regression, k=10)
	fs.fit(X, Y)
	X_fs = fs.transform(X)
	return X_fs, fs



def mean_squared_error(W,X,Y,N):
    Y_predicted = np.dot(X,W)
    diff = Y_predicted - Y
    x = np.dot(diff.transpose(),diff) 
    mse = np.dot(diff.transpose(),diff) [0,0]/ N
    return mse

def gradient(W,X,Y,N):
    X1 = np.dot(X.transpose(), X)
    X2 = np.dot(X1, W)
    X3 = np.dot(X.transpose(),Y)
    grdnt = (2/N)*(X2 - X3)
    return grdnt

def gradient_descent(X, Y, learning_rate, maxit = 1000, reltol= 1e-3):
    d = len(X[0])
    N = len(X)
    current_weight_vector = np.array([[0]]*d)
    mse = np.array([])
    weights = np.array([])
    previous_cost = 0
    
    
    for i in range(maxit):
        
        current_cost = mean_squared_error(current_weight_vector, X, Y, N)
        
        if previous_cost and abs(previous_cost-current_cost)<= reltol:
            break
        
        previous_cost = current_cost
        mse = np.append(mse,current_cost)
        
        v = - gradient(current_weight_vector, X, Y, N)
        
        current_weight_vector = np.add(current_weight_vector, learning_rate*v)
        print(i)
    return current_weight_vector, current_cost, mse

def mean_absolute_error(X,W,Y):
    sum = 0
    N = len(X)
    for i in range(N):
        y = np.dot(X[i], W)
        diff = abs(y - Y[i])
        sum += diff [0]
    return (sum/N)

X_fs, fs = select_features(X1,Y1)
current_weight_vector, current_cost, mse = gradient_descent(X_fs, Y1_a, 0.001)
print(current_cost)
print(mean_absolute_error(X_fs, current_weight_vector, Y1_a))







