#SECTION 3.2 - RIDGE REGRESSION

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd



path1 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\train.csv"
path2 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\validation.csv"
path3 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\test.csv-20230211T055604Z-001\test.csv"
#print(path1)
train_data = pd.read_csv(path1, header=None)
validation_data = pd.read_csv(path2, header = None)
test_data = pd.read_csv(path3, header=None)
Y = []
for i in train_data[1]:
    Y.append([i])
Y1=[]
for i in validation_data[1]:
    Y1.append([i])

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

data2 =  validation_data.T

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
Sample_names = np.array(test_data[0])
data3 = test_data.T
X_test = []
for i in range(len(test_data[0])):
    y = [1]
    for j in data3 [i]:
        x+=1 
        if(x>=2):
            y.append(j)  
    x = 0
    X_test.append(y)
X_test = np.array(X_test, float)



def mean_squared_error(W,X,Y,N,param):
    Y_predicted = np.dot(X,W)
    diff = Y_predicted - Y
    bias = param * (np.dot(W.transpose(),W) [0,0])
    c = np.dot(diff.transpose(),diff) [0,0]
    if c >= 1e+309:
        c = 1e+308
    if bias >= 1e+309:
        bias = 1e+308
    cost = c + bias
    mse = cost/ N
    return mse

def gradient(W,X,Y,N,param):
    X1 = np.dot(X.transpose(), X)
    X2 = np.dot(X1, W)
    X3 = np.dot(X.transpose(),Y)
    bias = param*W
    g = (X2 - X3 + bias)
    grdnt = (2/N)*(g)
    return grdnt

def gradient_descent(X, Y, learning_rate, param, maxit = 1000, reltol= 1e-3):
    d = X [0].size
    N = len(X)
    current_weight_vector = np.array([[0]]*d)
    mse = np.array([0])
    previous_cost = 0
    
    
    for i in range(maxit):
        
        current_cost = mean_squared_error(current_weight_vector, X, Y, N, param)
        
        if previous_cost and abs(previous_cost-current_cost)<= reltol:
            break
        
        previous_cost = current_cost
        mse = np.append(mse,current_cost)
        #weights = np.append(weights, current_weight_vector)
        
        v = - gradient(current_weight_vector, X, Y, N, param)
        
        current_weight_vector = np.add(current_weight_vector, learning_rate*v)
        print(i)
        #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
    return current_weight_vector, current_cost, mse

def mean_absolute_error(X,W,Y):
    sum = 0
    N = len(X)
    for i in range(N):
        y = np.dot(X[i], W)
        diff = abs(y - Y[i])
        sum += diff [0]
    return (sum/N)














