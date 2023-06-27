#SECTION - 3.7 : Generalization

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd



path1 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\New folder\2_d_train.csv"
path2 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\New folder\2_d_test.csv"

train_data = pd.read_csv(path1, header=None)
validation_data = pd.read_csv(path2, header = None)

print(train_data)
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
    y = []
    for j in data [i]:
        y.append(j)  
    x = 0
    X.append(y)
 
X1 = []
for i in range(len(Y1)):
    y = []
    for j in data [i]:
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
    if c >= 1e+309:
        c = 1e+308
    mse = c / N
    return mse

def gradient(W,X,Y,N):
    X1 = np.dot(X.transpose(), X)
    X2 = np.dot(X1, W)
    X3 = np.dot(X.transpose(),Y)
    grdnt = (2/N)*(X2 - X3)
    return grdnt

def gradient_descent(X, Y, learning_rate, maxit = 1000, reltol= 1e-4):
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
        #weights = np.append(weights, current_weight_vector)
        
        v = - gradient(current_weight_vector, X, Y, N)
        
        current_weight_vector = np.add(current_weight_vector, learning_rate*v)
        print(i)
        #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
    return current_weight_vector, current_cost, mse

def output(W, X, Sample_names):
    Y = np.dot(X, W)
    for i in range(len(Y)):
        print(Sample_names[i], Y[i][0])


def mean_absolute_error(X,W,Y):
    sum = 0
    N = len(X)
    for i in range(N):
        y = np.dot(X[i], W)
        diff = abs(y - Y[i])
        sum += diff [0]
    return (sum/N)

 
current_weight_vector, current_cost, mse = gradient_descent(X, Y, 0.0001)
#print(current_cost)
print(mean_squared_error(current_weight_vector, X, Y, len(X)))
print(mean_squared_error(current_weight_vector, X1, Y1, len(X1)))
#print(mean_absolute_error(X1, current_weight_vector, Y1))

