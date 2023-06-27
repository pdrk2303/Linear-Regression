
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

def section_1(train_path, val_path, test_path, out_path):
    path1 = train_path
    path2 = val_path
    path3 = test_path
    outpath = out_path

    train_data = pd.read_csv(path1, header=None)
    validation_data = pd.read_csv(path2, header = None)
    test_data = pd.read_csv(path3, header = None)
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
    Sample_names = np.array(test_data[0])
    data3 = test_data.T
    
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
    X_test = np.array(X_test, dtype = float)

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

    def gradient_descent(X, Y, learning_rate, maxit = 1000, reltol= 1e-3):
        d = len(X[0])
        N = len(X)
        current_weight_vector = np.array([[0]]*d)
        mse = np.array([])
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
            #print(i)
            #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
        return current_weight_vector, current_cost, mse

    def output(W, X, Sample_names):
        Y = np.dot(X, W)
        out = []
        for i in range(len(Y)):
            out.append([Sample_names[i], Y[i][0]])
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(out)
        

    def mean_absolute_error(X,W,Y):
        sum = 0
        N = len(X)
        for i in range(N):
            y = np.dot(X[i], W)
            diff = abs(y - Y[i])
            sum += diff [0]
        return (sum/N)
    current_weight_vector, current_cost, mse = gradient_descent(X, Y, 0.001)
    output(current_weight_vector, X_test, Sample_names)
    
def section_2(train_path, val_path, test_path, out_path):
    path1 = train_path
    path2 = val_path
    path3 = test_path
    outpath = out_path

    train_data = pd.read_csv(path1, header=None)
    validation_data = pd.read_csv(path2, header = None)
    test_data = pd.read_csv(path3, header = None)
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
    Sample_names = np.array(test_data[0])
    data3 = test_data.T
    
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
    X_test = np.array(X_test, dtype = float)



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
            #print(i)
            #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
        return current_weight_vector, current_cost, mse

    def output(W, X, Sample_names):
        Y = np.dot(X, W)
        out = []
        for i in range(len(Y)):
            out.append([Sample_names[i], Y[i][0]])
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(out)
        

    def mean_absolute_error(X,W,Y):
        sum = 0
        N = len(X)
        for i in range(N):
            y = np.dot(X[i], W)
            diff = abs(y - Y[i])
            sum += diff [0]
        return (sum/N)
    current_weight_vector, current_cost, mse = gradient_descent(X, Y, 0.001, 5)
    output(current_weight_vector, X_test, Sample_names)
        
def section_5(train_path, val_path, test_path, out_path):
    path1 = train_path
    path2 = val_path
    path3 = test_path
    outpath = out_path

    train_data = pd.read_csv(path1, header=None)
    validation_data = pd.read_csv(path2, header = None)
    test_data = pd.read_csv(path3, header = None)
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
    Sample_names = np.array(test_data[0])
    data3 = test_data.T
    
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
    X_test = np.array(X_test, dtype = float)


    def cost_function(W, X, Y, N):
        output = 0
        
        for i in range(N):
            sum = 1
            for j in range(0,8):
                p = np.dot(W[j], X[i])
                e_ = math.exp(p)
                sum+= e_
            if Y[i][0] != 9:
                W_k = W[int(Y[i][0]-1)]
                power = np.dot(W_k, X[i])
                e = math.exp(power)
                if(e != 0):           
                    output += math.log(e/sum)
            else:
                output += math.log(1/sum)
        return -(output/N)


    def gradient(W, X, Y, N):
        out = []
        for r in range(1,9):
            output = np.array([0]*len(X[0]))
            for i in range(N):
                if Y[i][0] != 9:
                    W_k = W[int(Y[i][0] - 1)]
                    power = np.dot(W_k, X[i])
                    e = math.exp(power)
                    sum = 1
                    for j in range(0,8):
                        p = np.dot(W[j], X[i])
                        e_ = math.exp(p)
                        sum+= e_
                    if r == Y[i][0]:
                        z = (1 - (e/sum))
                    else:
                        z = (e/sum)
                    output = np.add(output, z*X[i])
                else:
                    output = np.add(output, np.array([0]*len(X[0])))
            out.append(output)
        return (1/N)*np.array(out)
            

    def gradient_descent(X, Y, learning_rate, maxit = 1000, reltol= 1e-4):
        d = len(X[0])
        N = len(X)
        current_weight_vector = np.array([[0]*d]*8)
        mse = np.array([])
        previous_cost = 0
        
        
        for i in range(maxit):
            
            current_cost = cost_function(current_weight_vector, X, Y, N)
            
            if previous_cost and abs(previous_cost-current_cost)<= reltol:
                break
            
            previous_cost = current_cost
            mse = np.append(mse,current_cost)
            #weights = np.append(weights, current_weight_vector)
            
            v = - gradient(current_weight_vector, X, Y, N)
            
            current_weight_vector = np.add(current_weight_vector, learning_rate*v)
            #print(i)
            #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
        return current_weight_vector, current_cost, mse 
    def output(W, X, Sample_names):
        Y = np.dot(X, W)
        out = []
        for i in range(len(Y)):
            out.append([Sample_names[i], Y[i][0]])
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(out)
        

    def mean_absolute_error(X,W,Y):
        sum = 0
        N = len(X)
        for i in range(N):
            y = np.dot(X[i], W)
            diff = abs(y - Y[i])
            sum += diff [0]
        return (sum/N)
    current_weight_vector, current_cost, mse = gradient_descent(X, Y, 0.001)
    output(current_weight_vector, X_test, Sample_names)
    

def main(train_path, val_path, test_path, out_path, section):
    
    if section == 1:
        section_1(train_path, val_path, test_path, out_path)
    elif section == 2:
        section_2(train_path, val_path, test_path, out_path)
    elif section == 5:
        section_5(train_path, val_path, test_path, out_path)
        

    
    