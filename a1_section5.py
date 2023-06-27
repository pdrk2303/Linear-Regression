#SECTION 3.5 - Classification
import math
import numpy as np
import pandas as pd

path1 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\train.csv"
path2 = r"D:\priya iitd\COL 341\assignments\prgramming assignment\validation.csv"
#df = pd.read_csv(path)

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

#print(X)
#print(Y)


def cost_function(W, X, Y, N):
    #W_k = W [y]
    output = 0
    
    for i in range(N):
        sum = 1
        for j in range(0,8):
            a = W[j]
            b = X[i]
            p = np.dot(W[j], X[i])
            e_ = math.exp(p)
            sum+= e_
        if Y[i][0] != 9:
            W_k = W[int(Y[i][0]-1)]
            c = X[i]
            d = np.dot(W_k, X[i])
            power = np.dot(W_k, X[i])
            #print("power:", power)
            e = math.exp(power)
            """print(e/sum)
            print("e:", e)
            print('sum:', sum)"""
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
    weights = np.array([])
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
        """W = []
        for i in range(0,4):
            current_weight_vector[i] = np.add(current_weight_vector[i], learning_rate*v[i])
            W.append(current_weight_vector[i])
        current_weight_vector = np.array(W)"""
        print(i)
        #print(f"Iteration {i+1}: Cost {current_cost}, Weight {current_weight_vector}")
    return current_weight_vector, current_cost, mse 


"""X = np.array([[1,2,2,1,3,2],
              [1,2,2,2,2,2],
              [3,4,2,1,2,3],
              [2,3,4,1,2,3],
              [1,2,3,3,4,3],
              [3,2,1,4,3,2],
              [4,3,2,3,4,1]])
Y = np.array([[1],
              [2],
              [5],
              [3],
              [4],
              [4],[3]])"""

current_weight_vector, current_cost, mse = gradient_descent(X, Y, 0.00001)

print(current_weight_vector, current_cost)

























