# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:03:55 2017

@author: haowen
"""

import numpy as np
import matplotlib.pyplot as plt

def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    num = len(dataSet)
    
    X = np.zeros((num,3))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i,0] = 1.0
        X[i,1] = data[0]
        X[i,2] = data[1]
        Y[i,0] = np.float(data[-1])
    return X,Y
    
def LinearRegression_Regularization(x,y,LAMBDA):
    x = np.matrix(x)
    y = np.matrix(y)
    w = np.linalg.inv(x.T*x+LAMBDA*np.eye(x.shape[1])) * x.T * y
    return w
    
def sign(x,w):
    if np.dot(x,w)[0] >= 0:
        return 1.0
    else:
        return -1.0
        
def error_rate(x,y,w):
    error = 0.0
    for i in range(len(x)):
        if sign(x[i],w) != y[i,0]:
            error = error +1.0
    return error/len(x)
    
filename = r"C:\Users\haowen\Dropbox\Machine learning\Machine Learning Foundations" + "\hw4_train.dat"
X_train, Y_train = getDataSet(filename)

filename = r"C:\Users\haowen\Dropbox\Machine learning\Machine Learning Foundations" + "\hw4_test.dat"
X_test, Y_test = getDataSet(filename)
#Q13
LAMBDA = 10
w = LinearRegression_Regularization(X_train,Y_train,LAMBDA)
Ein = error_rate(X_train,Y_train,w)
Eout = error_rate(X_test,Y_test,w)
#Q14
#test_list = np.linspace(2,-10,13)
#Ein_Q14 = []
#Eout_Q15 = []
#for i in test_list:
#    LAMBDA = 10**i
#    w = LinearRegression_Regularization(X_train,Y_train,LAMBDA)
#    Ein = error_rate(X_train,Y_train,w)
#    Ein_Q14.append(Ein)
#    Eout = error_rate(X_test,Y_test,w)
#    Eout_Q15.append(Eout)
#plt.plot(test_list,Ein_Q14)
#plt.plot(test_list,Eout_Q15)

X_Dtrain = X_train[:120]
Y_Dtrain = Y_train[:120]
X_Dval = X_train[120:200]
Y_Dval = Y_train[120:200]
test_list = np.linspace(2,-10,13)
Etrain_Q16 = []
Eout_Q16 = []
Eval_Q16 = []
for i in test_list:
    LAMBDA = 10**i
    w = LinearRegression_Regularization(X_Dtrain,Y_Dtrain,LAMBDA)
    Etrain = error_rate(X_Dtrain,Y_Dtrain,w)
    Etrain_Q16.append(Etrain)
    Eout = error_rate(X_test,Y_test,w)
    Eout_Q16.append(Eout)
    Eval = error_rate(X_Dval,Y_Dval,w)
    Eval_Q16.append(Eval)
plt.plot(test_list,Etrain_Q16)
plt.plot(test_list,Eout_Q16)
plt.plot(test_list,Eval_Q16)