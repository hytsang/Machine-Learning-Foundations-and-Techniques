# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:59:37 2017

@author: haowen
"""

import numpy as np
import matplotlib.pyplot as plt

def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    num = len(dataSet)
    
    X = np.zeros((num,21))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i,0] = 1.0
        Y[i,0] = np.float(data[-1])
        for n in range(len(data)-1):
            X[i,n+1] = np.float(data[n])
    return X,Y

def calculate_gradient(x, y, w):
    N = len(x)
    gradient = np.zeros((N,21))
    for i in range(N):
        s = -y[i,0]*np.dot(x[i],w)
        theta = 1.0/(1+np.exp(-s))
        gradient[i] = theta * (-y[i,0]*x[i])
    gradient_average = np.sum(gradient, axis=0) / len(x)
    return gradient_average
    
def sign(s):
    if s >= 0.5:
        return 1
    else:
        return -1
        
def calcuError(x, y, w):
    N = len(x)
    error = 0.0
    for i in range(N):
        scores = 1.0/(1.0+np.exp(-np.dot(x[i],w)))
        if sign(scores) != Y[i,0]:
            error = error +1.0
    return error/len(X)

def update_w(w, learning_rate, gradient):
    return w - learning_rate* gradient
    
filename = r"C:\Users\haowen\Dropbox\Machine learning\Machine Learning Foundations" + "\hw3_train.dat"
X, Y = getDataSet(filename)
w = np.zeros((21,1))
loop =20000
learning_rate = 0.01
for i in range(loop):
    grad = calculate_gradient(X, Y, w).reshape(-1, 1)
    w = update_w(w, learning_rate, grad)
Error = calcuError(X,Y,w)
 