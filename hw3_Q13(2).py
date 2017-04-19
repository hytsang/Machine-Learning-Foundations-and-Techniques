# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:29:20 2017

@author: haowen

"""
import numpy as np
import matplotlib.pyplot as plt
import random

def Addnoise(x):
    index = range(len(x))
    #10% noise
    noise = int(0.1*len(x))
    rand_index = random.sample(index,noise)
    for i in rand_index:
        x[i,0] = -x[i,0]
    return x

def generateDataSet():
    N = 1000
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.uniform(-1,1,N)
    y = np.zeros((N,1))
    f = x1**2 + x2**2 - 0.6
    for i in range(N):
        if f[i] <= 0 :
            y[i] = -1
        else: 
            y[i] = 1
    Y = np.zeros((N,1))
    Y = Addnoise(y)
    X = np.zeros((N,3))
    for i in range(N):
        X[i,0] = 1.0
        X[i,1] = x1[i]
        X[i,2] = x2[i]
    return X,Y
   
def linearRegression(x,y):
    X = np.matrix(x)
    Y = np.matrix(y)
    Wlin = np.linalg.inv(X.T*X)*X.T*Y
    return Wlin

def FeatureRransformation(x):
    N = len(x)
    Z = np.zeros((N,6))
    for i in range(N):
        Z[i,0] = x[i,0]
        Z[i,1] = x[i,1]
        Z[i,2] = x[i,2]
        Z[i,3] = x[i,1]*x[i,2]
        Z[i,4] = x[i,1]*x[i,1]
        Z[i,5] = x[i,2]*x[i,2]
    return Z
    
def sign(x,w):
    if np.dot(x,w)[0] >= 0:
        return 1
    else:
        return -1
        
def error_rate(X,Y,w):
    error = 0.0
    for i in range(len(X)):
        if sign(X[i],w) != Y[i,0]:
            error = error +1.0
    return error/len(X)

errorrate = []     
for i in range(1000):
    X, Y = generateDataSet()
    Z = FeatureRransformation(X)
    w = linearRegression(Z,Y) 
    error = error_rate(Z,Y,w)
    errorrate.append(error)
plt.hist(errorrate)