# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:29:20 2017

@author: haowen

"""
import numpy as np
import matplotlib.pyplot as plt
import random

def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()
    num = len(dataSet)
    
    X = np.zeros((num,5))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i,0] = 1.0
        X[i,1] = np.float(data[0])
        X[i,2] = np.float(data[1])
        X[i,3] = np.float(data[2])
        X[i,4] = np.float(data[3])
        Y[i,0] = np.float(data[4])
    return X,Y

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
        
def PLA_Pocket(X, Y, w, speed, updates):
    error = 0.0
    num = len(X)
    rand_sort = range(len(X))
    rand_sort = random.sample(rand_sort, len(X))
    for i in range(updates):
        for j in range(num):
            if sign(X[rand_sort[j]],w) != Y[rand_sort[j],0]:
                wt = w + speed*Y[rand_sort[j],0]*np.matrix(X[rand_sort[j]]).T
                error0 = error_rate(X,Y,w)
                error1 = error_rate(X,Y,wt)
                if error1 < error0:
                    w = wt
                    error = error1
                break
    return w, error
    
filename = r"C:\Users\haowen\Dropbox\Machine learning\Machine Learning Foundations" + "\hw1_18_train.dat"
X, Y = getDataSet(filename)
w0 = np.zeros((5,1)) 
speed = 1
updates = 50
errorrate = []  
for i in range(100):
    w, error = PLA_Pocket(X, Y, w0, speed, updates)
    errorrate.append(error)
plt.hist(errorrate)