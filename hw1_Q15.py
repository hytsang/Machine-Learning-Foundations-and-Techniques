# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:48:36 2017

@author: haowen
"""
import numpy as np

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
        
def PLA_Naive(X, Y, w, speed, updates):
    iterations = 0
    num = len(X)
    flag = True
    for i in range(updates):
        flag = True
        for j in range(num):
            if sign(X[j],w) != Y[j,0]:
                flag = False
                w = w + speed*Y[j,0]*np.matrix(X[j]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    return w, flag, iterations
    
filename = r"C:\Users\haowen\Dropbox\Machine learning\Machine Learning Foundations" + "\hw1_15_train.dat"
X, Y = getDataSet(filename)
w0 = np.zeros((5,1)) 
speed = 1
updates = 80
w, flag, iterations = PLA_Naive(X, Y, w0, speed, updates)

print flag
print iterations
print w