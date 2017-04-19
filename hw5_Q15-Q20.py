#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 23:38:45 2017

@author: haowenhou
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def getDataSet(filepath,digit):
    datafile = open(filepath, 'r')
    dataSet = datafile.readlines()
    num = len(dataSet)
    
    X = np.zeros((num,2))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].split()
        if float(data[0]) == digit:
            Y[i] = 1.0
        else: Y[i] = -1.0
               
        X[i,0] = data[1]
        X[i,1] = data[2]
    datafile.close()
    return X,Y

#Q15
#train_filepath = "hw1_features.train.txt"
#test_filepath = "hw1_features.test.txt"
#X_train, Y_train = getDataSet(train_filepath, 0.0)
#X_test, Y_test = getDataSet(test_filepath, 0.0)
#
#Y_train = np.ravel(Y_train)
#
#logC = [-6,-4,-2,0,2]
#Error_rate = []
#for i in logC:
#    paraC = 10**i
#    linear_svm = svm.SVC(kernel = 'linear', C= paraC)
#    linear_svm.fit(X_train,Y_train)
#    Y_predict = linear_svm.predict(X_test).reshape(2007,1)
#    error = abs(0.5*(Y_test - Y_predict))
#    error_rate = sum(error) / len(error)
#    Error_rate.append(error_rate)
#plt.plot(logC,Error_rate, 'r')

#Q16
#train_filepath = "hw1_features.train.txt"
#test_filepath = "hw1_features.test.txt"
#X_train, Y_train = getDataSet(train_filepath, 8.0)
#X_test, Y_test = getDataSet(test_filepath, 8.0)
#
#Y_train = np.ravel(Y_train)
#
#logC = [-6,-4,-2,0]
##
#
#Eout_rate = []
#Ein_rate = []
#for i in logC:
#    paraC = 10**i
#    poly_svm = svm.SVC(kernel = 'poly', C= paraC, degree = 2)
#    poly_svm.fit(X_train,Y_train)
#    
#    Y_out = poly_svm.predict(X_test).reshape(2007,1)
#    errorout = abs(0.5*(Y_test - Y_out))
#    errorout_rate = sum(errorout) / len(errorout)
#    
#    Y_in = poly_svm.predict(X_train).reshape(7291,1)
#    errorin = abs(0.5*(Y_train.reshape(7291,1) - Y_in))
#    errorin_rate = sum(errorin) / len(errorin)
#    
#    Ein_rate.append(errorin_rate)
#    Eout_rate.append(errorout_rate)
#plt.plot(logC,Eout_rate, 'r')
#plt.plot(logC,Ein_rate,'blue')

#Q17 
#logC = [-6,-4,-2,0]
#sumalpha = []
#for i in logC:
#    paraC = 10**i
#    poly_svm = svm.SVC(kernel = 'poly', C= paraC, degree = 2)
#    poly_svm.fit(X_train,Y_train)
#    alpha = poly_svm.dual_coef_.sum()
#    sumalpha.append(alpha)
#plt.plot(logC,sumalpha, 'r')    
# plot the line, the points, and the nearest vectors to the plane

##Q18
#train_filepath = "hw1_features.train.txt"
#test_filepath = "hw1_features.test.txt"
#X_train, Y_train = getDataSet(train_filepath, 0.0)
#X_test, Y_test = getDataSet(test_filepath, 0.0)
#
#Y_train = np.ravel(Y_train)
#
#logC = [-3,-2,-1,0,1]
#Error_rate = []
#for i in logC:
#    paraC = 10**i
#    linear_svm = svm.SVC(kernel = 'rbf', C= paraC, gamma = 100)
#    linear_svm.fit(X_train,Y_train)
#    Y_predict = linear_svm.predict(X_test).reshape(2007,1)
#    error = abs(0.5*(Y_test - Y_predict))
#    error_rate = sum(error) / len(error)
#    Error_rate.append(error_rate)
#plt.plot(logC,Error_rate, 'r')

#Q18
train_filepath = "hw5_features.train.txt"
test_filepath = "hw5_features.test.txt"
X_train, Y_train = getDataSet(train_filepath, 0.0)
X_test, Y_test = getDataSet(test_filepath, 0.0)

Y_train = np.ravel(Y_train)

loggamma = [0,1,2,3,4]
Error_rate = []
for i in loggamma:
    paragamma = 10**i
    linear_svm = svm.SVC(kernel = 'rbf', C= 0.1, gamma = paragamma)
    linear_svm.fit(X_train,Y_train)
    Y_predict = linear_svm.predict(X_test).reshape(2007,1)
    error = abs(0.5*(Y_test - Y_predict))
    error_rate = sum(error) / len(error)
    Error_rate.append(error_rate)
plt.plot(loggamma,Error_rate, 'r')

#plt.scatter(poly_svm.support_vectors_[:, 0], poly_svm.support_vectors_[:, 1], s=30,
#                facecolors='none', zorder=10)
#plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, zorder=10, cmap=plt.cm.Paired)
#
#plt.axis('tight')
#x_min = 0
#x_max = 0.8
#y_min = -8
#y_max = 0
#
#XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#Z = poly_svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
#
## Put the result into a color plot
#Z = Z.reshape(XX.shape)
#plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                levels=[-.5, 0, .5])
#
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
