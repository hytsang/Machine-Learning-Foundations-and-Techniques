#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:03:32 2017

@author: haowenhou
"""

import numpy as np
import matplotlib.pyplot as plt

def getDataSet(filepath):
    datafile = open(filepath, 'r')
    dataSet = datafile.readlines()
    num = len(dataSet)
    X = np.zeros((num,2))
    Y = np.zeros((num,1))
    for i in range(num):
        data = dataSet[i].split()
        X[i,0] = data[0]
        X[i,1] = data[1]
        Y[i,0] = data[2]
    datafile.close()
    return X,Y

def DecisionStump(dataSet,i,theta,s):
    y = np.zeros(dataSet.shape[0])
    if s == 1:
        y = np.where(dataSet[:,i] > theta,1,-1)
    else:
        y = np.where(dataSet[:,i] < theta,1,-1)
    return y.reshape(dataSet.shape[0],1)

def u_update(Y_predict, Y, u):

    error = 0.5*np.abs(Y - Y_prediction)
    eps = np.sum(error * u)/ np.sum(u)
    correctness = np.sqrt((1.0 - eps) / eps)
    alpha = np.log(correctness)
    for i in range(len(error)):
        if error[i] == 1.0:
            u[i] = u[i]*correctness
        else:
            u[i] = u[i]/correctness
    #u = u/sum(u)
    return u,alpha,eps
    

def StumpPosition(dataArr,label,u):
    dataMat = np.matrix(dataArr); labelMat = np.matrix(label).T
    uMat = np.matrix(u)
    m, n = dataMat.shape
    numSteps = 100.0; bestStump = {}; #bestClassEst = np.matrix(np.zeros((m, 1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = min(dataMat[:, i]); rangeMax = max(dataMat[:, i])
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps)+1):
            for inequal in [-1, 1]:
                threshVal = rangeMin + float(j) * stepSize
                predicatedVal = DecisionStump(dataMat, i, threshVal, inequal)
                #errArr = np.matrix(np.ones((m, 1)))
                #for the row that predicatedVal == labelMat, errArr[row] = 0
                errArr = (predicatedVal != labelMat.T)
                weightedError = uMat.T * errArr
                #print 'split: dim %d, thesh %.2f, ineqal: %s, \
                #weighted error:%.3f' %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    #bestClassEst = predicatedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal   
    
    return bestStump['thresh'],bestStump['dim'], bestStump['ineq'], minError            

def predict_ensemble(dataSet, gt):
    predictions = np.array([DecisionStump(dataSet, i, theta, s) * alpha for (theta,s,i,alpha) in gt])
    prediction = np.sum(predictions, 0)
    prediction = (2 * (prediction > 0) - 1).astype(int)
    return prediction


# hw2Q12-Q18
# ===========
train_filepath = "hw6_adaboost_train.dat.txt"
test_filepath = "hw6_adaboost_test.dat.txt"
X_train, Y_train = getDataSet(train_filepath)
X_test, Y_test = getDataSet(test_filepath)

iters = 300
u = np.ones([len(Y_train),1])/len(Y_train)
gt = []
Eps = []
Ein = []
U = []
feature_num = X_train.shape[1]
Best_theta = np.zeros([feature_num,1])
Best_s = np.zeros([feature_num,1])
Best_Ein = np.zeros([feature_num,1])
for n in range(iters):
    Best_theta,Best_i, Best_s,Best_Ein = StumpPosition(X_train,Y_train,u)
    Y_prediction = DecisionStump(X_train,Best_i,Best_theta,Best_s)
    u,alpha,eps = u_update(Y_prediction,Y_train,u)
    Eps.append(eps)
    gt.append((float(Best_theta),Best_s,Best_i,alpha))
    Ein.append(float(Best_Ein)) #Q12
    U.append(sum(u))
plt.plot(range(iters),Ein) #Q13 plot Ein(gt) with iteration time
plt.plot(range(iters),U) #Q15 plot sum(u) with iteration time
plt.plot(range(iters),Eps) #Q16 plot eps with iteration time
Y_predict_Ein = predict_ensemble(X_train,gt)

#Q14 Plot Ein(G) with iteration
Ein_train_iters = []
for i in range(1,len(gt)+1):
    Y_predict_iters = predict_ensemble(X_train,gt[:i])
    Ein_train = float(np.sum(Y_train != Y_predict_iters))/len(Y_train)
    Ein_train_iters.append(Ein_train)
plt.plot(range(iters),Ein_train_iters) 
#Q18 plot Eout(G) with iteration time        
Eout_train_iters = []
for i in range(1,len(gt)+1):
    Y_predict_Eout = predict_ensemble(X_test,gt[:i])
    Eout_train = float(np.sum(Y_test != Y_predict_Eout))/len(Y_test)
    Eout_train_iters.append(Eout_train)
plt.plot(range(iters),Eout_train_iters) 
        
Y_predict_Eout = predict_ensemble(X_test,gt)
Eout_test = float(np.sum(Y_test != Y_predict_Eout))/len(Y_test)
    
# let's plot the decision function
beg = 0 
end = 1
npoints = 150
xrange = np.linspace(beg, end, npoints)
yrange = np.linspace(beg, end, npoints)
xgrid, ygrid = np.meshgrid(xrange, yrange)
zgrid = np.empty((npoints, npoints))

X_grid = np.array([xgrid.reshape(npoints * npoints), ygrid.reshape(npoints * npoints)]).T
zgrid = predict_ensemble(X_grid, gt).reshape(npoints, npoints)

plt.figure()
plt.suptitle('AdaBoost')

plt.subplot(211)
plt.pcolor(xgrid, ygrid, zgrid)
plt.scatter(X_train[:,0],X_train[:,1], c = Y_train, cmap=plt.cm.Paired)

plt.subplot(212)
plt.pcolor(xgrid, ygrid, zgrid)
plt.scatter(X_test[:,0],X_test[:,1], c = Y_test,cmap=plt.cm.Paired)

plt.show()    
