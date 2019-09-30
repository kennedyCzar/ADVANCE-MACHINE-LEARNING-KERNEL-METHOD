#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:31:30 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss

#logistic regression using gradient descent.
class Logistic(EvalC):
    def __init__(self):
        return
    
    @staticmethod
    def cost(X, Y, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: Y: label y \in {0,1} N x 1 dimension
        :params: beta: weights N x 1
        
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(loss.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - loss.sigmoid(X, beta))))
    
    def fit(self, X, Y, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        for ii in range(self.iterations):
            #compute gradient
            self.beta = self.beta + (1/len(Y)) *(self.alpha) * X.T.dot(Y - loss.sigmoid(X, self.beta))
            self.beta_rec[ii, :] = self.beta.T
            self.cost_rec[ii] = self.cost(X, Y, self.beta)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(len(y_pred)):
            if loss.sigmoid(X[ii], self.beta) > 0.5:
                y_pred[ii] = 1
        return y_pred

'''   
Logistic regression using stochastic
logistic regression
'''
class stochasticLogistic(EvalC):
    def __init__(self, alpha, iterations):
        super().__init__()
        self.alpha = alpha
        self.iterations = iterations
        return
           

    @staticmethod
    def cost(X, Y, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: Y: label y \in {0,1} N x 1 dimension
        :params: beta: weights N x 1
        
        '''
        return -(1/len(Y)) * np.sum((Y*np.log(loss.sigmoid(X, beta))) +\
                 ((1 - Y)*np.log(1 - loss.sigmoid(X, beta))))
    
    def fit(self, X, Y):
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        ylen = len(Y)
        for ii in range(self.iterations):
            #compute stochastic gradient
            sampledCost = []
            for ij in range(ylen):
                random_samples = np.random.randint(1, ylen)
                X_samp = X[:random_samples]
                Y_samp = Y[:random_samples]
                self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - loss.sigmoid(X_samp, self.beta))
                self.beta_rec[ii, :] = self.beta.T
                sampledCost.append(self.cost(X_samp, Y_samp, self.beta))
            self.cost_rec[ii] = np.average(sampledCost)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(len(y_pred)):
            if loss.sigmoid(X[ii], self.beta) > 0.5:
                y_pred[ii] = 1
        return y_pred
    

#%% Test and compare with that from sklearn

#logistic regression withhout regularization using GD
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
logit = Logistic().fit(X_train, Y_train, 0.1, 100)
logit.summary(Y_test, logit.predict(X_test), logit.beta)    

#logistic regression withhout regularization using SGD
slogit = stochasticLogistic(alpha=0.1, iterations=100).fit(X_train, Y_train)
slogit.summary(Y_test, slogit.predict(X_test), slogit.beta)

import matplotlib.pyplot as plt
plt.plot(np.arange(logit.iterations), logit.cost_rec, label = 'logistic via GD')
plt.plot(np.arange(slogit.iterations), slogit.cost_rec, label = 'logistic via SGD')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()




