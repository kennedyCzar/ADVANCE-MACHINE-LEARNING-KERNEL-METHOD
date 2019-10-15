#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:51:41 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalR
from Utils.Loss import loss


class linearSVM(loss):
    def __init__(self, C = None):
        '''
        Linear SVM via Gradient descent
        :params: C: misclassification penalty or regularizer. 
                    Default value is 1.0.
        '''
        if not C:
            C = 10
            self.C = C
        else:
            self.C = C
        return
    
    def cost(self, X, y, beta, b):
        '''
        Hinge loss function
        --------------------
        :params: X: feature space
        :params: y: target
        :params: beta: weights parameters.
        '''
        return 0.5 * beta.dot(beta) + self.C * np.sum(loss.hinge(X, y, beta, b))
    
    def margins(self, X, y, beta, b):
        '''
        :param: X: NxD
        :param: y: Nx1
        :param: beta: Dx1
        '''
        return y*(np.dot(X, beta) + b) 
        
    def fit(self, X, y, alpha:float = None, iterations:int = None, earlystoping = None):
        if not alpha:
            alpha = 1e-5
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 500
            self.iterations = iterations
        else:
            self.iterations = iterations
        if not earlystoping:
            earlystoping = True
            self.earlystoping = earlystoping
        else:
            self.earlystoping = earlystoping
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        self.b = 0
        for ii in range(self.iterations):
            self.margin = self.margins(X, y, self.beta, self.b)
            #adjust parameters according to misclafication
            indices = np.where(self.margin < 1)
            self.beta = self.beta - self.alpha*(self.beta - self.C * y[indices].dot(X[indices]))
            self.b = self.b - self.alpha * self.C * np.sum(y[indices])
            self.beta_rec[ii, :] = self.beta.T
            self.cost_rec[ii] = self.cost(X, y, self.beta, self.b)
            print(f"cost of computation: {self.cost(X, y, self.beta, self.b)}, Gradient: {[self.b] + self.beta - self.C * y[indices].dot(X[indices])}")
            if not earlystoping:
                pass
            else:
                if self.cost_rec[ii] > self.cost_rec[ii - 1]:
                    break
        return self
    
    def predict(self, X):
        yhat:int = np.zeros(X.shape[0])
        for enum, ii in enumerate(np.sign(X.dot(self.beta) + self.b)):
            if ii >0:
                yhat[enum] = 1
        return yhat
    

class StochasticlinearSVM(loss):
    def __init__(self, C = None):
        '''
        Linear SVM via Stochastic Gradient descent
        :params: C: misclassification penalty. 
                    Default value is 0.1.
        '''
        if not C:
            C = 10.0
            self.C = C
        else:
            self.C = C
        return
    
    def cost(self, X, y, beta, b):
        '''
        Hinge loss function
        ------------------
        :params: X: feature space
        :params: y: target
        :params: beta: weights parameters.
        '''
        return 0.5 * beta.dot(beta) + self.C * np.sum(loss.hinge(X, y, beta, b))
    
    def margins(self, X, y, beta:float, b):
        '''
        :param: X: NxD
        :param: y: Nx1
        :param: beta: Dx1
        '''
        return y*(np.dot(X, beta) + b)
        
    def fit(self, X, y, alpha:float = None, iterations:int = None, earlystoping = None):
        if not alpha:
            alpha = 1e-5
            self.alpha = alpha
        else:
            self.alpha = alpha
        if not iterations:
            iterations = 50
            self.iterations = iterations
        else:
            self.iterations = iterations
        if not earlystoping:
            earlystoping = True
            self.earlystoping = earlystoping
        else:
            self.earlystoping = earlystoping
        self.b = 0
        self.beta = np.zeros(X.shape[1])
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        ylen = len(y)
        for ii in range(self.iterations):
            for ij in range(ylen):
                random_samples = np.random.randint(1, ylen)
                X_samp = X[:random_samples]
                y_samp = y[:random_samples]
                self.margin = self.margins(X_samp, y_samp, self.beta, self.b)
                #adjust parameters according to misclafication
                indices = np.where(self.margin < 1)
                self.beta = self.beta - self.alpha*(self.beta - self.C * y_samp[indices].dot(X_samp[indices]))
                self.b = self.b - self.alpha * self.C * np.sum(y_samp[indices])
                self.beta_rec[ii, :] = self.beta.T
                self.cost_rec[ii] += self.cost(X, y, self.beta, self.b)
                print(f"cost is computation: {self.cost(X, y, self.beta, self.b)}")
                if not earlystoping:
                    pass
                else:
                    if self.cost_rec[ii] > self.cost_rec[ii - 1]:
                        break
        return self
    
    def predict(self, X):
        yhat:int = np.zeros(X.shape[0])
        for enum, ii in enumerate(np.sign(X.dot(self.beta) + self.b)):
            if ii >0:
                yhat[enum] = 1
        return yhat
              
#%%
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
X, y = make_moons(n_samples=1000, noise=.1)
X, y = make_blobs(n_samples=1000, centers=2, n_features=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
lsvm = linearSVM().fit(X_train, Y_train)
lsvm.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = lsvm.predict(X_test))
np.mean(lsvm.predict(X_test) == Y_test)
plt.plot(np.arange(lsvm.iterations), lsvm.cost_rec)

slsvm = StochasticlinearSVM().fit(X_train, Y_train)
slsvm.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = slsvm.predict(X_test))
np.mean(slsvm.predict(X_test) == Y_test)
plt.plot(np.arange(slsvm.iterations), slsvm.cost_rec)


