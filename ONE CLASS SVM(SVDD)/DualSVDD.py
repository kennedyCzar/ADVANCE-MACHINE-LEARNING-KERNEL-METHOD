#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:11:23 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class DualSVDD(EvalC, loss, Kernels):
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not C:
            C = 1.0
            self.C = C
        else:
            self.C = C
        return
        
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
       
    def kernelize(self, x1, x2):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
    
    def cost(self):
        '''
        :Return type: cost
        '''
#        return np.sum(self.alpha_i_s * self.knl * self.y_i_s ) - np.sum(np.dot(self.Y, self.knl))
        return self.alpha.dot(np.dot(self.alpha, self.knl)) -  np.sum(self.alpha*self.knl.diagonal())
#    
    def alpha_y_i_kernel(self, X, y):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.random.randn(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.y_i_s = self.y_i(y) #y_i's y_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.y_i_s, self.k)
        
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: lr: scalar learning rate value
        :params: iterations: integer iteration
        '''
        self.X = X
        self.Y = y
        if not lr:
            lr = 1e-2
            self.lr = lr
        else:
            self.lr = lr
        if not iterations:
            iterations = 1
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(self.X, self.Y)
        self.cost_rec = np.zeros(self.iterations)
        for ii in range(self.iterations):
            self.cost_rec[ii] = self.cost()
            print(f'Cost of computation: {self.cost_rec[ii]}')
            self.alpha = self.alpha + self.lr * (np.dot(self.knl, self.alpha) - self.knl.diagonal())
#            self.alpha = self.alpha + self.lr * (np.dot(self.y_i_s * self.knl, self.alpha)) - np.dot(self.knl, self.Y)
#            self.alpha[self.alpha > 0] = 1
#            self.alpha[self.alpha > self.C] = self.C
        indices = np.where((self.alpha > 0) & (self.alpha < self.C))
        self.support_vectors = indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        self.rho = self.alpha.dot(self.kernelize(self.X, self.X[self.support_vectors]))
        yhat:int = np.sign(np.dot(self.alpha, self.kernelize(self.X, X)))
        return yhat
    
#%% Testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
X, y = make_circles(1000, noise = .07, factor = .5)
plt.scatter(X[:, 0], X[:, 1])
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
dsvdd = DualSVDD().fit(X_train, Y_train)
dsvdd.predict(X_test)
dsvdd.summary(Y_test, dsvdd.predict(X_test), dsvdd.alpha)  
plt.scatter(X_test[:, 0], X_test[:, 1], c = dsvdd.predict(X_test))   
