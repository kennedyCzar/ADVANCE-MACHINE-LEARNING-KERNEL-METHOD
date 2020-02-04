#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 06:18:58 2020

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class TwoClassSVDD(EvalC, loss, Kernels):
    def __init__(self, kernel = None, C = None):
        '''Two Class SVDD
        parameters:
            kernel: kernel
            C = hyperparam
        '''
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not C:
            C = .01
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
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
    
    def cost(self, x, y):
        '''
        :Return type: cost
        '''
        return self.alpha.dot(np.dot(self.alpha, self.knl * self.y_i(self.y))) - np.sum(self.alpha*y*(np.ones_like(self.alpha)*np.linalg.norm(x)))
    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.ones(X.shape[0])
#        alpha = np.random.dirichlet(np.ones(X.shape[0]),size=1).reshape(-1, )
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.k)
        
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: lr: scalar learning rate value
        :params: iterations: integer iteration
        '''
        self.X = X
        self.y = y
        if not lr:
            lr = 1e-2
            self.lr = lr
        else:
            self.lr = lr
        if not iterations:
            iterations = 100
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.alpha, self.alpha_i_s,  self.knl = self.alpha_y_i_kernel(self.X)
        self.cost_rec = np.zeros(self.iterations)
        for ii in range(self.iterations):
            self.cost_rec[ii] = self.cost(self.X, self.y)
            print(f'Cost of computation: {self.cost_rec[ii]}')
            self.alpha = self.alpha + self.lr * (self.alpha * np.dot(self.y_i(self.y), self.knl).diagonal() - np.dot(self.knl, self.alpha * self.y))
            self.alpha[self.alpha < 0 ] = 0
            self.alpha[self.alpha > self.C] = self.C
        self.indices = np.where((self.alpha >= 0) & (self.alpha <= self.C))[0]
        self.R_squared = self.kernelize(self.X[self.indices], self.X[self.indices]).diagonal() - 2*np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices])) + \
                         self.alpha[self.indices].dot(np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices])))
        self.b = np.mean(self.R_squared - self.alpha[self.indices].dot(np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices]))))
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(2*np.dot(self.alpha, self.kernelize(self.X, X)) + self.kernelize(X, self.X)[:, 0] + self.b)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
    
#%%
##%% Testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
X, y = make_circles(1000, noise = .07, factor = .3)
#X = np.hstack((X, y.reshape(-1, 1)))
df = X[X[:, 2] == 1][:, [0, 1]]
dy = X[X[:, 2] == 1][:, 2]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)

plt.scatter(df[:, 0], df[:, 1])
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'coolwarm', s = 2) 

tcl_dsvdd = TwoClassSVDD(kernel='polynomial').fit(X_train, Y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c = tcl_dsvdd.predict(X_test), cmap = 'coolwarm', s = 2) 
    