#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:48:41 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class linearSVDD(EvalC, loss, Kernels):
    def __init__(self, kernel = None):
        super().__init__()
        if not kernel:
            kernel = 'linear'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
        
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
       
    def kernelize(self, x1, x2, type = None):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            if not type:
                return Kernels.linear_svdd(x1, x2)
            else:
                return Kernels.linear(x1, x2)
        
    def distance(self, x, w, axs = 0):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun Euclidean distance matrix
        '''
        return np.linalg.norm(x - w, axis = axs)
    
    def cost(self):
        '''
        :Return type: cost
        '''
        return np.sum(self.alpha*self.knl.diagonal()) - self.alpha.dot(np.dot(self.alpha, self.knl))
    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.random.dirichlet(np.ones(X.shape[1]),size=1).reshape(-1, )
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.k)
        
    def fit(self, X, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: lr: scalar learning rate value
        :params: iterations: integer iteration
        '''
        self.X = X
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
            self.cost_rec[ii] = self.cost()
            print(f'Cost of computation: {self.cost_rec[ii]}')
            self.alpha = self.alpha + self.lr * (self.knl.diagonal() - np.dot(self.knl, self.alpha))
            self.alpha[self.alpha < 0 ] = 0
        self.indices = np.where((self.alpha >= 0))[0]
        self.R_squared = self.kernelize(self.X[self.indices], self.X[self.indices]).diagonal() - 2*np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices])) + \
                         self.alpha[self.indices].dot(np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices])))
        self.b = np.mean(self.R_squared - self.alpha[self.indices].dot(np.dot(self.alpha[self.indices], self.kernelize(self.X[self.indices], self.X[self.indices]))))
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        for ii in range(X.shape[0]):
            euclid = self.distance(X[ii], self.alpha)
        yhat:int = np.sign(X[:, [0, 1]].dot(self.R_squared) -0.5*self.kernelize(X, X, type = 'euc').diagonal() + euclid)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
    
class linearSVDD_NE(EvalC, loss, Kernels):
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'linear'
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
       
    def kernelize(self, x1, x2, type = None):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            if not type:
                return Kernels.linear_svdd(x1, x2)
            else:
                return Kernels.linear(x1, x2)
        
    def distance(self, x, w, axs = 0):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun Euclidean distance matrix
        '''
        return np.linalg.norm(x - w, axis = axs)
    
    def cost(self):
        '''
        :Return type: cost
        '''
        return np.sum(self.alpha*self.knl.diagonal()) - self.alpha.dot(np.dot(self.alpha, self.knl))
    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
#        alpha = np.ones(X.shape[0])
        alpha = np.random.dirichlet(np.ones(X.shape[1]),size=1).reshape(-1, )
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.k)
        
    def fit(self, X, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: lr: scalar learning rate value
        :params: iterations: integer iteration
        '''
        self.X = X
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
            self.cost_rec[ii] = self.cost()
            print(f'Cost of computation: {self.cost_rec[ii]}')
            self.alpha = self.alpha + self.lr * (self.knl.diagonal() - np.dot(self.knl, self.alpha))
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
        for ii in range(X.shape[0]):
            euclid = self.distance(X[ii], self.alpha)
        yhat:int = np.sign(X[:, [0, 1]].dot(self.R_squared) -0.5*self.kernelize(X, X, type = 'euc').diagonal() + euclid)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
    
#%% SVDD No Error
    
#linsvdd_ne = linearSVDD(kernel='linear').fit(df)
#plt.plot(np.arange(100), linsvdd_ne.cost_rec)
#plt.scatter(X[:, 0], X[:, 1], c = linsvdd_ne.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 5)
#
#roc_auc_score(y, np.sign(X[:, [0, 1]].dot(linsvdd_ne.R_squared) -0.5*x.diagonal()))
#
#
##%% SVDD with Error
#    
#linsvdd_ne = linearSVDD_NE(kernel='linear').fit(df)
#plt.scatter(X[:, 0], X[:, 1], c = linsvdd_ne.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 5)
#
#roc_auc_score(y, np.sign(X[:, [0, 1]].dot(linsvdd.R_squared) -0.5*x.diagonal()))


