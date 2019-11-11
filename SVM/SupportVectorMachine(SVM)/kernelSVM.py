#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:28:19 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class kDualSVM(EvalC, loss, Kernels):
    '''
    Kernelized SVM via Gradient ascent.
    ------------------------------------
    Dual Lagrangian formulation
    for kernel SVMs.
    '''
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
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
        
    def cost(self):
        '''
        return type: x E R
        '''
        return np.dot(self.alpha, np.ones(self.X.shape[0])) - .5 * np.sum(self.alpha_i_s * self.knl * self.y_i_s )
    
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
            iterations = 3
            self.iteration = iterations
        else:
            self.iteration = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(self.X, self.Y)
        self.b = 0
        cost = np.zeros(iterations)
        for ii in  range(self.iteration):
            cost[ii] = self.cost()
            print(f"Cost of computation: {cost[ii]}")
            #perform gradient ascent for maximization.
            self.alpha = self.alpha + self.lr * (np.ones(X.shape[0]) - np.dot(self.y_i_s * self.knl, self.alpha))
            #0 < alpha < C
            self.alpha[self.alpha < 0] = 0
            self.alpha[self.alpha > self.C] = self.C
        self.indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        self.b = self.Y[self.indices] - np.dot(self.alpha * self.Y, self.kernelize(self.X, self.X[self.indices]))
        self.b = np.mean(self.b)
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(np.dot(self.alpha * self.Y, self.kernelize(self.X, X)) + self.b)
        for enum, ii in enumerate(yhat):
            if ii <=0:
                yhat[enum] = 0
        return yhat
            
        
class kprimalSVM(EvalC, loss, Kernels):
    '''
    Kernelized SVM via Gradient ascent.
    -----------------------------------
    Primal function tranformed to dual formulation
    for kernel SVMs.
    '''
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
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
        :params: X: NxD
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
        
    def hingeloss(self):
        '''
        Hinge loss for the dual primal
        '''
        return np.maximum(0, 1 - self.Y * (self.alpha.dot(self.kernelize(self.X, self.X)) + self.b))
    
    def cost(self):
        '''
        return type: x E R
        '''
        return .5* self.alpha.dot(self.knl.dot(self.alpha)) + self.C * np.sum(self.hingeloss())
    
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :params: X: NxD feature matrix
        :params: y: Dx1 target vector
        :params: alpha: scalar alpha value
        :params: iterations: integer iteration
        '''
        self.X = X
        self.Y = y
        self.b = 0
        if not lr:
            lr = 1e-2
            self.lr = lr
        else:
            self.lr = lr
        if not iterations:
            iterations = 3
            self.iteration = iterations
        else:
            self.iteration = iterations
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(X, y)
        cost = np.zeros(self.iteration)
        self.cost_rec = np.zeros(self.iteration)
        for ii in  range(self.iteration):
            self.margin = self.Y * (self.alpha.dot(self.kernelize(self.X, self.X)) + self.b)
            indices = np.where(self.margin < 1)
            cost[ii] = self.cost()
            print(f"Cost of computation: {cost[ii]}")
            #perform gradient descent for maximization.
            self.alpha = self.alpha - self.lr * (self.knl.dot(self.alpha) - self.C * self.Y[indices].dot(self.knl[indices]))
            self.b = self.b - self.lr * self.C * np.sum(self.Y[indices])
        #the support vectors are datapoints exactly on the margin
        self.support_vectors = np.where((self.Y * (self.alpha.dot(self.kernelize(self.X, self.X)) + self.b)) <= 1)[0]
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(np.dot(self.alpha * self.Y, self.kernelize(self.X, X)) + self.b)
        for enum, ii in enumerate(yhat):
            if ii <0:
                yhat[enum] = 0
        return yhat
    
    
    
    
#%% Testing
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs, make_moons
#from sklearn.model_selection import train_test_split
#X, y = make_moons(1000)
#X, y = make_blobs(n_samples=1000, centers=2, n_features=2)
##X = np.c_[np.ones(X.shape[0]), X]
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
#kernelsvm = kDualSVM(kernel='rbf').fit(X_train, Y_train)
#kernelsvm.predict(X_test)
#kernelsvm.summary(Y_test, kernelsvm.predict(X_test), kernelsvm.alpha)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = kernelsvm.predict(X_test))      
#
##%% For Testing One class | SVDD Comparison
#kernelsvm = kDualSVM(kernel='linear').fit(df, dy)
#kernelsvm.predict(X[:, [0, 1]])
#plt.scatter(X[:, 0], X[:, 1], c = kernelsvm.predict(X[:, [0, 1]]))
#kernelsvm.summary(y, kernelsvm.predict(X[:, [0, 1]]), kernelsvm.alpha)
#
##%% Thyroid Dataset
#kernelsvm = kDualSVM(kernel='rbf').fit(Xsample, ysample)
#pred = kernelsvm.predict(sample[:, :-1])
#kernelsvm.summary(dfy, pred, kernelsvm.alpha)
##%%
#
#primalkernelsvm = kprimalSVM(kernel='rbf').fit(X_train, Y_train)
#primalkernelsvm.predict(X_test)
#primalkernelsvm.summary(Y_test, primalkernelsvm.predict(X_test), primalkernelsvm.alpha)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = primalkernelsvm.predict(X_test))          
#plt.plot(np.arange(primalkernelsvm.iteration), primalkernelsvm.cost_rec)            
#       
#           
##%% Testing SVC from sklearn
#
#from sklearn.svm import SVC
#svc = SVC(C = 1.0, kernel = 'rbf',gamma='auto')
#svc.fit(X_train, Y_train)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = svc.predict(X_test))







