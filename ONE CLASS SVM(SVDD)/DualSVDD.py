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
            C = 0.01
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
    
    def cost(self):
        '''
        :Return type: cost
        '''
        return np.sum(self.alpha*self.knl.diagonal()) - self.alpha.dot(np.dot(self.alpha, self.knl))
#    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.ones(X.shape[0])
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
        yhat:int = np.sign(2*np.dot(self.alpha, self.kernelize(self.X, X)) + self.kernelize(X, self.X)[:, 0] + self.b)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
class DualSVDD_NE(EvalC, loss, Kernels):
    def __init__(self, kernel = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
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
        alpha = np.ones(X.shape[0])
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
            self.alpha = self.alpha + self.lr * ( self.knl.diagonal() - np.dot(self.knl, self.alpha))
            self.alpha[self.alpha < 0 ] = 0
            self.alpha[self.alpha > 1] = 1
        self.indices = np.where((self.alpha > 0))[0]
        self.R_squared = self.kernelize(self.X, self.X[self.indices]).diagonal() - 2*np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])) + \
                         self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])))
        self.b = np.mean(self.R_squared - self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices]))))
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(2*np.dot(self.alpha, self.kernelize(self.X, X)) + self.b)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    

class MiniDualSVDD(EvalC, loss, Kernels):
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not C:
            C = 1e-2
            self.C = C
        else:
            self.C = C
        return
        
       
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
    
    def kernelsamp(self, X):
        '''
        :return type: kernel
        '''
        return self.kernelize(X, X)
    
    def cost(self, kernl, alpha):
        '''
        :Return type: cost
        '''
        return np.sum(alpha*kernl.diagonal()) - alpha.dot(np.dot(alpha, kernl))
    
    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.ones(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.k)
        
    def fit(self, X, lr:float = None, iterations:int = None, batch = None):
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
        if not batch:
            batch = 20
            self.batch = batch
        else:
            self.batch = batch
        self.alpha, self.alpha_i_s,  self.knl = self.alpha_y_i_kernel(self.X)
        self.cost_rec = np.zeros(self.iterations)
        datlen = len(self.X)
        for ii in range(self.iterations):
            self.sampledCost = []
            random_samples = np.random.permutation(datlen)
            X_samp = self.X[random_samples]
            for ij in range(0, datlen, self.batch):
                X_samp = self.X[ij:ij+self.batch]
                self.alphasample = self.alpha[ij:ij+self.batch]
                self.knlsample = self.kernelsamp(X_samp)
                self.alphasample = self.alphasample + self.lr * (self.knlsample.diagonal() - np.dot(self.knlsample, self.alphasample))
                self.alpha[ij:ij+self.batch] = self.alphasample
                self.alpha[self.alpha < 0] = 0
                self.alpha[self.alpha > 0] = self.alpha
                self.alpha[self.alpha > self.C] = self.C
                self.alpha = np.round(self.alpha, 2)
                self.sampledCost.append(self.cost(self.knlsample, self.alphasample))
                if self.sampledCost[ij] >= self.sampledCost[ij -1]:
                    break
                else:
                    continue
            self.cost_rec[ii] = np.sum(self.sampledCost)
            print('*'*40)
            print(f'Cost of computation: {self.cost_rec[ii]}')
        self.indices = np.where((self.alpha >= 0) & (self.alpha <= self.C))[0]
        self.R_squared = self.kernelize(self.X, self.X[self.indices]).diagonal() - 2*np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])) + \
                         self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])))
        self.b = np.mean(self.R_squared - self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices]))))
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(2*np.dot(self.alpha, self.kernelize(self.X, X)) + self.b)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
class MiniDualSVDD_NE(EvalC, loss, Kernels):
    def __init__(self, kernel = None, C = None):
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
        
       
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
    
    def kernelsamp(self, X):
        '''
        :return type: kernel
        '''
        return self.kernelize(X, X)
    
    def cost(self, kernl, alpha):
        '''
        :Return type: cost
        '''
        return np.sum(alpha*kernl.diagonal()) - alpha.dot(np.dot(alpha, kernl))
    
    
    def alpha_y_i_kernel(self, X):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.ones(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.k)
        
    def fit(self, X, lr:float = None, iterations:int = None, batch = None):
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
        if not batch:
            batch = 20
            self.batch = batch
        else:
            self.batch = batch
        self.alpha, self.alpha_i_s,  self.knl = self.alpha_y_i_kernel(self.X)
        self.cost_rec = np.zeros(self.iterations)
        datlen = len(self.X)
        for ii in range(self.iterations):
            self.sampledCost = []
            random_samples = np.random.permutation(datlen)
            X_samp = self.X[random_samples]
            for ij in range(0, datlen, self.batch):
                X_samp = self.X[ij:ij+self.batch]
                self.alphasample = self.alpha[ij:ij+self.batch]
                self.knlsample = self.kernelsamp(X_samp)
                self.alphasample = self.alphasample + self.lr * ( self.knlsample.diagonal() - np.dot(self.knlsample, self.alphasample))
                self.alpha[ij:ij+self.batch] = self.alphasample
                self.alpha[self.alpha <= 0] = 0
                self.alpha[self.alpha < 1] = 1
                self.alpha = np.round(self.alpha, 2)
                self.sampledCost.append(self.cost(self.knlsample, self.alphasample))
                if self.sampledCost[ij] >= self.sampledCost[ij -1]:
                    break
                else:
                    continue
            self.cost_rec[ii] = np.sum(self.sampledCost)
            print('*'*40)
            print(f'Cost of computation: {self.cost_rec[ii]}')
        self.indices = np.where((self.alpha >= 0))[0]
        self.R_squared = self.kernelize(self.X, self.X[self.indices]).diagonal() - 2*np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])) + \
                         self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices])))
        self.b = np.mean(self.R_squared - self.alpha.dot(np.dot(self.alpha, self.kernelize(self.X, self.X[self.indices]))))
        self.support_vectors = self.indices
        print(f'Total support vectors required for classification: {len(self.support_vectors)}')
        return self
    
    def predict(self, X):
        yhat:int = np.sign(2*np.dot(self.alpha, self.kernelize(self.X, X)) + self.b)
        for enum, ii in enumerate(yhat):
            if yhat[enum] == -1:
                yhat[enum] = 0
        return yhat
    
##%% Testing
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs, make_moons, make_circles
#from sklearn.model_selection import train_test_split
#X, y = make_circles(1000, noise = .07, factor = .3)
#X = np.hstack((X, y.reshape(-1, 1)))
#df = X[X[:, 2] == 1][:, [0, 1]]
#dy = X[X[:, 2] == 1][:, 2]
#plt.scatter(df[:, 0], df[:, 1])
#plt.scatter(X[:, 0], X[:, 1])
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
#dsvdd = DualSVDD(kernel='linear').fit(df)
#plt.plot(np.arange(100), dsvdd.cost_rec)
#dsvdd.predict(X[:, [0, 1]])
#plt.scatter(X[:, 0], X[:, 1], c = dsvdd.predict(X[:, [0, 1]]))
#
##%% SVDD-GD No Errors
#dsvddNE = DualSVDD_NE(kernel='rbf').fit(df)
#plt.plot(np.arange(100), dsvddNE.cost_rec)
#dsvddNE.predict(X[:, [0, 1]])
#plt.scatter(X[:, 0], X[:, 1], c = dsvddNE.predict(X[:, [0, 1]]))
#
#
##%% Minibatch with Errors
#stochdsvdd = MiniDualSVDD(kernel='rbf').fit(df)
#plt.plot(np.arange(100), stochdsvdd.cost_rec)
#stochdsvdd.predict(X[:, [0, 1]])
#stochdsvdd.summary(y, stochdsvdd.predict(X[:, [0, 1]]), stochdsvdd.alpha)
#plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 1)
#
##%% Minibatch version with no errors
#stochdsvdd = MiniDualSVDD_NE(kernel='laplace').fit(df)
#plt.plot(np.arange(100), stochdsvdd.cost_rec)
#stochdsvdd.predict(X[:, [0, 1]])
#plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]))
#
##%% Theroid Dataset
#from scipy.io import loadmat
#import os
#
#path = '/home/kenneth/Documents/ALGOSTATML/SUPERVISED-ML/CLASSIFICATION/ONE CLASS SVM(SVDD)/DATASET'
#dfthy, dfy = loadmat(os.path.join(path, 'thyroid.mat'))['X'], loadmat(os.path.join(path, 'thyroid.mat'))['y']
#sample  = np.hstack((dfthy, dfy.reshape(-1, 1)))
#Xsample = sample[sample[:, -1] == 0]
#Xsample = Xsample[:, :-1]
#ysample = sample[sample[:, -1] == 0]
#ysample = ysample[:, -1]
#stochdsvdd = MiniDualSVDD(kernel='rbf').fit(Xsample)
#pred = stochdsvdd.predict(sample[:, :-1])
#stochdsvdd.summary(dfy, pred, stochdsvdd.alpha)
#plt.plot(np.arange(100), stochdsvdd.cost_rec)
#plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]))
#
#
##%%
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
#X, y = make_moons(n_samples=1000, noise=.1)
#plt.scatter(X[:, 0], X[:, 1], c = y, s = 5, cmap='coolwarm_r')
#X = np.hstack((X, y.reshape(-1, 1)))
#df = X[X[:, 2] == 1][:, [0, 1]]
#dy = X[X[:, 2] == 1][:, 2]
#
#def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
#    h = 0.01
#    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
#    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    plt.figure(figsize=(5,5))
#    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.20)
#    plt.contour(xx, yy, Z, colors='k', linewidths=0.01)
#    plt.scatter(X[:,0], X[:,1], c =  Y, cmap = cmap, edgecolors='k', label = f'F1: {round(stochdsvdd.f1(y, stochdsvdd.predict(X[:, [0, 1]])), 2)}')
#    plt.legend()
#plot_decision_boundary(stochdsvdd, X, y, cmap='coolwarm_r')

