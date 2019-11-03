#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:49:47 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class KLR(EvalC, loss, Kernels):
    def __init__(self, kernel = None, lamda = None):
        '''
        Kernel Logistic Regression via Gradient Descent. [Solution 1]
        ---------------------------------------------------------------------
        :param: kernel: kernel for computing Gram matrix
        :param: lamda: regularization parameter
        Reference: http://dept.stat.lsa.umich.edu/~jizhu/pubs/Zhu-JCGS05.pdf
        '''
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not lamda:
            lamda = 1e-5
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    def activation(self):
        '''
        return binary array
        '''
        return 1/(1+ np.exp(-self.knl.dot(self.alpha)))
    
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
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2)
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
        
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
    
    def alpha_y_i_kernel(self, X, y):
        '''
        :params: X: NxD feature space
        :params: y: Dx1 dimension
        '''
        alpha = np.zeros(X.shape[0])
        self.alph_s = np.outer(alpha, alpha) #alpha_i's alpha_j's
        self.y_i_s = self.y_i(y) #y_i's y_j's
        self.k = self.kernelize(X, X)
        return (alpha, self.alph_s, self.y_i_s, self.k)
    
    def cost(self):
        '''
        return: cost
         loss as seen in paper 1/len(self.Y)*np.ones(self.X.shape[0]).T.dot(1 + np.exp(-self.knl.dot(self.Y)))
        '''
        
#        return 1/len(self.Y)*np.ones(self.X.shape[0]).T.dot(1 + np.exp(-self.knl.dot(self.Y)))
        return -1/len(self.Y)*(self.Y.T.dot(np.dot(self.knl, self.alpha)) - np.sum(np.log(1 + np.exp(np.dot(self.knl, self.alpha)))) -\
                .5*self.lamda*np.dot(self.alpha.T, np.dot(self.knl, self.alpha)))
        
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :param: X: NxD
        :param: y: Nx1
        :return type: vector Nx1
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
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.cost_rec = np.zeros(self.iterations)
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(X, y)
        for ii in range(self.iterations):
            self.cost_rec[ii] = self.cost()
            print(f"Cost of computation: {self.cost_rec[ii]}")
            #compute the gradient
#            self.hettian = -self.knl.T.dot((self.activation()*(1 - self.activation())*np.eye(X.shape[0])).dot(self.knl)) - self.lamda*self.knl
            self.alpha = self.alpha - self.lr*(self.knl.dot(self.Y)*(np.ones(self.X.shape[0]) - self.activation()) + self.lamda * np.dot(self.knl, self.alpha))
#                self.alpha = self.alpha - self.knl.dot(self.Y - self.activation() - self.lamda*self.alpha)
#            if self.cost_rec[ii] >= self.cost_rec[ii -1]:
#                break
#            else:
#                continue
        return self
    
    def predict(self, X):
        y_pred = np.array(np.sign(1/(1 + np.exp(-np.dot(self.Y * self.alpha, self.kernelize(self.X, X))))))
        for enum, ii in enumerate(y_pred):
            if y_pred[enum] > 0:
                y_pred[enum] = 0
            else:
                y_pred[enum] = 1
        return y_pred
    

class StochKLR(EvalC, loss, Kernels):
    def __init__(self, kernel = None, lamda = None):
        '''
        Kernel Logistic Regression via Gradient Descent. [Solution 1]
        ---------------------------------------------------------------------
        :param: kernel: kernel for computing Gram matrix
        :param: lamda: regularization parameter
        Reference: http://dept.stat.lsa.umich.edu/~jizhu/pubs/Zhu-JCGS05.pdf
        '''
        super().__init__()
        if not kernel:
            kernel = 'rbf' #default
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not lamda:
            lamda = 1e-5
            self.lamda = lamda
        else:
            self.lamda = lamda
        return
    
    def activation(self, kernel, alpha):
        '''
        return binary array
        '''
        return 1/(1+ np.exp(-kernel.dot(alpha)))
    
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
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2)
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
        
    def y_i(self, y):
        '''
        :param: y: Nx1
        '''
        return np.outer(y, y)
    
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
    
    def kernelsamp(self, X):
        '''
        :return type: kernel
        '''
        return self.kernelize(X, X)
    
    def cost(self, kenl, alpha, Y):
        '''
        return: cost
         loss as seen in paper 1/len(self.Y)*np.ones(self.X.shape[0]).T.dot(1 + np.exp(-self.knl.dot(self.Y)))
        '''
        
        return -1/len(Y)*(Y.T.dot(np.dot(kenl, alpha)) - np.sum(np.log(1 + np.exp(np.dot(kenl, alpha)))) -\
                .5*self.lamda*np.dot(alpha.T, np.dot(alpha, kenl)))

        
    def fit(self, X, y, lr:float = None, iterations:int = None):
        '''
        :param: X: NxD
        :param: y: Nx1
        :return type: vector Nx1
        '''
        self.X = X
        self.Y = y
        if not lr:
            lr = 1e-2
            self.lr = lr
        else:
            self.lr = lr
        if not iterations:
            iterations = 5
            self.iterations = iterations
        else:
            self.iterations = iterations
        self.cost_rec = np.zeros(self.iterations)
        self.alpha, self.alpha_i_s, self.y_i_s,  self.knl = self.alpha_y_i_kernel(X, y)
        ylen = len(self.Y)
        for ii in range(self.iterations):
            self.sampledCost = []
            for ij in range(ylen):
                random_samples = np.random.randint(1, ylen)
                X_samp = self.X[:random_samples]
                Y_samp = self.Y[:random_samples]
                self.alphasample = self.alpha[:random_samples]
                self.knlsample = self.kernelsamp(X_samp)
                self.alphasample = self.alphasample - self.lr*(self.knlsample.dot(Y_samp)*(np.ones(X_samp.shape[0]) - self.activation(self.knlsample, self.alphasample)) + self.lamda * np.dot(self.knlsample, self.alphasample))
                self.alpha[:random_samples] = self.alphasample
                self.sampledCost.append(self.cost(self.knlsample, self.alphasample, Y_samp))
                if self.sampledCost[ij] >= self.sampledCost[ij -1]:
                    break
                else:
                    continue
            self.cost_rec[ii] = np.sum(self.sampledCost)
            print('*'*40)
            print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
            if self.cost_rec[ii] == np.inf:
                break
        return self
    
    def predict(self, X):
        y_pred = np.array(np.sign(1/(1 + np.exp(-np.dot(self.Y * self.alpha, self.kernelize(self.X, X))))))
        for enum, ii in enumerate(y_pred):
            if y_pred[enum] > 0:
                y_pred[enum] = 0
            else:
                y_pred[enum] = 1
        return y_pred
    
    
#%% Test
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
#from sklearn.model_selection import train_test_split
#X, y = make_moons(1000, noise = .05)
#X, y = make_blobs(n_samples=1000, centers=2, n_features=2)
#X, y = make_circles(n_samples = 1000, noise = .05, factor=.5)
#X, y = make_classification(n_samples=1000,n_features=20)
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
#klrmodel = KLR(kernel='rbf').fit(X_train, Y_train)
#klrmodel.predict(X_test)
#klrmodel.summary(Y_test, klrmodel.predict(X_test), klrmodel.alpha)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = klrmodel.predict(X_test), cmap = 'coolwarm')       
#
#plt.plot(np.arange(5), klrmodel.cost_rec)
##%%
#klrmodel_irls = StochKLR(kernel='rbf').fit(X_train, Y_train)
#klrmodel_irls.predict(X_test)
#klrmodel_irls.summary(Y_test, klrmodel_irls.predict(X_test), klrmodel_irls.alpha)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = klrmodel_irls.predict(X_test), cmap = 'coolwarm')       
#plt.plot(np.arange(10), klrmodel_irls.cost_rec)
