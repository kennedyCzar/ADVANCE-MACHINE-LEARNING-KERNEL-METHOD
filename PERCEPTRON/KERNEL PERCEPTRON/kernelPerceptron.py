#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:51:54 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss

class kperceptron(Kernels, EvalC, loss):
    def __init__(self, kernel = None, gamma = None, d = None):
        '''Kernel Perceptron Algorithm
        :Arguments:
            :kernels: Specified mercer's kernel to compute 
                        inner products
        :Reference: http://jmlr.csail.mit.edu/papers/volume10/orabona09a/orabona09a.pdf
        '''
        super().__init__()
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        if not gamma:
            gamma = 5
            self.gamma = gamma
        else:
            self.gamma = gamma
        if not d:
            d = 10
            self.d = d
        else:
            self.d = d
        return
    
        return
    
    def kernelize(self, x1, x2):
        '''
        :params: X: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2, gamma = self.gamma)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2, d = self.d)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2, gamma = self.gamma)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2, gamma = self.gamma)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
        
    def pred_update(self, x1, x2, alpha):
        '''
        :param: X: NxD
        :param: y: dx1
        '''
        return 1 if np.dot(alpha, self.kernelize(x1, x2)) >0 else 0
    
    def fit(self, X, y):
        '''
        :Arguments:
            :X: NxD feature space
            :y: Dx1 
        :Return type:
            fitted components in self
        '''
        self.X = X
        self.Y = y
        N, D = self.X.shape
        ypred = np.zeros(N)
        self.alpha = np.zeros(N)
        self.s_t = []
        for enum, (x_i, y_i) in enumerate(zip(self.X, self.Y)):
            ypred[enum] = self.pred_update(x_i, x_i, self.alpha[enum])
            if ypred[enum] != y_i:
                self.alpha[enum] = self.alpha[enum] + y_i*self.kernelize(x_i, x_i)
                self.s_t.append(x_i)
        self.alpha = self.alpha[self.alpha != 0] #support set hypothesis
        self.s_t = np.array(self.s_t) #support set
        return self
    
    def predict(self, X):
        '''
        :param: X: NxD
        '''
        return np.sign(np.dot(self.alpha, self.kernelize(self.s_t, X)))
    
    
#%% Testing Kernel Perceptron
color = 'coolwarm_r'
kpctron = kperceptron(kernel = 'linear').fit(X_train, Y_train)
#pred = kpctron.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = kpctron.predict(X_test), s = 1, cmap = color)
