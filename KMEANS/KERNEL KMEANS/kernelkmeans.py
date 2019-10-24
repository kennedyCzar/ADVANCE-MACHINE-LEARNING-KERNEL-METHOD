#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:41:45 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels

class kkmeans(Kernels):
    def __init__(self, k = None,kernel = None):
        super().__init__()
        if not k:
            k = 3
            self.k = k
        else:
            self.k = k
        if not kernel:
            kernel = 'linear'
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
        
    def distance(self, x, nu, axs = 1):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun distance matrix
        '''
       
        return np.linalg.norm(x - nu, axis = axs)
    
    
    def fit(self, X, iteration = None):
        '''
        :param: X: NxD
        '''
        if not iteration:
            iteration = 20
            self.iteration = iteration
        else:
            self.iteration = iteration
        N, D = X.shape
        self.cluster = np.random.randint(low = 0, high = 2, size = N)
        self.cost_rec = np.zeros(self.iteration)
        for self.iter in range(self.iteration):
            self.kappar = np.tile(self.kernelize(X, X).diagonal().reshape((-1, 1)), self.k)
            self.z_i = np.bincount(self.cluster)
            for self.c in range(self.k):
                self.kappar[:, self.c] = self.kappar[:, self.c] + np.sum((self.kernelize(X, X))[self.cluster == self.c][:, self.cluster == self.c])/\
                                            (self.z_i[self.c]**2) - 2*np.sum(self.kernelize(X, X)[:, self.cluster == self.c], axis = 1)/self.z_i[self.c]
        return self
    
    def predict(self):
        '''
        :param: X: NxD
        :return type: labels
        '''
        
        return np.argmin(self.kappar, axis = 1)
    

#%% Testing
from sklearn.datasets import make_circles, make_blobs
X, y = make_blobs(1000, )
import matplotlib.pyplot as plt  

trans = [[.6, -.6], [-.4, .8]]
X = X.dot(trans)
kernelkmns = kkmeans(kernel='polynomial').fit(X)

plt.scatter(X[:, 0], X[:, 1], c = kernelkmns.predict())




