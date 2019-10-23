#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:41:45 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
import copy
from Utils.kernels import Kernels

class kkmeans(Kernels):
    def __init__(self, k = None,kernel = None):
        super().__init__()
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
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
        
    def distance(self, x, nu):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun distance matrix
        '''
       
        return self.kernelize(x, x) + self.kernelize(nu, nu) - 2*self.kernel(x, nu)
    
    
    def fit(self, X):
        '''
        :param: X: NxD
        '''
        self.X = X
        #random sample
        N, D = X.shape
        #randomly initialize k centroids
        self.nu = X[np.random.choice(N, self.k, replace = False)]
        self.prev_c = np.zeros((self.k, D))
        self.cluster = np.zeros(X.shape[0])
        '''iterate by checking to see if new centroid
        of new center is same as old center, then we reached an
        optimum point.
        '''
        while self.distance(self.X, self.nu) != 0:
            for ii in range(X.shape[0]):
                self.distance_matrix = self.distance(self.X[ii], self.nu)
                self.cluster[ii] = np.argmin(self.distance_matrix)
            self.prev_c = copy.deepcopy(self.nu)
            for ij in range(self.k):
                #mean of the new found clusters
                self.newPoints = [X[ii] for ii in range(X.shape[0]) if self.cluster[ii] == ij]
                self.nu[ij] = np.mean(self.newPoints, axis = 0)
        return self
    
    def predict(self, X):
        '''
        :param: X: NxD
        :return type: labels
        '''
        pred = np.zeros(X.shape[0])
        #compare new data to final centroid
        for ii in range(X.shape[0]):
            distance_matrix = self.distance(X[ii], self.nu)
            pred[ii] = np.argmin(distance_matrix)
        return pred
    

#%% Testing
        
    
kernelkmns = kkmeans().fit(X)
