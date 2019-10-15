#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:19:22 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels

class kPCA(Kernels):
    def __init__(self, k = None, kernel = None):
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
    
    def explained_variance_(self):
        '''
        :Return: explained variance.
        '''
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse = True)[:self.k]]
        return self.explained_variance
    
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
        
    def fit(self, X):
        '''
        param: X: NxD
        '''
        self.X = X
        #normalized kernel
        self.normKernel = self.kernelize(X, X) - 2*1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(self.kernelize(X, X)) + \
                            1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(np.dot(1/X.shape[0]*np.ones((X.shape[0], X.shape[0])), self.kernelize(X, X)))
        self.eival, self.eivect = np.linalg.eig(self.normKernel)
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        #sort eigen values and return explained variance
        self.explained_variance = self.explained_variance_()
        #return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        return self
    
    
    def fit_transform(self):
        '''
        Return: transformed data
        '''
        return self.kernelize(self.X, self.X).dot(self.eivect)
    
#%% Testing
        
kpca = kPCA(kernel='linear').fit(X)
kpca.explained_variance
newX = kpca.fit_transform()
plt.scatter(newX[:, 0], newX[:, 1], c = y)
