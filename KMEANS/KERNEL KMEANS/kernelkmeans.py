#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:41:45 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC

class kkMeans(EvalC, Kernels):
    def __init__(self, k = None, kernel = None, gamma = None, d = None):
        '''
        :param: k: number of clusters
        :param: kernel
        
        '''
        super().__init__()
        if not gamma:
            gamma = 5
            self.gamma = gamma
        else:
            self.gamma = gamma
        if not d:
            d = 3
            self.d = d
        else:
            self.d = d
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
    
    #compute suqared kernel distance
    def distance(self, kappa):
        self.dist_w = np.zeros(self.k)
        c_k_init = np.ones(kappa.shape[0])
        for ii in range(self.k):
            self.c_k_indices = self.clusters == ii
            self.c_k = np.sum(c_k_init[self.c_k_indices])
            self.c_k_squared = np.square(self.c_k)
            self.kappa_ii = kappa[self.c_k_indices][:, self.c_k_indices]
            self.distance_ii[:, ii] += np.sum(self.kappa_ii)/self.c_k_squared - 2*\
                                    np.sum(kappa[:, self.c_k_indices], axis = 1)/self.c_k
        return self.distance_ii
           
    #fit and return cluster labels
    def fit_predict(self, X, iteration = None, halt = None):
        '''
        :param: X: NxD Feature
        :param: iteration: 100
        :param: tolerance: 1e-3 default
        
        '''
        if not halt:
            halt = 1e-3
            self.halt = halt
        else:
            self.halt = halt
        if not iteration:
            iteration = 100
            self.iteration = iteration
        else:
            self.iteration = iteration
        self.X = X
        N, D = X.shape
        self.distance_ii = np.zeros((N, self.k))
        self.kx = self.kernelize(self.X, self.X)
        self.clusters = np.random.randint(self.k, size = N)
        '''iterate by checking to see if new and previous cluster
        labels are similar. If they are, algorithm converges and halts..
        '''
        for ii in range(self.iteration):
            #compute distance
            self.distance_k = self.distance(self.kx)
            self.prev_clusters = self.clusters
            self.clusters = self.distance_k.argmin(axis=1)
            if 1 - float(np.sum((self.clusters - self.prev_clusters) == 0)) / N < self.halt:
                break
        return self

    def rand_index_score(self, clusters, classes):
        '''Compute the RandIndex
        :param: Clusters: Cluster labels
        :param: classses: Actual class
        :returntype: float
        '''
        from scipy.special import comb
        tp_fp = comb(np.bincount(clusters), 2).sum()
        tp_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                 for i in set(clusters))
        fp = tp_fp - tp
        fn = tp_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        self.randindex = (tp + tn) / (tp + fp + fn + tn)
        return self.randindex
    
##%% Testing
#from sklearn.datasets import make_circles, make_moons, load_iris
#X, y = load_iris().data, load_iris().target
#X, y = make_moons(n_samples=samples, noise=.05)
#X, y = make_circles(n_samples = 1000, noise = .10, factor=.05)
#kkmeans = kkMeans(k = 3, kernel = 'linear', gamma=1).fit_predict(X)
#plt.scatter(X[:, 0], X[:, 1], c = kkmeans.clusters)

