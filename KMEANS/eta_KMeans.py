#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:27:36 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
import copy

class etaMeans:
    def __init__(self, eta = None, k = None):
        '''
        :param: k: number of clusters
        '''
        if not eta:
            eta = 70000000
            self.eta = eta
        else:
            self.eta = eta
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    def distance(self, x, nu, axs = 1):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun distance matrix
        '''
       
        return 1/self.eta *(np.linalg.norm(x - nu, axis = axs)) + self.eta
    
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
        while np.linalg.norm(self.nu - self.prev_c) != 0:
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
from sklearn.datasets import make_blobs, make_moons, make_circles
X, y = make_circles(1000, noise = .07, factor = .5)
import matplotlib.pyplot as plt

new_x = np.array([[1, 5], [10, 6], [10, 3]])
kmns = etaMeans(k=2).fit(X)
pred = kmns.predict(new_x)
plt.scatter(X[:, 0], X[:, 1], c = kmns.cluster)
plt.scatter(kmns.nu[:, 0], kmns.nu[:, 1], marker = '.')

