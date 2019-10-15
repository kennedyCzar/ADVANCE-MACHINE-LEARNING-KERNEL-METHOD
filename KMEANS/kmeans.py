#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:40:59 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
import copy

class kMeans:
    def __init__(self, k = None):
        '''
        :param: k: number of clusters
        '''
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    @staticmethod
    def distance(x, nu, axs = 1):
        '''
        :param: x: datapoint
        :param: nu: mean
        :retrun distance matrix
        '''
       
        return np.linalg.norm(x - nu, axis = axs)
    
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
                self.distance_matrix = kMeans.distance(self.X[ii], self.nu)
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
            distance_matrix = kMeans.distance(X[ii], self.nu)
            pred[ii] = np.argmin(distance_matrix)
        return pred
    
    
#%% Testing
from sklearn.datasets import make_blobs, make_moons, make_circles
X, y = make_circles(1000, noise = .07, factor = .5)
import matplotlib.pyplot as plt
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
new_x = np.array([[1, 5], [10, 6], [10, 3]])
kmns = kMeans(k=2).fit(X)
pred = kmns.predict(new_x)
plt.scatter(X[:, 0], X[:, 1], c = kmns.cluster)
plt.scatter(kmns.nu[:, 0], kmns.nu[:, 1], marker = '.')
#%% Kmeans from Sklearn

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
plt.scatter(X[:, 0], X[:, 1], c = kmeans.labels_)
kmeans.predict(new)
    










