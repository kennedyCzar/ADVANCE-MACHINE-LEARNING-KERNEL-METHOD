#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:50:07 2021

@author: ifeanyi.ezukwoke
"""
import tensorflow as tf

class PCA:
    def __init__(self, k:int = None):
        '''Principal Component Analysis implementation in Tensorflow 2.0

        Parameters
        ----------
        k : int, optional
            Number of components. The default is 2 if None.

        Returns
        -------
        None.

        '''
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
            
    
    def explained_variance_(self, eival):
        '''Explained variance
        

        Parameters
        ----------
        eival : list of tensors
            Sorted k-eigen values.

        Returns
        -------
        dictionary
            explained_variance of individual components 
            and toatl explained variance.

        '''
        self.eival = eival
        self.total_eigenvalue = tf.math.reduce_sum(self.eival)
        self.evar = [x/self.total_eigenvalue*100 for x in eival] #explained variance of the individual eigen values
        self.ev = {'explained_variance': self.evar, 'total_ev': tf.math.reduce_sum(self.evar)} 
        return self.ev
        
    
    def fit(self, X):
        '''
        

        Parameters
        ----------
        X : tf.tensor
            Dataset.

        Returns
        -------
        fit model
            tensorflow containing attributes.

        '''
        self.X = tf.constant(X)
        n, m = self.X.shape
        self.Xcopy = X
        self.mean = tf.math.reduce_mean(self.X,  axis = 0)
        #centered mean
        self.X = self.X - self.mean
        #covariance
        self.cov = (1/m) * tf.tensordot(tf.transpose(self.X), self.X, axes=1) #np.dot(self.X.T, self.X)
        self.eival, self.eivect = tf.linalg.eig(self.cov) #np.linalg.eig(self.cov)
        self.eival, self.eivect = tf.math.real(self.eival), tf.math.real(self.eivect) #eigen values are already sorted in increasing order
        self.sorted_eigval = sorted(self.eival, reverse = True)[:self.k]
        self.sorted_eigen_idx = tf.argsort(self.eival)[::-1][:self.k] #find the indices of the k-sorted eigen values
        self.sorted_eivect = tf.constant(self.eivect.numpy()[:, self.sorted_eigen_idx]) #sorted eigen vectors
        #sort eigen values and return explained variance
        self.explained_variance = self.explained_variance_(self.sorted_eigval)
        #return eigen value and corresponding eigenvectors
        self.components_ = tf.transpose(self.sorted_eivect)
        return self
    
    
    def fit_transform(self):
        '''
        

        Returns
        -------
        tensor
            dimension reduced data.

        '''
        return tf.tensordot(self.X, self.sorted_eivect, axes=1) #self.X.dot(self.eivect)
    
    def inverse_transform(self):
        '''
        Returns
        -------
        tensor
            reconstructed data
        '''
        self.transformed = self.fit_transform() #self.X.dot(self.eivect)
        return tf.tensordot(self.transformed, self.components_, axes=1) + self.mean
    
    
    
#%% Test model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.datasets import load_iris, make_checkerboard, make_swiss_roll
dfiris, yiris = load_iris().data, load_iris().target
x_train = Normalizer().fit_transform(dfiris)

pc = PCA(k = 2).fit(dfiris)
pcc = pc.fit_transform()
inv_trans = pc.inverse_transform().shape #reconstructed X
plt.scatter(pcc[:, 0], pcc[:, 1], c = yiris)
        


