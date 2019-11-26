#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:27:59 2019

@author: kenneth
"""
import numpy as np


class Perceptron(object):
    def __init__(self):
        return
    
    @staticmethod
    def activation(X, beta):
        '''
        :params: X: train data
        :params: X: weights
        '''
        return 1 if np.sign(np.dot(X, beta)) >0 else 0
    
    @staticmethod
    def sigmoid(X, beta):
        '''Docstring
        :params: X: features N x (M+1)
        :params: beta: weights N x 1
        
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def relu(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or max
        '''
        return np.maximum(np.dot(X, beta), 0)
    
    @staticmethod
    def tanh(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or tanh(X, beta)
        '''
        return (np.exp(np.dot(X, beta)) - np.exp(-np.dot(X, beta)))/\
                (np.exp(np.dot(X, beta)) + np.exp(-np.dot(X, beta)))
                
    @staticmethod
    def cost(y, ypred):
        '''
        :params: y: actual label
        :params: ypred: predicted label
        '''
        return (y - ypred)
        
    def fit(self, X, Y, alpha, iterations):
        '''
        :params: X: train data
        :params: Y: train labels
        :params: alpha: learning rate
        :iterations: number of iterations
        '''
        self.iterations = iterations
        self.alpha = alpha
        self.beta = np.zeros(X.shape[1])
        self.pred = np.zeros(len(Y))
        for ij, (x_i, y_i) in enumerate(zip(X, Y)):
            self.pred[ij] = Perceptron.tanh(X[ij], self.beta)
            if self.pred[ij] != y_i:
                self.beta = self.beta + self.alpha * Perceptron.cost(y_i, self.pred[ij])*x_i
            print(f'{self.beta}')
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        for ii in range(X.shape[0]):
            if Perceptron.activation(X[ii], self.beta) == 1:
                y_pred[ii] = 1
        return y_pred
    
    
    
#%%
pcp = Perceptron().fit(X_train, Y_train, 0.001, 100)

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c = pcp.predict(X_train))
np.sum(Y_test == pcp.predict(X_test))/len(Y_test)
