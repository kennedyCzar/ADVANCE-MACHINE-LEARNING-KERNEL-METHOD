#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:34:58 2019

@author: kenneth
"""
import numpy as np


class Perceptron(object):
    def __init__(self, activation = False, norm = None, lamda = None):
        self.activation = activation
        self.norm = norm
        self.lamda = lamda
        if self.norm == 'l2':
            if self.lamda is None:
                self.lamda = 0.001
            else:
                self.lamda = lamda
        elif self.norm == 'l1':
            if self.lamda is None:
                self.lamda = 0.001
            else:
                self.lamda = lamda
        elif self.norm == 'ElasticNet':
            if self.lamda is None:
                self.lamda = 0.001
            else:
                self.lamda = lamda
        return
    
    @staticmethod
    def sigmoid(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or 1
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
    
    def cost(self, X, Y, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        if not self.norm:
            if not self.activation or self.activation == 'sigmoid':
                return -(1/len(Y)) * np.sum((Y*np.log(Perceptron.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Perceptron.sigmoid(X, beta))))
            elif self.activation == 'relu':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))))))
            elif self.activation == 'tanh':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.tanh(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))))))
        elif self.norm == 'l2':
            if not self.activation or self.activation == 'sigmoid':
                return -(1/len(Y)) * (np.sum((Y*np.log(Perceptron.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Perceptron.sigmoid(X, beta)))) + ((self.lamda/2)*np.sum(np.square(beta))))
            elif self.activation == 'relu':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))))) + \
                                ((self.lamda/2)*np.sum(np.square(beta))))
            elif self.activation == 'tanh':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.tanh(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))))) + \
                                ((self.lamda/2)*np.sum(np.square(beta))))
        elif self.norm == 'l1':
            if not self.activation or self.activation == 'sigmoid':
                return -(1/len(Y)) * (np.sum((Y*np.log(Perceptron.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Perceptron.sigmoid(X, beta)))) + ((self.lamda)*np.sum(beta)))
            elif self.activation == 'relu':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))))) + \
                                ((self.lamda)*np.sum(beta)))
            elif self.activation == 'tanh':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.tanh(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))))) + \
                                ((self.lamda)*np.sum(beta)))
        elif self.norm == 'ElasticNet':
            if not self.activation or self.activation == 'sigmoid':
                return -(1/len(Y)) * (np.sum((Y*np.log(Perceptron.sigmoid(X, beta))) + ((1 - Y)*np.log(1 - Perceptron.sigmoid(X, beta)))) + ((self.lamda/2)*np.sum(np.square(beta))) + ((self.lamda)*np.sum(beta)))
            elif self.activation == 'relu':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.relu(X, beta)))))) + \
                                ((self.lamda/2)*np.sum(np.square(beta))) + ((self.lamda)*np.sum(beta)))
            elif self.activation == 'tanh':
                return -(1/len(Y)) * (np.sum((Y*np.where(np.log(Perceptron.tanh(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))) + \
                               ((1 - Y)*np.log(1 - np.where(np.log(Perceptron.relu(X, beta)) == 1, 0.99, np.log(Perceptron.tanh(X, beta)))))) + \
                                ((self.lamda/2)*np.sum(np.square(beta))) + ((self.lamda)*np.sum(beta)))
            
    def fit(self, X, Y, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.beta = np.zeros(X.shape[1]).reshape(-1, 1)
        self.cost_rec = np.zeros(self.iterations)
        self.beta_rec = np.zeros((self.iterations, X.shape[1]))
        #--Unregularized
        if not self.norm:
            if not self.activation or self.activation == 'sigmoid':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - Perceptron.sigmoid(X_samp, self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'relu':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - Perceptron.relu(X_samp, self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'tanh':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * X_samp.T.dot(Y_samp - Perceptron.tanh(X_samp, self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
        #--l2
        elif self.norm == 'l2':
            if not self.activation or self.activation == 'sigmoid':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.sigmoid(X_samp, self.beta)) +\
                                                 ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'relu':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.relu(X_samp, self.beta)) +\
                                                 ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'tanh':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.tanh(X_samp, self.beta)) +\
                                                 ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
        #--l1
        elif self.norm == 'l1':
            if not self.activation or self.activation == 'sigmoid':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.sigmoid(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'relu':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.relu(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'tanh':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.tanh(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
        #--Elastic Net
        else:
            if not self.activation or self.activation == 'sigmoid':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.sigmoid(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)) + ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'relu':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.relu(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)) + ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
            elif self.activation == 'tanh':
                ylen = len(Y)
                for ii in range(self.iterations):
                    #compute stochastic gradient
                    sampledCost = 0
                    for ij in range(ylen):
                        random_samples = np.random.randint(1, ylen)
                        X_samp = X[:random_samples]
                        Y_samp = Y[:random_samples]
                        self.beta = self.beta + (1/len(Y_samp)) *(self.alpha) * (X_samp.T.dot(Y_samp - Perceptron.tanh(X_samp, self.beta)) +\
                                                 (self.lamda*np.sign(self.beta)) + ((self.lamda/len(Y))*self.beta))
                        self.beta_rec[ii, :] = self.beta.T
                        sampledCost += self.cost(X_samp, Y_samp, self.beta)
                    self.cost_rec[ii] = sampledCost
                    print('*'*40)
                    print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
                return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        y_pred = np.zeros(X.shape[0])
        if not self.activation or self.activation == 'sigmoid':
            for ii in range(len(y_pred)):
                if Perceptron.sigmoid(X[ii], self.beta) >= 0.5:
                    y_pred[ii] = 1
                elif Perceptron.sigmoid(X[ii], self.beta) < 0:
                    y_pred[ii] = 0
            return y_pred
        elif self.activation == 'relu':
            for ii in range(len(y_pred)):
                if Perceptron.relu(X[ii], self.beta) >= 0:
                    y_pred[ii] = 1
                elif Perceptron.relu(X[ii], self.beta) < 0:
                    y_pred[ii] = 0
            return y_pred
        elif self.activation == 'tanh':
            for ii in range(len(y_pred)):
                if Perceptron.tanh(X[ii], self.beta) > 0:
                    y_pred[ii] = 1
                elif Perceptron.tanh(X[ii], self.beta) < 0:
                    y_pred[ii]
            return  y_pred
        
        
#%%
import matplotlib.pyplot as plt
#--nonregularized
pctron = Perceptron().fit(X_train, Y_train.reshape(-1, 1), 0.1, 100)
#--regularized
pctron = Perceptron(activation='relu', norm = 'l2').fit(X_train, Y_train.reshape(-1, 1), 0.1, 100)
pctron.predict(X_test)
plt.plot(np.arange(pctron.iterations), pctron.cost_rec)
















