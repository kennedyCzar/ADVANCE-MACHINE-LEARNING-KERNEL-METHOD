#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:19:27 2019

@author: kenneth
"""

import numpy as np

class loss:
    '''Loss function used in Machine Learning/Deep learning for minimization
        or maximations problems.
        
     Parameters
     ----------
     None
     
     Attributes
     ----------
     None
    '''
    def __init__(self):
        return
    
    @staticmethod
    def sigmoid(X, beta):
        '''
        Also known as the logistic loss,
        especially because it is used 
        for logistic regression
        :params: X: traing data at ith iteration
        :return: 0 or 1
        '''
        return 1/(1  + np.exp(-(np.dot(X, beta))))
    
    @staticmethod
    def hinge(X, y, beta, b):
        '''
        Hinge loss function
        is used for Support vector machines (SVM)
        :params: X: traing data at ith iteration
        :return: 0 or max margin
        '''
        return np.maximum(0, 1 - y*(np.dot(X, beta) + b))
    
    @staticmethod
    def relu(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: max, 0
        '''
        return np.maximum(np.dot(X, beta), 0)
    
    @staticmethod
    def leakyrelu(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: max, 0.1z
        '''
        return np.maximum(np.dot(X, beta), 0.1*np.dot(X, beta))
    
    @staticmethod
    def square(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: square loss
        '''
        return .5*(np.dot(X, beta) + 1)
    
    @staticmethod
    def exponential(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: exponential
        '''
        return np.exp(2*np.dot(X, beta))/(1 + 2*np.dot(X, beta))
    
    @staticmethod
    def tanh(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: 0 or tanh(X, beta)
        '''
        return (np.exp(np.dot(X, beta)) - np.exp(-np.dot(X, beta)))/\
                (np.exp(np.dot(X, beta)) + np.exp(-np.dot(X, beta)))
                
    @staticmethod
    def softplus(X, beta):
        '''
        :params: X: traing data at ith iteration
        :return: log(1 + e^x)
                NOTE that log1p is the reverse of exp(x) - 1
        '''
        return np.log1p(np.dot(X, beta))
                

