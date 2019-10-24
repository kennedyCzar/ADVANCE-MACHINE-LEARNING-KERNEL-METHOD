#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class Kernels:
    '''
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into a subspace, making it easy to
    almost accurately classify our data as if it were
    still in linear space.
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        else:
            c = c
        return x1.dot(x2.T) + c
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
        
    @staticmethod
    def sigmoid(x1, x2, gamma = None, c = None):
        '''
        logistic or sigmoid kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        if not c:
            c = 1
        return np.tanh(gamma * x1.dot(x2.T) + c)
    
    @staticmethod
    def polynomial(x1, x2, d = None):
        '''
        polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: d: polynomial degree
        :return type: kernel(Gram) matrix
        '''
        if not d:
            d = 3
        else:
            d = d
        return (x1.dot(x2.T))**d
    
    @staticmethod
    def cosine(x1, x2):
        '''
        Cosine kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        
        return (x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1))
    
    @staticmethod
    def correlation(x1, x2, gamma = None):
        '''
        Correlation kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        return np.exp((x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1)) - gamma)
    
    @staticmethod
    def linrbf(x1, x2, gamma = None, op = None):
        '''
        MKL: Lineaar + RBF kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        if not op:
            op = 'add' #add seems like the best performning here
        else:
            op = op
        if op == 'multiply':
            return Kernels.linear(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.linear(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.linear(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.linear(x1, x2) - Kernels.rbf(x1, x2, gamma))

    @staticmethod
    def rbfpoly(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        if not d:
            d = 3
        else:
            d = d
        if not op:
            op = 'add'
        else:
            op = op
        if op == 'multiply':
            return Kernels.polynomial(x1, x2, d) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.polynomial(x1, x2, d) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.polynomial(x1, x2, d) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.polynomial(x1, x2, d) - Kernels.rbf(x1, x2, gamma))
        
    @staticmethod
    def rbfcosine(x1, x2, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1/x1.shape[1]
        else:
            gamma = gamma
        if not op:
            op = 'add'
        else:
            op = op
        if op == 'multiply':
            return Kernels.cosine(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.cosine(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.cosine(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.cosine(x1, x2) - Kernels.rbf(x1, x2, gamma))
        
        
        
        