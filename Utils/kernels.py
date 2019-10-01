#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class Kernels:
    '''Docstring
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into another subspace
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2):
        return x1.T.dot(x2)
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        if not gamma:
            gamma = 1.0/x1.shape[1]
            
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 1)**2)
        
    @staticmethod
    def sigmoid(x1, x2, gamma = None, C = None):
        if not gamma:
            gamma = .05
        if not C:
            C = 1
        return np.tanh(gamma * x1.T.dot(x2))