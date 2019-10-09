#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:48:41 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.utils import EvalC
from Utils.Loss import loss
from Utils.kernels import Kernels

class linearSVDD(EvalC, loss, Kernels):
    def __init__(self):
        super().__init__()
        return
        
    def cost(self, X, y, beta):
        '''
        :param: X: NxD
        :param: Dx1
        :param: beta: Nx1
        '''
        return 
    
    def fit(self):
        '''
        '''
        return
    
    def predict(self):
        '''
        '''
        return
    