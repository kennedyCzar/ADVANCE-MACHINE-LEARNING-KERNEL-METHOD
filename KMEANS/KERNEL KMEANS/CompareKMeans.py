#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:25:08 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_checkerboard, make_swiss_roll
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.datasets import make_moons, make_biclusters
from kmeans import kMeans
from kernelkmeans import kkMeans
import time
import os
np.random.seed(1000)
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 200

samples = 1000
dfmoon, ymoon = make_moons(n_samples=samples, noise=.05)
dfcircle, ycircle = make_circles(n_samples = samples, noise = .10, factor=.05)
dfclass, yclass = make_classification(n_samples = samples, n_features=20)
dfiris, yiris = load_iris().data, load_iris().target
plt.scatter(dfiris[:, 0], dfiris[:, 1], c = yiris)

#%%
kernels  = ['linear', 'rbf', 'sigmoid', 'polynomial',
            'linrbf', 'rbfpoly', 'etakernel', 'laplace']
data_name = {'moon': (dfmoon, ymoon), 'circle': (dfcircle, ycircle), 'class': (dfclass, yclass), 'iris': (dfiris, yiris)}

              
kernel_outcome = {'linear': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []}, 
                             'rbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []}, 
                             'sigmoid': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []}, 
                             'polynomial': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []},
                             'linrbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []},
                             'rbfpoly': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []},
                             'etakernel': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []},
                             'laplace': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'randind': []}}

for p, q in data_name.items():
    for ii in kernels:
        start = time.time()
        if p == 'moon':
            gamma = 1
            d = 3
            kmeans = kkMeans(k = 2, kernel = ii, gamma = gamma).fit_predict(q[0])
            kernel_outcome[ii]['acc'].append(kmeans.accuracy(q[1], kmeans.clusters))
            kernel_outcome[ii]['prec'].append(kmeans.precision(q[1], kmeans.clusters))
            kernel_outcome[ii]['rec'].append(kmeans.recall(q[1], kmeans.clusters))
            kernel_outcome[ii]['f1'].append(kmeans.f1(q[1], kmeans.clusters))
            kernel_outcome[ii]['randind'].append(kmeans.rand_index_score(kmeans.clusters, q[1]))
        elif p == 'circle':
            gamma = 1
            d = 3
            kmeans = kkMeans(k = 2, kernel = ii, gamma = gamma).fit_predict(q[0])
            kernel_outcome[ii]['acc'].append(kmeans.accuracy(q[1], kmeans.clusters))
            kernel_outcome[ii]['prec'].append(kmeans.precision(q[1], kmeans.clusters))
            kernel_outcome[ii]['rec'].append(kmeans.recall(q[1], kmeans.clusters))
            kernel_outcome[ii]['f1'].append(kmeans.f1(q[1], kmeans.clusters))
            kernel_outcome[ii]['randind'].append(kmeans.rand_index_score(kmeans.clusters, q[1]))
        elif p == 'class':
            gamma = 1
            d = 2
            kmeans = kkMeans(k = 2, kernel = ii, gamma = gamma).fit_predict(q[0])
            kernel_outcome[ii]['acc'].append(kmeans.accuracy(q[1], kmeans.clusters))
            kernel_outcome[ii]['prec'].append(kmeans.precision(q[1], kmeans.clusters))
            kernel_outcome[ii]['rec'].append(kmeans.recall(q[1], kmeans.clusters))
            kernel_outcome[ii]['f1'].append(kmeans.f1(q[1], kmeans.clusters))
            kernel_outcome[ii]['randind'].append(kmeans.rand_index_score(kmeans.clusters, q[1]))
        elif p == 'iris':
            gamma = 0.1
            d = 2
            kmeans = kkMeans(k = 3, kernel = ii, gamma = gamma).fit_predict(q[0])
            kernel_outcome[ii]['acc'].append(kmeans.accuracy(q[1], kmeans.clusters))
            kernel_outcome[ii]['prec'].append(kmeans.precision(q[1], kmeans.clusters))
            kernel_outcome[ii]['rec'].append(kmeans.recall(q[1], kmeans.clusters))
            kernel_outcome[ii]['f1'].append(kmeans.f1(q[1], kmeans.clusters))
            kernel_outcome[ii]['randind'].append(kmeans.rand_index_score(kmeans.clusters, q[1]))
        end = time.time() - start
        kernel_outcome[ii][f'{p}'] = kmeans.clusters
        kernel_outcome[ii]['time'].append(end)
        
#%%

s = .5
color = 'coolwarm_r'
fig, ax = plt.subplots(4, 9, figsize=(12, 4),gridspec_kw=dict(hspace=0, wspace=0),
                       subplot_kw={'xticks':[], 'yticks':[]})

ax[0, 0].scatter(dfcircle[:, 0], dfcircle[:, 1], c = ycircle, s = s, cmap = color)
ax[1, 0].scatter(dfmoon[:, 0], dfmoon[:, 1], c = ymoon, s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 0].scatter(dfclass[:, 0], dfclass[:, 1], c = yclass, s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 0].scatter(dfiris[:, 0], dfiris[:, 1], c = yiris, s = s, cmap = color)
#--linear
ax[0, 1].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['linear']['circle'], s = s, cmap = color)
ax[1, 1].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['linear']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 1].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['linear']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 1].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['linear']['iris'], s = s, cmap = color)
#--linear
ax[0, 2].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['rbf']['circle'], s = s, cmap = color)
ax[1, 2].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['rbf']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 2].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['rbf']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 2].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['rbf']['iris'], s = s, cmap = color)

#--linear
ax[0, 3].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['sigmoid']['circle'], s = s, cmap = color)
ax[1, 3].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['sigmoid']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 3].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['sigmoid']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 3].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['sigmoid']['iris'], s = s, cmap = color)
    #--linear
ax[0, 4].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['polynomial']['circle'], s = s, cmap = color)
ax[1, 4].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['polynomial']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 4].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['polynomial']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 4].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['polynomial']['iris'], s = s, cmap = color)
#--linear
ax[0, 5].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['laplace']['circle'], s = s, cmap = color)
ax[1, 5].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['laplace']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 5].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['laplace']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 5].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['laplace']['iris'], s = s, cmap = color)
#--linear
ax[0, 6].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['linrbf']['circle'], s = s, cmap = color)
ax[1, 6].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['linrbf']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 6].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['linrbf']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 6].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['linrbf']['iris'], s = s, cmap = color)
#--linear
ax[0, 7].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['rbfpoly']['circle'], s = s, cmap = color)
ax[1, 7].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['rbfpoly']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 7].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['rbfpoly']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 7].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['rbfpoly']['iris'], s = s, cmap = color)
    
#--linear
ax[0, 8].scatter(dfcircle[:, 0], dfcircle[:, 1], c = kernel_outcome['etakernel']['circle'], s = s, cmap = color)
ax[1, 8].scatter(dfmoon[:, 0], dfmoon[:, 1], c = kernel_outcome['etakernel']['moon'], s = 1, cmap = color)
#for ii in range(1, dfclass.shape[1]):
ax[2, 8].scatter(dfclass[:, 0], dfclass[:, 1], c = kernel_outcome['etakernel']['class'], s = s, cmap = color)
#for ii in range(1, dfiris.shape[1]):
ax[3, 8].scatter(dfiris[:, 0], dfiris[:, 1], c = kernel_outcome['etakernel']['iris'], s = s, cmap = color)
    


ax[0, 0].set_title('original')
ax[0, 1].set_title('linear')
ax[0, 2].set_title('rbf')
ax[0, 3].set_title('poly')
ax[0, 4].set_title('sigmoid')
ax[0, 5].set_title('laplace')
ax[0, 6].set_title('rbfpoly')
ax[0, 7].set_title('linrbf')
ax[0, 8].set_title('etakernel')
ax[0, 0].set_ylabel('Circle')
ax[1, 0].set_ylabel('Moons')
ax[2, 0].set_ylabel('classifciation')
ax[3, 0].set_ylabel('Iris')
fig.set_tight_layout(True)

#%% Image compression




