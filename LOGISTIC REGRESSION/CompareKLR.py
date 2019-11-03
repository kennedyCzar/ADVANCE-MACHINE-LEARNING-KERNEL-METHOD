#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:53:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_checkerboard, make_swiss_roll
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.datasets import make_moons, make_biclusters
from sklearn.model_selection import train_test_split
from kernellogistic import KLR, StochKLR
from logisticregression import Logistic, stochasticLogistic
import time
import os
np.random.seed(1000)
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 200

def data(samples = None):
    if not samples:
        samples = 1000
    else:
        samples = samples
    dfmoon, ymoon = make_moons(n_samples=samples, noise=.05)
    dfblob, yblob = make_blobs(n_samples=samples, centers = 2, n_features=2)
    dfcircle, ycircle = make_circles(n_samples = samples, noise = .05, factor=.5)
    dfclass, yclass = make_classification(n_samples=samples,n_features=20)
    data_name = {'moon': {'data': dfmoon, 'label': ymoon}, 'blob': {'data': dfblob, 'label': yblob}, 
                 'circle': {'data': dfcircle, 'label': ycircle}, 'class': {'data': dfclass, 'label': yclass}}
    
    train_test = {'moon': {'train': [], 'test': []}, 'blob': {'train': [], 'test': []},
                  'circle': {'train': [], 'test': []}, 'class': {'train': [], 'test': []}}
    
    for ii, ij in data_name.items():
        X_train, X_test, Y_train, Y_test = train_test_split(data_name[ii]['data'], data_name[ii]['label'], test_size = 0.3)
        train_test[ii]['train'].append(X_train)
        train_test[ii]['test'].append(X_test)
        train_test[ii]['train'].append(Y_train)
        train_test[ii]['test'].append(Y_test)
    return train_test
        
train_test = data()
        
#%%

kernels  = ['linear', 'rbf', 'sigmoid', 'polynomial',
            'linrbf', 'rbfpoly', 'etakernel', 'laplace']


kernel_outcome = {'linear': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'rbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'sigmoid': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}, 
                             'polynomial': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'linrbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'rbfpoly': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'etakernel': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []},
                             'laplace': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'impr_f1': []}}

for _ in range(3):
    for p, q in train_test.items():
        for ii in kernels:
            if ii == 'linear':
                start = time.time()
                logit = stochasticLogistic(0.1, 10).fit(q['train'][0], q['train'][1])
                end = time.time() - start
                kernel_outcome[ii][f'{p}'] = logit.predict(q['test'][0])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(logit.accuracy(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['prec'].append(logit.precision(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['rec'].append(logit.recall(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['f1'].append(logit.f1(q['test'][1], logit.predict(q['test'][0])))
                kernel_outcome[ii]['impr_f1'].append(logit.fscore(q['test'][1], logit.predict(q['test'][0]), logit.alpha))
            else:
                if p == 'blob':
                    start = time.time()
                    klogit = StochKLR(kernel = ii).fit(q['train'][0], q['train'][1], iterations=50)
                    end = time.time() - start
                    start = time.time()
                    kernel_outcome[ii][f'{p}'] = klogit.predict(q['test'][0])
                    kernel_outcome[ii]['time'].append(end)
                    kernel_outcome[ii]['acc'].append(klogit.accuracy(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['prec'].append(klogit.precision(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['rec'].append(klogit.recall(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['f1'].append(klogit.f1(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['impr_f1'].append(klogit.fscore(q['test'][1], klogit.predict(q['test'][0]), klogit.alpha))
                else:
                    start = time.time()
                    klogit = StochKLR(kernel = ii).fit(q['train'][0], q['train'][1], iterations=10)
                    end = time.time() - start
                    start = time.time()
                    kernel_outcome[ii][f'{p}'] = klogit.predict(q['test'][0])
                    kernel_outcome[ii]['time'].append(end)
                    kernel_outcome[ii]['acc'].append(klogit.accuracy(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['prec'].append(klogit.precision(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['rec'].append(klogit.recall(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['f1'].append(klogit.f1(q['test'][1], klogit.predict(q['test'][0])))
                    kernel_outcome[ii]['impr_f1'].append(klogit.fscore(q['test'][1], klogit.predict(q['test'][0]), klogit.alpha))
        
#%% Visualize dataset
color = 'coolwarm_r'
fig, ax = plt.subplots(4, 1, figsize=(2, 7),gridspec_kw=dict(hspace=0.01, wspace=0.01),
                       subplot_kw={'xticks':[], 'yticks':[]})
fig.set_tight_layout(True)

ax[0].scatter(train_test['moon']['train'][0][:, 0], train_test['moon']['train'][0][:, 1], c = train_test['moon']['train'][1], s = 1, cmap = color)
ax[1].scatter(train_test['blob']['train'][0][:, 0], train_test['blob']['train'][0][:, 1], c = train_test['blob']['train'][1], s = 1, cmap = color)
ax[2].scatter(train_test['circle']['train'][0][:, 0], train_test['circle']['train'][0][:, 1], c = train_test['circle']['train'][1], s = 1, cmap = color)
ax[3].scatter(train_test['class']['train'][0][:, 0], train_test['class']['train'][0][:, 1], c = train_test['class']['train'][1], s = 1, cmap = color)

ax[0].set_ylabel('Moons')
ax[1].set_ylabel('Blob')
ax[2].set_ylabel('Circle')
ax[3].set_ylabel('Classifciation')

#%% Multivisualization

s = .5
color = 'coolwarm_r'
fig, ax = plt.subplots(4, 9, figsize=(12, 4),gridspec_kw=dict(hspace=0, wspace=0),
                       subplot_kw={'xticks':[], 'yticks':[]})

ax[0, 0].scatter(train_test['moon']['train'][0][:, 0], train_test['moon']['train'][0][:, 1], c = train_test['moon']['train'][1], s = 1, cmap = color)
ax[1, 0].scatter(train_test['blob']['train'][0][:, 0], train_test['blob']['train'][0][:, 1], c = train_test['blob']['train'][1], s = 1, cmap = color)
ax[2, 0].scatter(train_test['circle']['train'][0][:, 0], train_test['circle']['train'][0][:, 1], c = train_test['circle']['train'][1], s = 1, cmap = color)
ax[3, 0].scatter(train_test['class']['train'][0][:, 0], train_test['class']['train'][0][:, 1], c = train_test['class']['train'][1], s = 1, cmap = color)

ax[0, 1].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['linear']['moon'], s = 1, cmap = color)
ax[1, 1].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['linear']['blob'], s = 1, cmap = color)
ax[2, 1].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['linear']['circle'], s = 1, cmap = color)
ax[3, 1].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['linear']['class'], s = 1, cmap = color)

ax[0, 2].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['rbf']['moon'], s = 1, cmap = color)
ax[1, 2].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['rbf']['blob'], s = 1, cmap = color)
ax[2, 2].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['rbf']['circle'], s = 1, cmap = color)
ax[3, 2].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['rbf']['class'], s = 1, cmap = color)

ax[0, 3].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['sigmoid']['moon'], s = 1, cmap = color)
ax[1, 3].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['sigmoid']['blob'], s = 1, cmap = color)
ax[2, 3].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['sigmoid']['circle'], s = 1, cmap = color)
ax[3, 3].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['sigmoid']['class'], s = 1, cmap = color)

ax[0, 4].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['polynomial']['moon'], s = 1, cmap = color)
ax[1, 4].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['polynomial']['blob'], s = 1, cmap = color)
ax[2, 4].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['polynomial']['circle'], s = 1, cmap = color)
ax[3, 4].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['polynomial']['class'], s = 1, cmap = color)

ax[0, 5].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['laplace']['moon'], s = 1, cmap = color)
ax[1, 5].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['laplace']['blob'], s = 1, cmap = color)
ax[2, 5].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['laplace']['circle'], s = 1, cmap = color)
ax[3, 5].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['laplace']['class'], s = 1, cmap = color)

ax[0, 6].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['linrbf']['moon'], s = 1, cmap = color)
ax[1, 6].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['linrbf']['blob'], s = 1, cmap = color)
ax[2, 6].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['linrbf']['circle'], s = 1, cmap = color)
ax[3, 6].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['linrbf']['class'], s = 1, cmap = color)

ax[0, 7].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['moon'], s = 1, cmap = color)
ax[1, 7].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['blob'], s = 1, cmap = color)
ax[2, 7].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['circle'], s = 1, cmap = color)
ax[3, 7].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['rbfpoly']['class'], s = 1, cmap = color)

ax[0, 8].scatter(train_test['moon']['test'][0][:, 0], train_test['moon']['test'][0][:, 1], c = kernel_outcome['etakernel']['moon'], s = 1, cmap = color)
ax[1, 8].scatter(train_test['blob']['test'][0][:, 0], train_test['blob']['test'][0][:, 1], c = kernel_outcome['etakernel']['blob'], s = 1, cmap = color)
ax[2, 8].scatter(train_test['circle']['test'][0][:, 0], train_test['circle']['test'][0][:, 1], c = kernel_outcome['etakernel']['circle'], s = 1, cmap = color)
ax[3, 8].scatter(train_test['class']['test'][0][:, 0], train_test['class']['test'][0][:, 1], c = kernel_outcome['etakernel']['class'], s = 1, cmap = color)

ax[0, 0].set_title('original')
ax[0, 1].set_title('linear')
ax[0, 2].set_title('rbf')
ax[0, 3].set_title('poly')
ax[0, 4].set_title('sigmoid')
ax[0, 5].set_title('laplace')
ax[0, 6].set_title('rbfpoly')
ax[0, 7].set_title('linrbf')
ax[0, 8].set_title('etakernel')
ax[0, 0].set_ylabel('Moons')
ax[1, 0].set_ylabel('Blob')
ax[2, 0].set_ylabel('Circle')
ax[3, 0].set_ylabel('Classifciation')
fig.set_tight_layout(True)

#%%

