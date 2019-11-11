#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:26:28 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_checkerboard, make_swiss_roll
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.datasets import make_moons, make_biclusters
from PCA import PCA
from KPCA import kPCA
from sklearn.decomposition import KernelPCA
import time
import os
np.random.seed(1000)
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 200

samples = 1000
dfmoon, ymoon = make_moons(n_samples=samples, noise=.05)
dfcircle, ycircle = make_circles(n_samples = samples, noise = .10, factor=.5)
dfclass, yclass = make_classification(n_samples=samples,n_features=20)
dfswis, yswiss = make_swiss_roll(n_samples=samples, noise = .03)
dfiris, yiris = load_iris().data, load_iris().target


#%%
kernels  = ['linear', 'rbf', 'sigmoid', 'polynomial',
            'correlation', 'linrbf', 'rbfpoly', 'rbfcosine', 'etakernel', 'laplace']
data_name = {'moon': dfmoon, 'circle': dfcircle, 'class': dfclass, 'swiss': dfswis, 'iris': dfiris}

kernel_outcome = {'linear': {'time': [], 'variance': []}, 'rbf': {'time': [], 'variance': []}, 'sigmoid': {'time': [], 'variance': []}, 'polynomial': {'time': [], 'variance': []},
                  'correlation': {'time': [], 'variance': []}, 'linrbf': {'time': [], 'variance': []}, 'rbfpoly': {'time': [], 'variance': []}, 'rbfcosine': {'time': [], 'variance': []},
                  'etakernel': {'time': [], 'variance': []}, 'laplace': {'time': [], 'variance': []}}

for p, q in data_name.items():
    for ii in kernels:
        start = time.time()
        kpca = kPCA(k = 2, kernel = ii).fit(q)
        end = time.time() - start
        kernel_outcome[ii][f'{p}'] = kpca.fit_transform()
        kernel_outcome[ii]['time'].append(end)
        kernel_outcome[ii]['variance'].append(kpca.explained_variance)
        
        
#pca = kPCA(k = 2, kernel = kernels[0]).fit(dfclass)
#newX = pca.fit_transform()
#plt.scatter(newX[:, 0], newX[:, 1], c = yclass,  s = 3)

#%% Visualize dataset

fig, ax = plt.subplots(5, 1, figsize=(2, 7),gridspec_kw=dict(hspace=0.01, wspace=0.01),
                       subplot_kw={'xticks':[], 'yticks':[]})
fig.set_tight_layout(True)
ax[0].scatter(dfcircle[:, 0], dfcircle[:, 1], c = ycircle, s = 1, cmap = color)
ax[1].scatter(dfmoon[:, 0], dfmoon[:, 1], c = ymoon, s = 1, cmap = color)
for ii in range(1, dfclass.shape[1]):
    ax[2].scatter(dfclass[:, 0], dfclass[:, ii], c = yclass, s = 1, cmap = color)
for ii in range(1, dfswis.shape[1]):
    ax[3].scatter(dfswis[:, 0], dfswis[:, ii], c = yswiss, s = 1, cmap = color)
for ii in range(1, dfiris.shape[1]):
    ax[4].scatter(dfiris[:, 0], dfiris[:, ii], c = yiris, s = 1, cmap = color)

ax[0].set_ylabel('Circle')
ax[1].set_ylabel('Moons')
ax[2].set_ylabel('classifciation')
ax[3].set_ylabel('Swiss roll')
ax[4].set_ylabel('Iris')

#%% Multivisualization
s = .5
color = 'coolwarm_r'
fig, ax = plt.subplots(5, 10, figsize=(12, 4),gridspec_kw=dict(hspace=0, wspace=0),
                       subplot_kw={'xticks':[], 'yticks':[]})

ax[0, 0].scatter(dfcircle[:, 0], dfcircle[:, 1], c = ycircle, s = s, cmap = color)
ax[1, 0].scatter(dfmoon[:, 0], dfmoon[:, 1], c = ymoon, s = 1, cmap = color)
for ii in range(1, dfclass.shape[1]):
    ax[2, 0].scatter(dfclass[:, 0], dfclass[:, ii], c = yclass, s = s, cmap = color)
for ii in range(1, dfswis.shape[1]):
    ax[3, 0].scatter(dfswis[:, 0], dfswis[:, ii], c = yclass, s = s, cmap = color)
for ii in range(1, dfiris.shape[1]):
    ax[4, 0].scatter(dfiris[:, 0], dfiris[:, ii], c = yiris, s = s, cmap = color)
#--linear

ax[0, 1].scatter(kernel_outcome['linear']['circle'][:, 0], kernel_outcome['linear']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 1].scatter(kernel_outcome['linear']['moon'][:, 0], kernel_outcome['linear']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 1].scatter(kernel_outcome['linear']['class'][:, 0], kernel_outcome['linear']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 1].scatter(kernel_outcome['linear']['swiss'][:, 0], kernel_outcome['linear']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 1].scatter(kernel_outcome['linear']['iris'][:, 0], kernel_outcome['linear']['iris'][:, 1], c = yiris, s = s, cmap = color)
#--rbf
ax[0, 2].scatter(kernel_outcome['rbf']['circle'][:, 0], kernel_outcome['rbf']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 2].scatter(kernel_outcome['rbf']['moon'][:, 0], kernel_outcome['rbf']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 2].scatter(kernel_outcome['rbf']['class'][:, 0], kernel_outcome['rbf']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 2].scatter(kernel_outcome['rbf']['swiss'][:, 0], kernel_outcome['rbf']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 2].scatter(kernel_outcome['rbf']['iris'][:, 0], kernel_outcome['rbf']['iris'][:, 1], c = yiris, s = s, cmap = color)
#---polynomial
ax[0, 3].scatter(kernel_outcome['polynomial']['circle'][:, 0], kernel_outcome['polynomial']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 3].scatter(kernel_outcome['polynomial']['moon'][:, 0], kernel_outcome['polynomial']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 3].scatter(kernel_outcome['polynomial']['class'][:, 0], kernel_outcome['polynomial']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 3].scatter(kernel_outcome['polynomial']['swiss'][:, 0], kernel_outcome['polynomial']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 3].scatter(kernel_outcome['polynomial']['iris'][:, 0], kernel_outcome['polynomial']['iris'][:, 1], c = yiris, s = s, cmap = color)
#--sigmoid
ax[0, 4].scatter(kernel_outcome['sigmoid']['circle'][:, 0], kernel_outcome['sigmoid']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 4].scatter(kernel_outcome['sigmoid']['moon'][:, 0], kernel_outcome['sigmoid']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 4].scatter(kernel_outcome['sigmoid']['class'][:, 0], kernel_outcome['sigmoid']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 4].scatter(kernel_outcome['sigmoid']['swiss'][:, 0], kernel_outcome['sigmoid']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 4].scatter(kernel_outcome['sigmoid']['iris'][:, 0], kernel_outcome['sigmoid']['iris'][:, 1], c = yiris, s = s, cmap = color)

#--Laplacian
ax[0, 5].scatter(kernel_outcome['laplace']['circle'][:, 0], kernel_outcome['laplace']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 5].scatter(kernel_outcome['laplace']['moon'][:, 0], kernel_outcome['laplace']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 5].scatter(kernel_outcome['laplace']['class'][:, 0], kernel_outcome['laplace']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 5].scatter(kernel_outcome['laplace']['swiss'][:, 0], kernel_outcome['laplace']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 5].scatter(kernel_outcome['laplace']['iris'][:, 0], kernel_outcome['laplace']['iris'][:, 1], c = yiris, s = s, cmap = color)
#---rbfpoly
ax[0, 6].scatter(kernel_outcome['rbfpoly']['circle'][:, 0], kernel_outcome['rbfpoly']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 6].scatter(kernel_outcome['rbfpoly']['moon'][:, 0], kernel_outcome['rbfpoly']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 6].scatter(kernel_outcome['rbfpoly']['class'][:, 0], kernel_outcome['rbfpoly']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 6].scatter(kernel_outcome['rbfpoly']['swiss'][:, 0], kernel_outcome['rbfpoly']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 6].scatter(kernel_outcome['rbfpoly']['iris'][:, 0], kernel_outcome['rbfpoly']['iris'][:, 1], c = yiris, s = s, cmap = color)
#--linrbf
ax[0, 7].scatter(kernel_outcome['linrbf']['circle'][:, 0], kernel_outcome['linrbf']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 7].scatter(kernel_outcome['linrbf']['moon'][:, 0], kernel_outcome['linrbf']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 7].scatter(kernel_outcome['linrbf']['class'][:, 0], kernel_outcome['linrbf']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 7].scatter(kernel_outcome['linrbf']['swiss'][:, 0], kernel_outcome['linrbf']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 7].scatter(kernel_outcome['linrbf']['iris'][:, 0], kernel_outcome['linrbf']['iris'][:, 1], c = yiris, s = s, cmap = color)

#--rbfcosine
ax[0, 8].scatter(kernel_outcome['rbfcosine']['circle'][:, 0], kernel_outcome['rbfcosine']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 8].scatter(kernel_outcome['rbfcosine']['moon'][:, 0], kernel_outcome['rbfcosine']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 8].scatter(kernel_outcome['rbfcosine']['class'][:, 0], kernel_outcome['rbfcosine']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 8].scatter(kernel_outcome['rbfcosine']['swiss'][:, 0], kernel_outcome['rbfcosine']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 8].scatter(kernel_outcome['rbfcosine']['iris'][:, 0], kernel_outcome['rbfcosine']['iris'][:, 1], c = yiris, s = s, cmap = color)

#--etakernel
ax[0, 9].scatter(kernel_outcome['etakernel']['circle'][:, 0], kernel_outcome['etakernel']['circle'][:, 1], c = ycircle, s = s, cmap = color)
ax[1, 9].scatter(kernel_outcome['etakernel']['moon'][:, 0], kernel_outcome['etakernel']['moon'][:, 1], c = ymoon, s = s, cmap = color)
ax[2, 9].scatter(kernel_outcome['etakernel']['class'][:, 0], kernel_outcome['etakernel']['class'][:, 1], c = yclass, s = s, cmap = color)
ax[3, 9].scatter(kernel_outcome['etakernel']['swiss'][:, 0], kernel_outcome['etakernel']['swiss'][:, 1], c = yswiss, s = s, cmap = color)
ax[4, 9].scatter(kernel_outcome['etakernel']['iris'][:, 0], kernel_outcome['etakernel']['iris'][:, 1], c = yiris, s = s, cmap = color)


ax[0, 0].set_title('original')
ax[0, 1].set_title('linear')
ax[0, 2].set_title('rbf')
ax[0, 3].set_title('poly')
ax[0, 4].set_title('sigmoid')
ax[0, 5].set_title('laplace')
ax[0, 6].set_title('rbfpoly')
ax[0, 7].set_title('linrbf')
ax[0, 8].set_title('rbfcosine')
ax[0, 9].set_title('etakernel')
ax[0, 0].set_ylabel('Circle')
ax[1, 0].set_ylabel('Moons')
ax[2, 0].set_ylabel('classifciation')
ax[3, 0].set_ylabel('Swiss roll')
ax[4, 0].set_ylabel('Iris')
fig.set_tight_layout(True)
#%%


start = time.time()
pca = PCA(k = 2).fit(X)
print(f'Classical Time: {round(time.time() - start, 4)}secs')
start = time.time()
kpca = kPCA(kernel='rbfcosine').fit(X)
print(f'Kernel Time: {round(time.time() - start, 4)}secs')

for ii in range(dfiris.shape[1]):
    plt.scatter(np.arange(dfiris.shape[0]), dfiris[:, ii], s = 5, cmap = color)
#%%

kpca = kPCA(kernel='polynomial').fit(X)
kpca.explained_variance
new = kpca.fit_transform()
new = kpca.fit_transform(X)
plt.scatter(new[:, 0], new[:, 1], c = y)

plt.plot(np.cumsum(kpca.explained_variance))
#%% PCA on raandom datasets

pca = PCA(2).fit(dfswis)
pca = kPCA(k=2, kernel='rbf').fit(dfclass)
newX = pca.fit_transform()
plt.scatter(newX[:, 0], newX[:, 1], c = yclass,  s = 3)
pca.explained_variance

#%% Image dataset example

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

pca = PCA(150).fit(faces.data)
#pca = kPCA(k=150, kernel='linear').fit(faces.data.T)
projected = pca.inverse_transform()
#fig, axes = plt.subplots(3, 8, figsize=(9, 4),
#                         subplot_kw={'xticks':[], 'yticks':[]},
#                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
#for i, ax in enumerate(axes.flat):
#    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
    
#%%
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction')