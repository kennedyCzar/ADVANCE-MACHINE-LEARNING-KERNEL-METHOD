#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:25:09 2019

@author: kenneth
"""
import time
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1111)
from sklearn.metrics import roc_auc_score
from DualSVDD import DualSVDD, DualSVDD_NE, MiniDualSVDD, MiniDualSVDD_NE
from linearSVDD import linearSVDD, linearSVDD_NE
from svm import linearSVM, StochasticlinearSVM
from kernelSVM import kDualSVM, kprimalSVM
np.random.seed(1000)
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 200

#generate dataset for testing SVDD
def SVDDdataset(n_samples= None, outlier_size = None, n_features = None):
    if not n_samples:
        n_samples = 1000
    else:
        n_samples = n_samples
    if not n_features:
        n_features = 2
    else:
        n_features = n_features
    if not outlier_size:
        outlier_size = 1000//20
    else:
        outlier_size = outlier_size
    insample = n_samples - outlier_size
    inliers = np.random.randn(insample, n_features) * .7
    inliers_y = np.ones(insample)
    outsample = np.random.uniform(low = -7, high = 7, size = (outlier_size, n_features))
    outsample_y = np.zeros(outlier_size)
    X = np.vstack((inliers, outsample))
    y = np.hstack((inliers_y, outsample_y))
    return X, y


#%%
X, y = SVDDdataset()
X = np.hstack((X, y.reshape(-1, 1)))
df = X[X[:, 2] == 1][:, [0, 1]]
dy = X[X[:, 2] == 1][:, 2]
#plt.scatter(df[:, 0], df[:, 1])
#plt.scatter(X[:, 0], X[:, 1])
#plt.scatter(X[:, 0], X[:, 1], c = y, s = 5, cmap = 'coolwarm_r')


#%%
kernels  = ['linear', 'rbf', 'sigmoid', 'polynomial',
            'linrbf', 'rbfpoly', 'etakernel', 'laplace']

comparing = ['svdd_ne', 'svdd_e', 'svm_ne', 'svm_e']

kernel_outcome = {'linear': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': [],}, 
                             'rbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []}, 
                             'sigmoid': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []}, 
                             'polynomial': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                             'linrbf': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                             'rbfpoly': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                             'etakernel': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                             'laplace': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []}}

for ij in comparing:
    for ii in kernels:
        if ii == 'linear':
            if ij == 'svdd_e':
                #--svdd with error
                start = time.time()
                linsvdd = linearSVDD(kernel='linear').fit(df)
                end = time.time() - start
                kernel_outcome[ii][ij] = linsvdd.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(linsvdd.accuracy(y, linsvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(linsvdd.precision(y, linsvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(linsvdd.recall(y, linsvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(linsvdd.f1(y, linsvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, linsvdd.predict(X[:, [0, 1]])))
            elif ij == 'svdd_ne':
                start = time.time()
                linsvdd_ne = linearSVDD_NE(kernel='linear').fit(df)
                end = time.time() - start
                kernel_outcome[ii][ij] = linsvdd_ne.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(linsvdd_ne.accuracy(y, linsvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(linsvdd_ne.precision(y, linsvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(linsvdd_ne.recall(y, linsvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(linsvdd_ne.f1(y, linsvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, linsvdd_ne.predict(X[:, [0, 1]])))
            elif ij == 'svm_ne':
                start = time.time()
                linsvm_ne = StochasticlinearSVM().fit(X[:, [0, 1]], y)
                end = time.time() - start
                kernel_outcome[ii][ij] = linsvm_ne.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(linsvm_ne.accuracy(y, linsvm_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(linsvm_ne.precision(y, linsvm_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(linsvm_ne.recall(y, linsvm_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(linsvm_ne.f1(y, linsvm_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, linsvm_ne.predict(X[:, [0, 1]])))
            elif ij == 'svm_e':
                start = time.time()
                ksvm_e = kDualSVM(kernel = ii).fit(X[:, [0, 1]], y)
                end = time.time() - start
                kernel_outcome[ii][ij] = ksvm_e.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(ksvm_e.accuracy(y, ksvm_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(ksvm_e.precision(y, ksvm_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(ksvm_e.recall(y, ksvm_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(ksvm_e.f1(y, ksvm_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, ksvm_e.predict(X[:, [0, 1]])))
        else:
            if ij == 'svdd_e':
                #--svdd with error
                start = time.time()
                minisvdd = MiniDualSVDD(kernel = ii).fit(df)
                end = time.time() - start
                kernel_outcome[ii][ij] = minisvdd.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(minisvdd.accuracy(y, minisvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(minisvdd.precision(y, minisvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(minisvdd.recall(y, minisvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(minisvdd.f1(y, minisvdd.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, minisvdd.predict(X[:, [0, 1]])))
            elif ij == 'svdd_ne':
                #--svdd with error
                start = time.time()
                minisvdd_ne = MiniDualSVDD_NE(kernel = ii).fit(df)
                end = time.time() - start
                kernel_outcome[ii][ij] = minisvdd_ne.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(minisvdd_ne.accuracy(y, minisvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(minisvdd_ne.precision(y, minisvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(minisvdd_ne.recall(y, minisvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(minisvdd_ne.f1(y, minisvdd_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, minisvdd_ne.predict(X[:, [0, 1]])))
            elif ij == 'svm_ne':
                #--svdd with error
                start = time.time()
                kprimal_ne = kprimalSVM(kernel = ii).fit(X[:, [0, 1]], y)
                end = time.time() - start
                kernel_outcome[ii][ij] = kprimal_ne.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(kprimal_ne.accuracy(y, kprimal_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(kprimal_ne.precision(y, kprimal_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(kprimal_ne.recall(y, kprimal_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(kprimal_ne.f1(y, kprimal_ne.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, kprimal_ne.predict(X[:, [0, 1]])))
            elif ij == 'svm_e':
                #--svdd with error
                start = time.time()
                kduall_e = kDualSVM(kernel = ii).fit(df, dy)
                end = time.time() - start
                kernel_outcome[ii][ij] = kduall_e.predict(X[:, [0, 1]])
                kernel_outcome[ii]['time'].append(end)
                kernel_outcome[ii]['acc'].append(kduall_e.accuracy(y, kduall_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['prec'].append(kduall_e.precision(y, kduall_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['rec'].append(kduall_e.recall(y, kduall_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['f1'].append(kduall_e.f1(y, kduall_e.predict(X[:, [0, 1]])))
                kernel_outcome[ii]['aurroc'].append(roc_auc_score(y, kduall_e.predict(X[:, [0, 1]])))


                    
#%%                   
s = .5
color = 'coolwarm_r'
fig, ax = plt.subplots(4, 8, figsize=(12, 4),gridspec_kw=dict(hspace=0, wspace=0),
                       subplot_kw={'xticks':[], 'yticks':[]})
#--linear
ax[0, 0].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linear']['svdd_ne'], s = 1, cmap = color)
ax[1, 0].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linear']['svdd_e'], s = 1, cmap = color)
ax[2, 0].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linear']['svm_ne'], s = 1, cmap = color)
ax[3, 0].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linear']['svm_e'], s = 1, cmap = color)

#--rbf
ax[0, 1].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbf']['svdd_ne'], s = 1, cmap = color)
ax[1, 1].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbf']['svdd_e'], s = 1, cmap = color)
ax[2, 1].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbf']['svm_ne'], s = 1, cmap = color)
ax[3, 1].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbf']['svm_e'], s = 1, cmap = color)

#--poly
ax[0, 2].scatter(X[:, 0], X[:, 1], c = kernel_outcome['polynomial']['svdd_ne'], s = 1, cmap = color)
ax[1, 2].scatter(X[:, 0], X[:, 1], c = kernel_outcome['polynomial']['svdd_e'], s = 1, cmap = color)
ax[2, 2].scatter(X[:, 0], X[:, 1], c = kernel_outcome['polynomial']['svm_ne'], s = 1, cmap = color)
ax[3, 2].scatter(X[:, 0], X[:, 1], c = kernel_outcome['polynomial']['svm_e'], s = 1, cmap = color)

#--sigmoid
ax[0, 3].scatter(X[:, 0], X[:, 1], c = kernel_outcome['sigmoid']['svdd_ne'], s = 1, cmap = color)
ax[1, 3].scatter(X[:, 0], X[:, 1], c = kernel_outcome['sigmoid']['svdd_e'], s = 1, cmap = color)
ax[2, 3].scatter(X[:, 0], X[:, 1], c = kernel_outcome['sigmoid']['svm_ne'], s = 1, cmap = color)
ax[3, 3].scatter(X[:, 0], X[:, 1], c = kernel_outcome['sigmoid']['svm_e'], s = 1, cmap = color)

#--laplace
ax[0, 4].scatter(X[:, 0], X[:, 1], c = kernel_outcome['laplace']['svdd_ne'], s = 1, cmap = color)
ax[1, 4].scatter(X[:, 0], X[:, 1], c = kernel_outcome['laplace']['svdd_e'], s = 1, cmap = color)
ax[2, 4].scatter(X[:, 0], X[:, 1], c = kernel_outcome['laplace']['svm_ne'], s = 1, cmap = color)
ax[3, 4].scatter(X[:, 0], X[:, 1], c = kernel_outcome['laplace']['svm_e'], s = 1, cmap = color)

#--rbfpoly
ax[0, 5].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbfpoly']['svdd_ne'], s = 1, cmap = color)
ax[1, 5].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbfpoly']['svdd_e'], s = 1, cmap = color)
ax[2, 5].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbfpoly']['svm_ne'], s = 1, cmap = color)
ax[3, 5].scatter(X[:, 0], X[:, 1], c = kernel_outcome['rbfpoly']['svm_e'], s = 1, cmap = color)

#--linrbf
ax[0, 6].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linrbf']['svdd_ne'], s = 1, cmap = color)
ax[1, 6].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linrbf']['svdd_e'], s = 1, cmap = color)
ax[2, 6].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linrbf']['svm_ne'], s = 1, cmap = color)
ax[3, 6].scatter(X[:, 0], X[:, 1], c = kernel_outcome['linrbf']['svm_e'], s = 1, cmap = color)

#--etakernel
ax[0, 7].scatter(X[:, 0], X[:, 1], c = kernel_outcome['etakernel']['svdd_ne'], s = 1, cmap = color)
ax[1, 7].scatter(X[:, 0], X[:, 1], c = kernel_outcome['etakernel']['svdd_e'], s = 1, cmap = color)
ax[2, 7].scatter(X[:, 0], X[:, 1], c = kernel_outcome['etakernel']['svm_ne'], s = 1, cmap = color)
ax[3, 7].scatter(X[:, 0], X[:, 1], c = kernel_outcome['etakernel']['svm_e'], s = 1, cmap = color)


ax[0, 0].set_title('linear')
ax[0, 1].set_title('rbf')
ax[0, 2].set_title('poly')
ax[0, 3].set_title('sigmoid')
ax[0, 4].set_title('laplace')
ax[0, 5].set_title('rbfpoly')
ax[0, 6].set_title('linrbf')
ax[0, 7].set_title('etakernel')
ax[0, 0].set_ylabel('SVDD No Error')
ax[1, 0].set_ylabel('SVDD Error')
ax[2, 0].set_ylabel('SVM No Error')
ax[3, 0].set_ylabel('SVM Error')
fig.set_tight_layout(True)



#%%

from sklearn.metrics import roc_auc_score
dsvdd = linearSVDD(kernel='linear').fit(df)
#plt.plot(np.arange(100), dsvdd.cost_rec)

dsvdd.predict(X[:, [0, 1]])
dsvdd.summary(y, dsvdd.predict(X[:, [0, 1]]), dsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = np.sign(X[:, [0, 1]].dot(dsvdd.R_squared) -0.5*x.diagonal() + R_squared), cmap = 'coolwarm_r', s = 5)
roc_auc_score(y, np.sign(X[:, [0, 1]].dot(dsvdd.R_squared) -0.5*x.diagonal()))
#%% SVDD-GD No Errors
dsvddNE = DualSVDD_NE(kernel='rbf').fit(df)
plt.plot(np.arange(100), dsvddNE.cost_rec)
dsvddNE.predict(X[:, [0, 1]])
plt.scatter(X[:, 0], X[:, 1], c = dsvddNE.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 1)
roc_auc_score(y, dsvddNE.predict(X[:, [0, 1]]))

#%% Minibatch with Errors
from sklearn.metrics import roc_auc_score
stochdsvdd = MiniDualSVDD(kernel='rbf').fit(df)
plt.plot(np.arange(100), stochdsvdd.cost_rec)
stochdsvdd.predict(X[:, [0, 1]])
stochdsvdd.summary(y, stochdsvdd.predict(X[:, [0, 1]]), stochdsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]), cmap = 'coolwarm_r', s = 1)
roc_auc_score(y, stochdsvdd.predict(X[:, [0, 1]]))
#%% Minibatch version with no errors
stochdsvdd = MiniDualSVDD_NE(kernel='linear').fit(df)
plt.plot(np.arange(100), stochdsvdd.cost_rec)
stochdsvdd.predict(X[:, [0, 1]])
stochdsvdd.summary(y, stochdsvdd.predict(X[:, [0, 1]]), stochdsvdd.alpha)
plt.scatter(X[:, 0], X[:, 1], c = stochdsvdd.predict(X[:, [0, 1]]))
roc_auc_score(y, stochdsvdd.predict(X[:, [0, 1]]))
#%% Decision boundary

def plot_decision_boundary(clf, X, Y, cmap='coolwarm_r'):
    h = 0.25
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha = 0.20)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.01)
    plt.scatter(X[:,0], X[:,1], c =  Y, cmap = cmap, edgecolors='k', label = f'F1: {round(clf.f1(y, clf.predict(X[:, [0, 1]])), 2)}')
    plt.legend()
plot_decision_boundary(dsvdd, X, y, cmap='coolwarm_r')



