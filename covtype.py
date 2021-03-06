#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import gzip
import math
import pickle
import random
import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import preprocessing

from algorithms.sgd_ts import SGD_TS
from algorithms.UCB import UCB
from algorithms.laplace_ts import LAPLACE_TS
from algorithms.gloc import GLOC
from algorithms.supcb_glm import SupCB_GLM
from algorithms.glmtsl import GLM_TSL
from algorithms.eps_greedy import Eps_Greedy
from tune import GridSearch
from algorithms.data_processor.data_generator import * 

import warnings
# silent the following warnings since that the step size in grid search set does not always offer convergence
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# ignore the following warning since that sklearn logistic regression does not always converge on the data
# it might be because that logistic model is not suitable for the data, this is probably the case especially for real datasets
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser(description='experiments for cover type data')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
parser.add_argument('-t', '--t', type=int, default = 1000, help = 'total time')
parser.add_argument('-d', '--d', type=int, default = 10, help = 'number of features, choice of 10 (not use categorical features), 55 (use cat)')
parser.add_argument('-center', '--center', type=int, default = 1, help = 'use centriods as features (1), random feature (0)')
parser.add_argument('-add', '--add', type=int, default = 0, help = 'add a constant column feature')

args = parser.parse_args()
rep = args.rep  # repeat times, set to 10
T = args.t  # total rounds, set to 1000
d = args.d  # feature dimension, if use only quantitative features, d = 10, otherwise, d = 55
center = args.center # if center == 1, use cluster centroid as features, if center == 0, use random features
add_constant = args.add # add_constant = 1 to add a constant feature to the data
d += add_constant 

if center == 1:
    print('use cluster centroid as features, d = {},'.format(d), 'start processing data')
if center == 0:
    print('use random features, d = {},'.format(d), 'start processing data')
    
# extract, centeralize, standardize and cluster cover type data
lines = []
labels = []
t0 = time.time()
# save the 'covtype.data.gz' under the 'data' folder before running this code
with gzip.open('data/covtype.data.gz', "r") as f:
    for line in f:
        line = line.split(b',')
        tmp = line[:d]
        y = int(line[-1])
        if y!=1:
            y = 0
        x = [float(i) for i in tmp]
        lines += [x]
        labels += [y]

X = np.array(lines)
y = np.array(labels)
X[:,:10] = preprocessing.scale(X[:,:10])
if add_constant == 1:
    X_add = np.ones((X.shape[0],X.shape[1]+1))
    X_add[:,:-1] = X
else:
    X_add = X

np.random.seed(0)
kmeans = KMeans(n_clusters=32, random_state=1).fit(X_add)
rewards = [0]*32
idx = [None for _ in range(32)]
features = np.array(kmeans.cluster_centers_)
for nc in range(32):
    idx[nc] = np.where(kmeans.labels_ == nc)[0]
    num, den = sum(y[idx[nc]]), len(idx[nc])
    rewards[nc] = num / den
bandit_data = (X_add, y, idx)
K, d = 32, X_add.shape[1]

# the following code and function sort the reward and calculate the frequencies for the pulls of best 6 arms
rew = sorted(rewards, reverse = True)
gap = dict()
for i in range(32):
    gap[round(rew[0] - rew[i],4)] = i
def frequency(regr):
    fre = [0] * 32
    pulled = gap[round(regr[0],4)]
    fre[pulled] += 1
    for t in range(1, len(regr)):
        r = round(regr[t] - regr[t-1], 4)
        pulled = gap[r]
        fre[pulled] += 1
    return fre
print('data process done, cost in total {} seconds'.format(time.time()-t0))
print('max reward = {}, min reward = {}'.format(max(rewards), min(rewards)))
print('feature vectors shape: K={}, d={}'.format(K,d))

model = 'logistic'
dist = 'ber'
if dist != 'ber' and model == 'logistic':
    raise NameError('logistic regression only supports bernoulli reward')

print('K: {}, T: {}, dimension: {}, model: {}, dist: {}'.format(K, T, d, model, dist)) 

reg_sgdts = np.zeros(T)
reg_ucbglm = np.zeros(T) 
reg_lts = np.zeros(T)
reg_gloc = np.zeros(T)
reg_tsl = np.zeros(T)
reg_supcb = np.zeros(T)
reg_eps = np.zeros(T)

fre_sgdts = []
fre_lts = []
fre_ucbglm = []
fre_gloc = []
fre_tsl = []
fre_supcb = []
fre_eps = []

parameters = {
        'step_size': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'C': list(range(1,11)),
        'explore': [0.01, 0.1, 1, 5, 10],
        'stability': 10**(-6) # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
    }

times = {
    'ucbglm': 0,
    'sgdts': 0,
    'gloc': 0,
    'lts': 0,
    'tsl': 0,
    'supcb': 0,
    'eps': 0,
}

print('start running bandit algorithms')
print('# of repeats: cumulative regret of {ucb-glm, sgd-ts, gloc, laplace-ts}')
for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    if center:
        bandit = covtype(rewards, features, T, d)
    else:
        bandit = covtype_random_feature(rewards, bandit_data, T, d)
        bandit.build_bandit()  
        
    gridsearch = GridSearch(parameters)

    reg, seconds = gridsearch.tune_ucbglm(bandit, dist, T, d, model)
    reg_ucbglm += reg
    times['ucbglm'] += seconds
    fre_ucbglm.append(frequency(reg))
    
    reg, seconds = gridsearch.tune_sgdts(bandit, dist, T, d, model)
    reg_sgdts += reg
    times['sgdts'] += seconds
    fre_sgdts.append(frequency(reg))
    
    reg, seconds = gridsearch.tune_gloc(bandit, dist, T, d, model)
    reg_gloc += reg
    times['gloc'] += seconds
    fre_gloc.append(frequency(reg))
    
    reg, seconds = gridsearch.tune_laplacets(bandit, dist, T, d, model)
    reg_lts += reg
    times['lts'] += seconds
    fre_lts.append(frequency(reg))

    reg, seconds = gridsearch.tune_glmtsl(bandit, dist, T, d, model)
    reg_tsl += reg
    times['tsl'] += seconds
    fre_tsl.append(frequency(reg))

    reg, seconds = gridsearch.tune_supcb(bandit, dist, T, d, model)
    reg_supcb += reg
    times['supcb'] += seconds
    fre_supcb.append(frequency(reg))
    
    reg, seconds = gridsearch.tune_epsgreedy(bandit, dist, T, d, model)
    reg_eps += reg
    times['eps'] += seconds
    fre_eps.append(frequency(reg))
    
    print( reg_ucbglm[-1], reg_sgdts[-1], reg_gloc[-1], reg_lts[-1], reg_tsl[-1], reg_supcb[-1], reg_eps[-1] )

for k in times.keys():
    times[k] /= rep
print(times)

result = {
    'ucb-glm': reg_ucbglm/rep,
    'sgd-ts': reg_sgdts/rep,
    'gloc': reg_gloc/rep,
    'lts': reg_lts/rep,
    'tsl': reg_tsl/rep,
    'supcb': reg_supcb/rep,
    'eps': reg_eps/rep,
}

frequent = {
    'ucb-glm': fre_ucbglm,
    'sgd-ts': fre_sgdts,
    'gloc': fre_gloc,
    'lts': fre_lts,
    'tsl': fre_tsl,
    'supcb': fre_supcb,
    'eps': fre_eps,
}

# save the averaged regret for four algorithms in the directory 'results/name', where name is specified below
name = 'covtype_d{}'.format(d)
if not center:
    name += '_rf'
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + name + '/'):
    os.mkdir('results/' + name + '/')
    
for k,v in result.items():
    np.savetxt('results/' + name + '/' + k, v)       

# save the frequency info for four algorithms in the directory 'results/name', where name is specified below
name = 'covtype_freq_d{}'.format(d)
if not center:
    name += '_rf'
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + name + '/'):
    os.mkdir('results/' + name + '/')

for k,v in frequent.items():
    np.savetxt('results/' + name + '/' + k, v)  