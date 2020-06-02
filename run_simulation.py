#!/usr/bin/env python
# coding: utf-8



import numpy as np
import random
import math
import time
import os
import argparse
from sklearn.linear_model import LogisticRegression

from algorithms.sgd_ts import SGD_TS
from algorithms.UCB import UCB
from algorithms.laplace_ts import LAPLACE_TS
from algorithms.gloc import GLOC
from algorithms.pg_ts import PG_TS_stream
from tune import GridSearch
from algorithms.data_processor.data_generator import * 

import warnings
# silent the following warnings since that the step size in grid search set does not always offer convergence
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-t', '--t', type=int, help = 'total time')
parser.add_argument('-d', '--d', type=int, help = 'dimension')
parser.add_argument('-k', '--k', type=int, help = 'number of arms')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')
args = parser.parse_args()

T = args.t
d = args.d
K = args.k
rep = args.rep

ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)
model = 'logistic'
dist = 'ber'
if dist != 'ber' and model == 'logistic':
    raise NameError('logistic regression only supports bernoulli reward')

name = 'simulation_d' + str(d) + '_k' + str(K)
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + name + '/'):
    os.mkdir('results/' + name + '/')

print('K: {}, T: {}, dimension: {}, data name: {}'.format(K, T, d, name)) 
reg_sgdts = np.zeros(T)
reg_ucbglm = np.zeros(T) 
reg_lts = np.zeros(T)
reg_gloc = np.zeros(T)
reg_pgts = np.zeros(T)
parameters = {
        'bc': np.arange(0, 1.1, 0.1),
        'Bc': np.arange(0.1, 1.1, 0.1),
        # the above two parameters are for PG-TS only
        'step_size': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], # total 7
        'C': list(range(1,11)), # total 10
        'explore': [0.01, 0.1, 1, 5, 10], # total 5
        'stability': 10**(-6) # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
    }

times = {
    'ucb-glm': 0,
    'sgd-ts': 0,
    'gloc': 0,
    'lts': 0,
    'pg-ts': 0
}

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    theta = np.random.normal(0.1, 1, d)
    bandit = context(K, lb, ub, T, d, true_theta = theta)
    bandit.build_bandit(model)
    gridsearch = GridSearch(parameters)
    
    t0 = time.time()
    reg_ucbglm += gridsearch.tune_ucbglm(bandit, dist, T, d, model)
    times['ucb-glm'] += (time.time()-t0) / 50
    
    t0 = time.time()
    reg_sgdts += gridsearch.tune_sgdts(bandit, dist, T, d, model)
    times['sgd-ts'] += (time.time()-t0) / 1750
    
    t0 = time.time()
    reg_gloc += gridsearch.tune_gloc(bandit, dist, T, d, model)
    times['gloc'] += (time.time()-t0) / 245
    
    t0 = time.time()
    reg_pgts += gridsearch.tune_pgts(bandit, dist, T, d, model)
    times['pg-ts'] += (time.time()-t0) / 110
    
    t0 = time.time()
    reg_lts += gridsearch.tune_laplacets(bandit, dist, T, d, model)
    times['lts'] += (time.time()-t0) / 7
    
    print(times)
    # print('cost {} minutes'.format( (time.time() - t0)/60 ))
    
for k in times:
    times[k] /= rep
print('average time: ', times)

result = {
    'ucb-glm': reg_ucbglm/rep,
    'sgd-ts': reg_sgdts/rep,
    'gloc': reg_gloc/rep,
    'lts': reg_lts/rep,
    'pg-ts': reg_pgts/rep
}
for k,v in result.items():
    np.savetxt('results/' + name + '/' + k, v)                                                                                                                       