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
from tune import GridSearch
from algorithms.data_processor.data_generator import * 

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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
parameters = {
        'step_size': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'C': list(range(1,11)),
        'explore': [0.01, 0.1, 1, 5, 10],
        'stability': 10**(-6) # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
    }

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    t0 = time.time()
    theta = np.random.normal(0.1, 1, d)
    bandit = context(K, lb, ub, T, d, true_theta = theta)
    bandit.build_bandit(model)
    gridsearch = GridSearch(parameters)
    reg_ucbglm += gridsearch.tune_ucbglm(bandit, dist, T, d, model)
    reg_sgdts += gridsearch.tune_sgdts(bandit, dist, T, d, model)
    reg_gloc += gridsearch.tune_gloc(bandit, dist, T, d, model)
    reg_lts += gridsearch.tune_laplacets(bandit, dist, T, d, model)
    print('cost {} minutes'.format( (time.time() - t0)/60 ))

result = {
    'ucb-glm': reg_ucbglm/rep,
    'sgd-ts': reg_sgdts/rep,
    'gloc': reg_gloc/rep,
    'lts': reg_lts/rep
}
for k,v in result.items():
    np.savetxt('results/' + name + '/' + k, v)                                                                                                                       