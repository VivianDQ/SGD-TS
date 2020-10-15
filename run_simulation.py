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
from algorithms.supcb_glm import SupCB_GLM
from algorithms.glmtsl import GLM_TSL
from algorithms.eps_greedy import Eps_Greedy
from tune import GridSearch
from algorithms.data_processor.data_generator import * 

import warnings
# silent the following warnings since that the step size in grid search set does not always offer convergence
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-t', '--t', type=int, default = 1000, help = 'total time')
parser.add_argument('-d', '--d', type=int, default = 6, help = 'dimension')
parser.add_argument('-k', '--k', type=int, default = 100, help = 'number of arms')
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
reg_tsl = np.zeros(T)
reg_supcb = np.zeros(T)
reg_eps = np.zeros(T)

parameters = {
        'step_size': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], # total 7
        'C': list(range(1,11)), # total 10
        'explore': [0.01, 0.1, 1, 5, 10], # total 5
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
print('# of repeats: cumulative regret of {ucb-glm, sgd-ts, gloc, laplace-ts, glm-tsl, supcb-glm, eps-greedy}')
for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    theta = np.random.uniform(lb, ub, d)
    bandit = context(K, lb, ub, T, d, true_theta = theta)
    bandit.build_bandit(model)
    gridsearch = GridSearch(parameters)

    reg, seconds = gridsearch.tune_ucbglm(bandit, dist, T, d, model)
    reg_ucbglm += reg
    times['ucbglm'] += seconds
    
    reg, seconds = gridsearch.tune_sgdts(bandit, dist, T, d, model)
    reg_sgdts += reg
    times['sgdts'] += seconds
    
    reg, seconds = gridsearch.tune_gloc(bandit, dist, T, d, model)
    reg_gloc += reg
    times['gloc'] += seconds
    
    reg, seconds = gridsearch.tune_laplacets(bandit, dist, T, d, model)
    reg_lts += reg
    times['lts'] += seconds
    
    reg, seconds = gridsearch.tune_glmtsl(bandit, dist, T, d, model)
    reg_tsl += reg
    times['tsl'] += seconds
    
    reg, seconds = gridsearch.tune_supcb(bandit, dist, T, d, model)
    reg_supcb += reg
    times['supcb'] += seconds
    
    reg, seconds = gridsearch.tune_epsgreedy(bandit, dist, T, d, model)
    reg_eps += reg
    times['eps'] += seconds
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

for k,v in result.items():
    np.savetxt('results/' + name + '/' + k, v)                                                                                                                       