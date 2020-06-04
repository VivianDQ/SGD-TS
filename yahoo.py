#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import random
import math
import time
import os
import argparse
from sklearn.linear_model import LogisticRegression
import pickle
import os.path

from algorithms.sgd_ts import SGD_TS
from algorithms.UCB import UCB
from algorithms.laplace_ts import LAPLACE_TS
from algorithms.gloc import GLOC
from tune import GridSearch
from algorithms.data_processor.yahoo_extract_data import extract_data
from algorithms.data_processor.data_generator import *

import warnings
# silent the following warnings since that the step size in grid search set does not always offer convergence
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# ignore the following warning since that sklearn logistic regression does not always converge on the data
# it might be because that logistic model is not suitable for the data, this is probably the case especially for real datasets
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

print('start processing yahoo data')
t0 = time.time()
if not os.path.isfile('data/rewards_yahoo.txt') or not os.path.isfile('data/features_yahoo.txt'):
    extract_data()
with open('data/rewards_yahoo.txt', 'rb') as f:
    rewards = pickle.load(f)
with open('data/features_yahoo.txt', 'rb') as f:
    features = pickle.load(f)
print('data processing done, cost time {} seconds'.format(time.time()-t0))

parser = argparse.ArgumentParser(description='experiments for yahoo data')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat times')                  
args = parser.parse_args()
rep = args.rep # number of times to repeat experiments

T = len(features)
K = 20
d = 6
model = 'logistic'
dist = 'ber'
dtype = 'yahoo'
if dist != 'ber' and model == 'logistic':
    raise NameError('logistic regression only supports bernoulli reward')
                                 
print('data: Yahoo, K: around {}, T: {}, dimension: {}'.format(K, T, d))                   
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

print('start running bandit algorithms')
print('# of repeats: cumulative regret of {ucb-glm, sgd-ts, gloc, laplace-ts}')
for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    bandit = yahoo(rewards, features, d)
    gridsearch = GridSearch(parameters)
    reg_ucbglm += gridsearch.tune_ucbglm(bandit, dist, T, d, model)
    reg_sgdts += gridsearch.tune_sgdts(bandit, dist, T, d, model)
    reg_gloc += gridsearch.tune_gloc(bandit, dist, T, d, model)
    reg_lts += gridsearch.tune_laplacets(bandit, dist, T, d, model)
    print( reg_ucbglm[-1], reg_sgdts[-1], reg_gloc[-1], reg_lts[-1] )

result = {
    'ucb-glm': reg_ucbglm/rep,
    'sgd-ts': reg_sgdts/rep,
    'gloc': reg_gloc/rep,
    'lts': reg_lts/rep
}

name = 'yahoo'
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + name + '/'):
    os.mkdir('results/' + name + '/')
for k,v in result.items():
    np.savetxt('results/' + name + '/' + k, v)                                                                                                                       

