#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression

from algorithms.sgd_ts import SGD_TS
from algorithms.UCB import UCB
from algorithms.laplace_ts import LAPLACE_TS
from algorithms.gloc import GLOC

class GridSearch:
    def __init__(self, paras):
        self.paras = paras 
    def tune_ucbglm(self, bandit, dist, T, d, model):
        linucb = getattr(UCB(bandit, dist, T), model)
        best = float('Inf')
        reg = None
        for C in self.paras['C']: 
            tau = int(max(d, math.log(T)) * C)
            for explore in self.paras['explore']: 
                tmp = linucb(tau, self.paras['stability'], explore)
                if tmp[-1] < best:
                    best = tmp[-1]
                    reg = tmp 
        return reg
    def tune_sgdts(self, bandit, dist, T, d, model):
        sgd_ts = SGD_TS(bandit, model, dist, T)
        best = float('Inf')
        for C in self.paras['C']:
            tau = int(max(d, math.log(T)) * C)
            for eta0 in self.paras['step_size']:  
                for g1 in self.paras['explore']:
                    for g2 in self.paras['explore']:
                        tmp = sgd_ts.glm(eta0, tau, g1, g2)
                        if tmp[-1] < best:
                            reg = tmp
                            best = tmp[-1]
        return reg
    def tune_gloc(self, bandit, dist, T, d, model):
        gloc = GLOC(bandit, model, dist, T)
        best = float('Inf')
        for eta in self.paras['step_size']:
            for k in self.paras['step_size']:
                for c in self.paras['explore']:
                    tmp = gloc.Gloc(c, 1, k, eta, lamda = 1, eps = 1)
                    if tmp[-1] < best:
                        best = tmp[-1]
                        reg = tmp
        return reg
    def tune_laplacets(self, bandit, dist, T, d, model):
        lts = LAPLACE_TS(bandit, model, dist, T)
        best = float('Inf')
        max_ite = 1000
        for eta0 in self.paras['step_size']:
            tmp = lts.laplace_ts(1, eta0, max_ite)
            if tmp[-1] < best:
                best = tmp[-1]
                reg = tmp
        return reg





