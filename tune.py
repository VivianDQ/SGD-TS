#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
import time
from sklearn.linear_model import LogisticRegression

from algorithms.sgd_ts import SGD_TS
from algorithms.UCB import UCB
from algorithms.laplace_ts import LAPLACE_TS
from algorithms.gloc import GLOC
from algorithms.supcb_glm import SupCB_GLM
from algorithms.glmtsl import GLM_TSL
from algorithms.eps_greedy import Eps_Greedy

class GridSearch:
    def __init__(self, paras):
        self.paras = paras 
        
    def tune_ucbglm(self, bandit, dist, T, d, model):
        ucbglm = getattr(UCB(bandit, dist, T), model)
        best = float('Inf')
        tbest = float('Inf')
        for C in self.paras['C']: 
            tau = int(max(d, math.log(T)) * C)
            for explore in self.paras['explore']: 
                t0 = time.time()
                tmp = ucbglm(tau, self.paras['stability'], explore)
                seconds = time.time() - t0
                if seconds <= tbest and tmp[-1] < float('Inf'):
                    tbest = seconds
                if tmp[-1] < best:
                    best = tmp[-1]
                    reg = tmp 
        return reg, tbest
    
    def tune_sgdts(self, bandit, dist, T, d, model):
        sgd_ts = SGD_TS(bandit, model, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        for C in self.paras['C']:
            tau = int(max(d, math.log(T)) * C)
            for eta0 in self.paras['step_size']:  
                for g1 in self.paras['explore']:
                    for g2 in self.paras['explore']:
                        t0 = time.time()
                        tmp = sgd_ts.glm(eta0, tau, g1, g2)
                        seconds = time.time() - t0
                        if seconds <= tbest and tmp[-1] < float('Inf'):
                            tbest = seconds
                        if tmp[-1] < best:
                            reg = tmp
                            best = tmp[-1]
        return reg, tbest
    
    def tune_gloc(self, bandit, dist, T, d, model):
        gloc = GLOC(bandit, model, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        for eta in self.paras['step_size']:
            for k in self.paras['step_size']:
                for c in self.paras['explore']:
                    t0 = time.time()
                    tmp = gloc.Gloc(c, k, eta)
                    seconds = time.time() - t0
                    if seconds <= tbest and tmp[-1] < float('Inf'):
                        tbest = seconds
                    if tmp[-1] < best:
                        best = tmp[-1]
                        reg = tmp
        return reg, tbest
    
    def tune_laplacets(self, bandit, dist, T, d, model):
        lts = LAPLACE_TS(bandit, model, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        max_ite = 1000
        for eta0 in self.paras['step_size']:
            t0 = time.time()
            tmp = lts.laplace_ts(eta0)
            seconds = time.time() - t0
            if seconds <= tbest and tmp[-1] < float('Inf'):
                tbest = seconds
            if tmp[-1] < best:
                best = tmp[-1]
                reg = tmp
        return reg, tbest
    
    def tune_glmtsl(self, bandit, dist, T, d, model):
        tsl = GLM_TSL(bandit, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        for C in self.paras['C']: 
            tau = int(max(d, math.log(T)) * C)
            for a_explore in self.paras['explore']: 
                t0 = time.time()
                tmp = tsl.logistic(tau, self.paras['stability'], a_explore)
                seconds = time.time() - t0
                if seconds <= tbest and tmp[-1] < float('Inf'):
                    tbest = seconds
                if tmp[-1] < best:
                    best = tmp[-1]
                    reg = tmp 
        return reg, tbest
    
    def tune_supcb(self, bandit, dist, T, d, model):
        supcb = SupCB_GLM(bandit, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        for C in self.paras['C']: 
            tau = int(max(d, math.log(T)) * C)
            for explore in self.paras['explore']: 
                t0 = time.time()
                tmp = supcb.logistic(tau, self.paras['stability'], explore)
                seconds = time.time() - t0
                if seconds <= tbest and tmp[-1] < float('Inf'):
                    tbest = seconds
                if tmp[-1] < best:
                    best = tmp[-1]
                    reg = tmp 
        return reg, tbest

    def tune_epsgreedy(self, bandit, dist, T, d, model):
        eps_greedy = Eps_Greedy(bandit, dist, T)
        best = float('Inf')
        tbest = float('Inf')
        for explore in self.paras['explore']: 
            t0 = time.time()
            tmp = eps_greedy.logistic(explore)
            seconds = time.time() - t0
            if seconds <= tbest and tmp[-1] < float('Inf'):
                tbest = seconds
            if tmp[-1] < best:
                best = tmp[-1]
                reg = tmp 
        return reg, tbest
    