#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class LAPLACE_TS:
    def __init__(self, class_context, model, dist, T):        
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.model = getattr(reward_model(), model)
    def mu(self, x):
        return 1/(1+np.exp(-x))
    def grad(self, w, q, m, X, y):
        d = self.d
        g = np.zeros(d)
        for i in range(len(X)):
            g += (w-m)*q - y[i]*X[i] / (1+np.exp(y[i]* X[i].dot(w)))
        return g
    def optimize(self, w, m, q, X, y, eta, max_ite):
        d = self.d
        w = np.zeros(d)
        for i in range(max_ite):
            grad = self.grad(w, q, m, X, y)
            grad_norm = np.linalg.norm(grad)
            if grad_norm <= 10**(-4):
                break
            if i%100 == 0:
                eta /= 2
            w -= eta * grad
        return w 
    def laplace_ts(self, eta0 = 0.1, lamda = 1, max_ite = 1000):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        m = np.zeros(d)
        q = np.ones(d)*lamda
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0]*K
            if np.isnan(m).any() or np.isnan(q).any() or np.isinf(m).any() or np.isinf(q).any():
                # print('inf or nan encountered in posterior, will change to another step size to continue grid search')
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(m, np.diag(1/q))
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.random_sample(t, pull) 
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.array([2*observe_r - 1])
            X = np.array([feature[pull]])
            w = self.optimize(theta, m, q, X, y, eta0, max_ite) 
            m[:] = w[:]
            p = self.mu(feature[pull].dot(w))
            q += p*(1-p)* feature[pull]**2
        return regret