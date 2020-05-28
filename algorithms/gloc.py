#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class GLOC:
    def __init__(self, class_context, model, dist, T):        
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.model = getattr(reward_model(), model) 
    def mu(self, x):
        return 1/(1+np.exp(-x))
    def grad(self, theta, X, y):
        d = self.d
        g = -y + self.mu(X.dot(theta))
        return g 
    def argm(self, theta_prime, A, S, eta):
        n = np.linalg.norm(theta_prime)
        if n <= S:
            return theta_prime
        theta = np.zeros(self.d)
        for ite in range(1000):
            grad = A.dot(theta - theta_prime)
            if np.linalg.norm(grad) <= 10**(-4):
                break
            theta -= eta * grad
        return theta
    def Gloc(self, c, S, k, eta, lamda = 1, eps = 1):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        A = eps * np.identity(d)
        A_inv = 1/eps * np.identity(d)
        V_inv = 1/lamda * np.identity(d)
        theta_hat = np.zeros(d)
        beta = c
        theta = np.zeros(d)
        xz = np.zeros(d)
        for t in range(T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta_hat) + math.sqrt(beta) * math.sqrt(feature[arm].dot(V_inv).dot(feature[arm]))
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull) 
            gs = self.grad(theta, feature[pull], observe_r)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            tmp = A_inv.dot(feature[pull])
            A_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            A += np.outer(feature[pull], feature[pull])
            tmp = V_inv.dot(feature[pull])
            V_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            theta_prime = theta - gs * k * A_inv.dot(feature[pull])
            theta = self.argm(theta_prime, A, S, eta)
            xz += feature[pull].dot(theta) * feature[pull]
            theta_hat = V_inv.dot(xz)
        return regret
    