import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class GLM_TSL:
    def __init__(self, class_context, dist, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist) 
        
    def mu_dot(self, x):
        tmp = np.exp(-x)
        return tmp / ((1+tmp)**2)
    
    def logistic(self, tau, eps, a):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)

        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1-y[0]

        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta_bar = clf.coef_[0]
            H = np.identity(d) * eps
            for l in range(t):
                tmp = self.mu_dot(X[l].dot(theta_bar))
                H += np.outer(X[l], X[l]) * tmp 
            H_inv = np.linalg.inv(H)
            # for instable posterior solving due to unsuitable parameters, end early
            if np.isnan(H_inv).any() or np.isinf(H_inv).any() or np.isnan(theta_bar).any() or np.isinf(theta_bar).any():
                regret[-1] = float('Inf')
                break
            theta = np.random.multivariate_normal(theta_bar, a**2 * H_inv)
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret