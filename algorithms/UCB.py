import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class UCB:
    def __init__(self, class_context, dist, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist) 
    def logistic(self, tau, eps, explore):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        B = np.identity(d) * eps 
        theta_hat = np.zeros(d)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        theta = np.zeros(d)
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            B += np.outer(feature[pull], feature[pull])
        if y[0] == y[1]:
            y[1] = 1-y[0]
        B_inv = np.linalg.inv(B)
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ucb_idx = [0]*K
            clf = LogisticRegression(penalty = 'none', max_iter = 1000, fit_intercept = False, solver = 'lbfgs').fit(X, y)
            theta = clf.coef_[0]
            for arm in range(K):
                ucb_idx[arm] = feature[arm].dot(theta) + explore * math.sqrt( feature[arm].T.dot(B_inv).dot(feature[arm]) )
            pull = np.argmax(ucb_idx)
            observe_r = self.random_sample(t, pull)
            tmp = B_inv.dot(feature[pull])
            B_inv -= np.outer(tmp, tmp)/ (1+feature[pull].dot(tmp))
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret
        
       

