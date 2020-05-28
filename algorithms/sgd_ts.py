import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class SGD_TS:
    def __init__(self, class_context, model, dist, T):        
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.grad = getattr(Gradient(), model)
        self.model = getattr(reward_model(), model)
    def glm(self, eta0, tau, g1, g2):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.random_sample(t, pull) 
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
        if y[0] == y[1]:
            y[1] = 1-y[0]
        clf = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'lbfgs').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0]*K
            if t%tau == 0:
                j = t//tau
                cov = (2*g1**2 + 2*g2**2) * np.identity(d) / j
                eta = eta0/j
                theta_tilde -= eta*grad
                distance = np.linalg.norm(theta_tilde-theta_hat) 
                if distance > 2:
                    theta_tilde = theta_hat + 2*(theta_tilde-theta_hat)/distance
                grad = np.zeros(d)
                theta_bar = (theta_bar * (j-1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts) 
            pull = np.argmax(ts_idx)
            observe_r = self.random_sample(t, pull) 
            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret   


