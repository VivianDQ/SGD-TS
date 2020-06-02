from pypolyagamma import PyPolyaGamma
import numpy as np
import random
import math
import time
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class PG_TS_stream:
    def __init__(self, class_context, model, dist, T):        
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.model = getattr(reward_model(), model)
        
    def pg_ts(self, bc, Bc):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        b = np.ones(d) * bc
        B = np.identity(d) * Bc
        theta = np.random.multivariate_normal(b, B)
        B_inv = np.linalg.inv(B)
        B_inv_b = B_inv.dot(b)
        
        pg = PyPolyaGamma(seed=0)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        
        t = 0
        K = len(self.data.fv[t])
        ts_idx = [0]*K
        feature = self.data.fv[t]
        for arm in range(K):
            ts_idx[arm] = feature[arm].dot(theta)
            
        pull = np.argmax(ts_idx)
        observe_r = self.random_sample(t, pull) 
        regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        y = np.concatenate((y, [observe_r]), axis = 0)
        X = np.concatenate((X, [feature[pull]]), axis = 0)
        
        theta_pg = np.zeros(d)
        theta_pg[:] = theta[:]
        
        for t in range(1, T):
            w = []
            for i in range(t):
                w.append( pg.pgdraw( 1, X[i].dot(theta_pg) ) )
            omega = np.diag(w)
            k = y-0.5
            V = np.linalg.inv(X.T.dot(omega).dot(X) + B_inv)
            m = V.dot(X.T.dot(k) + B_inv_b)
            theta_pg = np.random.multivariate_normal(m, V)
            theta[:] = theta_pg[:]
            
            feature = self.data.fv[t]
            K = len(feature)
            ts_idx = [0]*K
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta)
            pull = np.argmax(ts_idx)
            observe_r = self.random_sample(t, pull) 
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
        
        return regret
