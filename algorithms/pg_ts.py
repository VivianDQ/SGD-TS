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
        B_inv = np.identity(d) / Bc
        theta = np.random.multivariate_normal(b, B)
        pg = PyPolyaGamma(seed=0)
        
        t = 0
        K = len(self.data.fv[t])
        ts_idx = [0]*K
        feature = self.data.fv[t]
        for arm in range(K):
            ts_idx[arm] = feature[arm].dot(theta)
        
        pull = np.argmax(ts_idx)
        observe_r = self.random_sample(t, pull) 
        regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        
        theta_pg = np.zeros(d)
        theta_pg[:] = theta[:]
        xxt = []
        xxt.append(np.outer(feature[pull], feature[pull]))
        xt_k_plus_Bb = (observe_r - 0.5) * feature[pull] + B_inv_b
        draw_pg = [feature[pull].dot(theta_pg)]
        
        for t in range(1, T):
            xt_omega_x = np.zeros((d,d))
            for i in range(t):
                w = pg.pgdraw( 1, draw_pg[i] )
                xt_omega_x += w * xxt[i]
            
            V = np.linalg.inv(xt_omega_x + B_inv)
            m = V.dot(xt_k_plus_Bb)
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
            
            xt_k_plus_Bb += (observe_r - 0.5) * feature[pull]
            xxt.append(np.outer(feature[pull], feature[pull]))
            draw_pg.append( feature[pull].dot(theta_pg) )
        
        return regret
