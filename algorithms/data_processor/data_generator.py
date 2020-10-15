import numpy as np
import random
import math

class reward_model:
    def __init__(self):
        pass
    def logistic(self, x):
        return 1/(1+np.exp(-x))
    
class Gradient:
    def __init__(self):
        pass 
    def logistic(self, x, y, theta, lamda = 0):
        return x*( -y + 1/(1+np.exp(-x.dot(theta))) ) + 2*lamda*theta
    
class yahoo:
    def __init__(self, reward, fv, d=6):
        self.T = len(fv)
        self.fv = fv 
        self.d = d
        self.reward = reward 
        # note that we have set the optimal to be 0, and the regret at round t will be negative, 
        # regret at round t is optimal (0) - reward of pulled arm = -reward
        # in the plot.py, for yahoo data, we will change the sign back before plotting
        # this makes the plot to be reward aganist time
        self.optimal = [0 for t in range(self.T)]  
    def ber(self, t, i):
        return np.random.binomial(1, self.reward[t][i])

class covtype_random_feature:
    def __init__(self, reward, data, T, dim):
        self.d = dim
        self.T = T
        self.fv = []
        self.reward = [reward for t in range(self.T)]
        self.optimal = [max(reward) for t in range(self.T)]
        self.data = data
        self.y = []
    def build_bandit(self):
        for t in range(self.T):
            tmp = [None for _ in range(32)]
            tmpy = [0]*32
            idx = self.data[2]
            for i in range(32):
                data_idx = np.random.choice(idx[i])
                tmp[i] = self.data[0][data_idx]
                tmpy[i] = self.data[1][data_idx]
            self.fv.append(np.array(tmp))
            self.y.append(tmpy)
    def ber(self, t, i):
        return int(self.y[t][i])
        
class covtype:
    def __init__(self, reward, fv, T, dim = 10):
        self.d = dim
        self.T = T
        self.fv = [fv for t in range(self.T)] 
        self.reward = [reward for t in range(self.T)]
        self.optimal = [max(reward) for t in range(self.T)]  
    def ber(self, t, i):
        return np.random.binomial(1, self.reward[t][i])
    
class context:
    def __init__(self, K, lb_fv, ub_fv, T, d, true_theta, fv = None):
        if fv is None:
            fv = np.random.uniform(lb_fv, ub_fv, (T, K, d))
        self.K = K  
        self.d = d
        self.ub = ub_fv
        self.lb = lb_fv
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.theta = true_theta
    def build_bandit(self, model):
        f = getattr(reward_model(), model)
        for t in range(self.T):
            self.reward[t] = [f(self.fv[t][i].dot(self.theta)) for i in range(self.K)] 
            self.optimal[t] = max(self.reward[t])  # max reward
    def ber(self, t, i):
        return np.random.binomial(1, self.reward[t][i])



