import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from algorithms.data_processor.data_generator import reward_model, Gradient

class SupCB_GLM:
    def __init__(self, class_context, dist, T):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist) 
        
    def logistic(self, tau, eps, explore):
        T = self.T
        d = self.data.d
        regret = np.zeros(T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        
        S = math.floor(math.log(T))
        phi = [set() for _ in range(S)]
        
        F = set()
        for t in range(tau):
            feature = self.data.fv[t]
            K = len(feature)
            pull = np.random.choice(K)
            F.add(t)
            observe_r = self.random_sample(t, pull) 
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]

        if y[0] == y[1]:
            y[1] = 1-y[0]
        
        for t in range(tau, T):
            feature = self.data.fv[t]
            K = len(feature)
            A = set(list(range(0,K)))
            s = 1
            pull = None
            while not pull and s < S:
                # part a
                phi_cur = list(phi[s].union(F))
                clf = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'lbfgs').fit(X[phi_cur], y[phi_cur])
                theta = clf.coef_[0]
                
                Vt = np.identity(d) * eps 
                for xi in phi_cur:
                    Vt += np.outer(X[xi], X[xi])
                Vt_inv = np.linalg.inv(Vt)
                
                w = {}
                m = {}
                for i, candidate in enumerate(A):
                    m[candidate] = explore * np.sqrt( feature[candidate].dot(theta) )
                for i, candidate in enumerate(A):
                    w[candidate] = explore * np.sqrt( feature[candidate].T.dot(Vt_inv).dot(feature[candidate]) )
                    if w[candidate] > 2**(-s):
                        pull = candidate
                        phi[s] = phi[s].union(set([t]))
                        break
                if w == {}:
                    w_max = 0
                else:
                    w_max = np.max(list(w.values()))
                if pull is not None and pull >= 0 and pull < K: break
                
                elif w_max <= 1/math.sqrt(T):
                    pull = np.argmax(m)
                    phi[0] = phi[0].union(set([t]))
                elif w_max <= 2**(-s):
                    A_new = set()
                    for candidate in A:
                        if m[candidate] >= w_max - 2*2**(-s):
                            A_new.add(candidate)
                    s += 1
                    A = A_new
            if pull == None: 
                pull = np.random.choice(K)

            observe_r = self.random_sample(t, pull)
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][pull]
        return regret
        
       

