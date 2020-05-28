#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
from collections import defaultdict
import gzip
import math
import pickle

def extract_data():
    ar = set()
    tm = set()
    count = dict()
    article_idx = dict()
    reward = []
    arm_idx = []
    fv = [] 
    filename = ['ydata-fp-td-clicks-v1_0.2009050' + str(i) for i in list(range(1,10))] + ['ydata-fp-td-clicks-v1_0.20090510']
    c = 0
    prevt = 0
    tc = -1
    lc = 0
    for fn in filename:
        t0 = time.time()
        print('processing May {} data'.format(fn[-2:]))
        with gzip.open('data/R6/' + fn + '.gz', "r") as f:
            for line in f:
                lines = line.split(b"|")
                t, article, click = list(map(int, lines[0].strip().split(b" ")))[:3] 

                if t != prevt:
                    tc += 1
                    prevt = t
                    fv.append(defaultdict(np.array))
                    reward.append(dict())
                    count = dict()
                    arm_idx.append(set())
                for i in range(2, len(lines)):
                    a = lines[i].strip().split(b" ")
                    if int(a[1].split(b':')[0]) > 6: 
                        continue
                    ar_id = int(a[0])
                    if ar_id not in article_idx:
                        article_idx[ar_id] = c
                        c += 1
                    arm_idx[tc].add(article_idx[ar_id])
                    if article_idx[ar_id] in fv[tc]: continue
                    vec = np.zeros(6)
                    flag = True
                    for s in a[1:]:
                        if int(s.split(b":")[0]) > 6:
                            flag = False
                            break
                        vec[int(s.split(b":")[0])-1] = float(s.split(b":")[1])
                    if flag:
                        fv[tc][article_idx[ar_id]] = vec 
                if article not in article_idx: continue
                idx = article_idx[article]
                if idx in reward[tc]:
                    reward[tc][idx] = (reward[tc][idx] * count[idx] + click) / (count[idx]+1)
                    count[idx] += 1
                else:
                    count[idx] = 1
                    reward[tc][idx] = click
        
    features = []
    rewards = []
    T = len(fv)
    for t in range(T):
        curarm = list(arm_idx[t])
        fvv = []
        rew = []
        for arm in curarm:
            # print(t,arm)
            fvv += [fv[t][arm]]
            rew += [reward[t][arm]]
        features.append(fvv)
        rewards.append(rew)
    # this gives two variables
    # features[t]: it contains features for arms only valid at time t, changing all the time, from 0-20 maybe
    # rewards[t]: it contains reward for arms only valid at time t, changing all the time, from 0-20 maybe
    
    with open('data/rewards_yahoo.txt', 'wb') as f:
        pickle.dump(rewards, f)
    with open('data/features_yahoo.txt', 'wb') as f:
        pickle.dump(features, f)




