#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def plot_frequency(path):
    frequent = {
    'sgd-ts': [0, 'SGD-TS', 'C3'],
    'ucb-glm': [0, 'UCB-GLM', 'C2'],
    'gloc': [0, 'GLOC', 'C1'],
    'lts': [0, 'Laplace-TS', 'C0']
    }
    fn = path.split('/')[-2]
    fn_split = fn.split('_')
    d = int(fn_split[2][1:])
    for k in frequent.keys():
        tmp = np.loadtxt(path + k)
        frequent[k][0] = tmp 
    max_arm = 6
    labels = [str(i+1) for i in range(max_arm)]
    x = np.arange(len(labels)) 
    width = 0.12  
    fig = plot.figure(figsize=(6,4))
    start = x - width/2
    count = 0
    for k,v in frequent.items():
        name = v[1]
        col = v[-1]
        v = v[0]
        v = [int(p) for p in np.median(v, axis=0)]
        v = v[:max_arm]
        rects_k = plot.bar(start + count * width, v, width, label=name, color = col)
        count += 1
    plot.ylabel('Frequency')
    plot.xlabel('Arm')
    plot.title('number of times algorithm pulls arm 1-{}'.format(max_arm))
    plot.xticks(x + width*1, labels)
    plot.legend(frameon=False)
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    fig.savefig('plots/' + fn + '.pdf', dpi=300, bbox_inches = "tight")
    
def draw_figure():
    plot_style = {
            'ucb-glm': ['-.', 'green', 'UCB-GLM'],
            'sgd-ts': ['-', 'red', 'SGD-TS'],
            'gloc': ['--', 'orange', 'GLOC'],
            'lts': [':', 'blue', 'Laplace-TS']
        }
    plot_prior = {
            'sgd-ts': 1,
            'ucb-glm': 2,
            'gloc': 3,
            'lts': 4
        }
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
        
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'covtype' not in c and 'yahoo' not in c and 'simulation' not in c: continue
        folders = os.listdir(root+c)
        paths.append(root + c + '/')
    for path in paths:
        fn = path.split('/')[-2]
        if 'freq' in fn: 
            plot_frequency(path)
            continue
        if 'yahoo' in fn:
            title = 'News Article Recommendation Data'
        elif 'covtype' in fn:
            d = int(fn.split('_')[1][1:])
            title = 'Forest Cover Type Data (d = {})'.format(d)
        elif 'simulation' in fn:
            _, dstr, Kstr = fn.split('_')
            d = int(dstr[1:])
            K = int(Kstr[1:])
            title = 'Simulation, d={}, K={}'.format(d, K)
        else:
            continue
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)
        leg = []
        keys = os.listdir(path)
        keys = sorted(keys, key=lambda kv: plot_prior[kv])
        y_label = 'Cumulative Regret'
        for key in keys:
            if key not in plot_style.keys(): continue
            leg += [plot_style[key][-1]]
            data = np.loadtxt(path+key)
            T = len(data)
            if 'yahoo' in path:
                data *= -1
                data = [data[i]/(i+1) for i in range(T)]
                data = data[1:]
                T -= 1
                y_label = 'Click Through Rate / time'
            plot.plot((list(range(T))), data, linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
        plot.legend((leg), loc='best', fontsize=16, frameon=False)
        plot.xlabel('Time')
        if 'yahoo' in path:
            dates = ['May0'+str(i) for i in range(1,10)] + ['May10'] + ['May11']
            tmpx = list(range(T))
            plot.xticks(tmpx[::288] + [tmpx[-1]], dates, rotation=45)
        plot.ylabel(y_label)
        plot.title(title)
        fig.savefig('plots/' + fn + '.pdf', dpi=300, bbox_inches = "tight")

draw_figure()


