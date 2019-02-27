# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:12:11 2018

@author: damodara
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fname = 'cifar10_sym_n_0.4_drop_0.0_results.hd5.npz'
fname_d = 'cifar10_sym_n_0.4_drop_0.5_results.hd5.npz'


load = np.load(fname)
load_d = np.load(fname_d)


acc_list = [s for s in load.keys() if 'acc' in s]
acc_list.sort()
accdrop_list = [s for s in load_d.keys() if 'acc' in s]
accdrop_list.sort()

epochs = range(1,len(load[acc_list[0]]) + 1)

col = ['r','g', 'b']
fig = plt.figure(num=1)
count=0
for i in acc_list:
    plt.plot(epochs, load[i], color = col[count], label = i, )
    count = count+1
    
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc='best')
plt.title('with L2 reg and without dropout')

fig = plt.figure(num=2)
count=0
for i in accdrop_list:
    plt.plot(epochs, load_d[i], color = col[count], label = 'dropout'+i)
    count = count+1

plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc='best')
plt.title('with L2 reg and with dropout')


    
