#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basics of plotting

"""

import numpy as np
import matplotlib.pyplot as plt

apl_price=[93.95,112.15,104.05,144.85,169.49]
ms_price=[39.01,50.29,57.05,69.98,94.39]
year=[2014,2015,2016,2017,2018]

plt.plot(year,apl_price)
plt.xlabel('Year')
plt.ylabel('Stock Price')
#plt.show()

plt.plot(year,apl_price,':k',year,ms_price,'--r')
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.axis([2013,2019,35,170])


fig_1=plt.figure(1,figsize=(9.6,2.8))
chart_1=fig_1.add_subplot(121)
chart_2=fig_1.add_subplot(122)
chart_1.plot(year,apl_price)
chart_2.scatter(year,ms_price)


fig_2, axes=plt.subplots(2,2,figsize=(5,3))
axes[0,1].scatter(year,ms_price)
axes[1,0].plot(year,apl_price)


#%% Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2

# Two signals with a coherent part at 10Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

fig, axs = plt.subplots(2, 1)
axs[0].plot(t, s1, t, s2)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
axs[1].set_ylabel('coherence')

fig.tight_layout()
plt.show()