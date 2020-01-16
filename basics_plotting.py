#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basics of plotting

"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120')
os.getcwd()

apl_price=[93.95,112.15,104.05,144.85,169.49]
ms_price=[39.01,50.29,57.05,69.98,94.39]
year=[2014,2015,2016,2017,2018]

plt.figure()
plt.plot(year,apl_price)
plt.xlabel('Year')
plt.ylabel('Stock Price')
#plt.show()

plt.plot(year,apl_price,':k',year,ms_price,'--r')
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.axis([2013,2019,35,170])   #range on each axis
plt.savefig('Figures/test_plot.pdf') 

fig_1=plt.figure(1,figsize=(9.6,2.8))
chart_1=fig_1.add_subplot(121)
chart_2=fig_1.add_subplot(122)
chart_1.plot(year,apl_price)
chart_2.scatter(year,ms_price)

fig_2, sp=plt.subplots(2,2,figsize=(5,3))
sp[0,1].scatter(year,ms_price)
sp[1,0].plot(year,apl_price)


#%%
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,5,100)
y=np.sin(4*x)*np.exp(-x)
z=np.cos(4*x)*np.exp(-x)
plt.figure()
plt.plot(x,y,'r--',x,z,'b',linewidth=4.0)
plt.legend(['$sin(4x)*e^{-x}$','$cos(4x)*e^{-x}$'])
plt.xlabel('x')
plt.savefig('Figures/fig_test.pdf')

