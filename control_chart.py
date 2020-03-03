#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:29:08 2020

@author: Sam
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

m=30
n=5
data=np.random.normal(loc=10,scale=2,size=(m,n))
print(data)

x_bar_i=np.mean(data,axis=1)
R_i=np.max(data,axis=1)-np.min(data,axis=1)
x_bar_bar=np.mean(x_bar_i)
R_bar=np.mean(R_i)

A2=0.577    # from Appendix VI
D3=0        # from Appendix VI
D4=2.114    # from Appendix VI

CL_x=x_bar_bar
LCL_x=x_bar_bar-A2*R_bar
UCL_x=x_bar_bar+A2*R_bar

CL_R=R_bar
LCL_R=D3*R_bar
UCL_R=D4*R_bar

itr=list(range(m))

plt.figure()
plt.plot(itr,x_bar_i,'ob-',itr,CL_x*np.ones(m),'k-',itr,LCL_x*np.ones(m),'r-',itr,UCL_x*np.ones(m),'r-')
plt.xlabel('Subgroup')
plt.title('Sample mean')
plt.legend(['$\overline{x}_i$','CL','LCL','UCL'])

plt.figure()
plt.plot(itr,R_i,'ob-',itr,CL_R*np.ones(m),'k-',itr,LCL_R*np.ones(m),'r-',itr,UCL_R*np.ones(m),'r-')
plt.xlabel('Subgroup')
plt.title('Sample range')
plt.legend(['$R_i$','CL','LCL','UCL'],loc='upper right')