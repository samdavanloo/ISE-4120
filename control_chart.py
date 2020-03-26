#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:29:08 2020

@author: Sam
"""

#################### Control Charts for Variables  ####################


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


#################### X-bar and R control chart  ##########
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


#################### X-bar and S control chart  ##########
m=30
n=5
data=np.random.normal(loc=10,scale=2,size=(m,n))
print(data)

x_bar_i=np.mean(data,axis=1)
s_i=np.std(data,axis=1,ddof=1)
x_bar_bar=np.mean(x_bar_i)
s_bar=np.mean(s_i)

A3=1.427    # from Appendix VI
B3=0        # from Appendix VI
B4=2.089    # from Appendix VI

CL_x=x_bar_bar
LCL_x=x_bar_bar-A3*s_bar
UCL_x=x_bar_bar+A3*s_bar

CL_s=s_bar
LCL_s=B3*s_bar
UCL_s=B4*s_bar

itr=list(range(m))

plt.figure()
plt.plot(itr,x_bar_i,'ob-',itr,CL_x*np.ones(m),'k-',itr,LCL_x*np.ones(m),'r-',itr,UCL_x*np.ones(m),'r-')
plt.xlabel('Subgroup')
plt.title('$\overline{x}$ Control Chart')
plt.legend(['$\overline{x}_i$','CL','LCL','UCL'])

plt.figure()
plt.plot(itr,s_i,'ob-',itr,CL_s*np.ones(m),'k-',itr,LCL_s*np.ones(m),'r-',itr,UCL_s*np.ones(m),'r-')
plt.xlabel('Subgroup')
plt.title('$s$ Control Chart')
plt.legend(['$s_i$','CL','LCL','UCL'],loc='upper right')



#################### Control Charts for Attributes  ####################

#################### p control chart  ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8) 

m=30   # number of samples
n=50   # sample size
D=np.random.randint(19,size=m) # number of nonconformings
print(D)

p_bar=np.sum(D)/(m*n)  # Eq. (7.7)
print(p_bar)

p=D/n
CL=p_bar     # Eq. (7.8)
UCL=p_bar+3*np.sqrt(p_bar*(1-p_bar)/n)    # Eq. (7.8)
LCL=p_bar-3*np.sqrt(p_bar*(1-p_bar)/n)    # Eq. (7.8)


itr=list(range(m))

plt.figure()
plt.plot(itr,p,'ob-',itr,CL*np.ones(m),'k-',itr,LCL*np.ones(m),'r-',itr,UCL*np.ones(m),'r-')
plt.xlabel('Sample')
plt.title('p Control Chart')
plt.legend(['$p$','CL','LCL','UCL'])


#################### np control chart  ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8) 

m=30   # number of samples
n=50   # sample size
D=np.random.randint(19,size=m) # number of nonconformings
print(D)

p_bar=np.sum(D)/(m*n)
print(p_bar)


# Calculation below is based on (7.13)
CL=n*p_bar
UCL=n*p_bar+3*np.sqrt(n*p_bar*(1-p_bar))
LCL=n*p_bar-3*np.sqrt(n*p_bar*(1-p_bar))

itr=list(range(m))

plt.figure()
plt.plot(itr,D,'ob-',itr,CL*np.ones(m),'k-',itr,LCL*np.ones(m),'r-',itr,UCL*np.ones(m),'r-')
plt.xlabel('Sample')
plt.title('np Control Chart')
plt.legend(['D','CL','LCL','UCL'])


#################### c control chart  ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8) 

m=30   # number of samples (note that n=1 here)
num_defects=np.random.poisson(lam=22,size=m) # lambda is the parameter of poisson distribution 
print(num_defects)

c_bar=np.sum(num_defects)/m
print(c_bar)


# Eq. (7.17)
CL=c_bar
UCL=c_bar+3*np.sqrt(c_bar)
LCL=c_bar-3*np.sqrt(c_bar)

itr=list(range(m))

plt.figure()
plt.plot(itr,num_defects,'ob-',itr,CL*np.ones(m),'k-',itr,LCL*np.ones(m),'r-',itr,UCL*np.ones(m),'r-')
plt.xlabel('Sample')
plt.title('c Control Chart')
plt.legend(['N0. of defects','CL','LCL','UCL'])



#################### u control chart  ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8) 

m=30   # number of samples
n=50
total_num_defects=np.random.poisson(lam=5,size=m) # total number of defects in n items in a sample
print(total_num_defects)
u=total_num_defects/n
print(u)

u_bar=np.sum(total_num_defects)/(m*n)
print(u_bar)

# Eq. (7.19)
CL=u_bar
UCL=u_bar+3*np.sqrt(u_bar/n)
LCL=u_bar-3*np.sqrt(u_bar/n)

itr=list(range(m))

plt.figure()
plt.plot(itr,u,'ob-',itr,CL*np.ones(m),'k-',itr,LCL*np.ones(m),'r-',itr,UCL*np.ones(m),'r-')
plt.xlabel('Sample')
plt.title('u Control Chart')
plt.legend(['average N0. of defects','CL','LCL','UCL'])
