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


#################### CUSUM & EWMA Control Charts  ####################

#################### CUSUM control chart ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9) 
m1=20
m2=10
m=m1+m2
mu0=10
data1=np.random.normal(loc=mu0,scale=1,size=m1)
data2=np.random.normal(loc=mu0+1,scale=1,size=m2)
data=np.concatenate((data1,data2),axis=0)
print(data)

c_pls,c_neg=np.zeros(m+1),np.zeros(m+1)
K=1/2    
for i in range(1,m):
    c_pls[i]=np.maximum(0,data[i]-(mu0+K)+c_pls[i-1]) #Eq. (9.2)
    c_neg[i]=np.maximum(0,(mu0-K)-data[i]+c_neg[i-1]) #Eq. (9.3)

H=5
itr=list(range(m))
plt.figure()
plt.plot(itr,c_pls[range(m)],'ob-',itr,c_neg[range(m)],'ok-',itr,H*np.ones(m),'r-')
plt.xlabel('Observation')
plt.title('CUSUM Control Chart')
plt.legend(['C_pls','C_neg','H'])


#################### EWMA control chart ##########
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9) 
m1=20
m2=10
m=m1+m2
mu0=10
sigma=1
data1=np.random.normal(loc=mu0,scale=sigma,size=m1)
data2=np.random.normal(loc=mu0+1,scale=sigma,size=m2)
data=np.concatenate((data1,data2),axis=0)
print(data)


z=np.zeros(m+1)
z[m]=mu0   
lam=0.1
L=2.7
lcl=np.zeros(m)
ucl=np.zeros(m)
for i in range(0,m):
    z[i]=lam*data[i]+(1-lam)*z[i-1]  #Eq. (9.22)
    lcl[i]=mu0-L*sigma*np.sqrt((lam/(2-lam))*(1-np.power((1-lam),(2*(i+1)))))   #Eq. (9.25)
    ucl[i]=mu0+L*sigma*np.sqrt((lam/(2-lam))*(1-np.power((1-lam),(2*(i+1)))))#Eq. (9.26)

itr=list(range(m))
plt.figure()
plt.plot(itr,z[range(m)],'ob-',itr,mu0*np.ones(m),'k-',itr,lcl,'r-',itr,ucl,'r-')
plt.xlabel('Observation')
plt.title('EWMA Control Chart')
plt.legend(['EWMA','mu0','LCL','UCL'])   
    

