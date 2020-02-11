#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:44:02 2020

@author: Sam Davanloo

Statistical Inference

"""
## Rule
# If p_value is greater than alpha, then null cannot be rejected; otherwise, it will be rejected.


#################### 1. Inference on the mean of Normal distribution (variance knwon)  ##########
# required modules 
import numpy as np
from scipy.stats import norm


# H_0: mu=mu_0
# H_A: mu>mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=5
np.random.seed(8) 
data = np.random.normal(loc=96, scale=sigma, size=n)

# calculate the z-statistic
x_bar = data.mean()
z_0=(x_bar-mu_0)/(sigma/np.sqrt(n))
print(z_0)

# calculate the p_value
p_value=norm.cdf(z_0,loc=0,scale=1)
print(p_value)


## upper confidence interval
z_alpha = norm.isf(0.05, loc=0, scale=1) # let alpha=0.05
UB=x_bar+z_alpha*sigma/np.sqrt(n)
CI=[-float('inf'),UB]
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu<mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=5
np.random.seed(8) 
data = np.random.normal(loc=110, scale=sigma, size=n)

# calculate the z-statistic
x_bar = data.mean()
z_0=(x_bar-mu_0)/(sigma/np.sqrt(n))
print(z_0)

# calculate the p_value
p_value=1-norm.cdf(z_0,loc=0,scale=1)
print(p_value) 


### lower confidence interval
z_alpha = norm.isf(0.05, loc=0, scale=1) # let alpha=0.05
LB=x_bar-z_alpha*sigma/np.sqrt(n)
CI=[LB,float('inf')]
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu \neq mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=5
np.random.seed(8) 
data = np.random.normal(loc=91, scale=sigma, size=n)

# calculate the z-statistic
x_bar = data.mean()
z_0=(x_bar-mu_0)/(sigma/np.sqrt(n))
print(z_0)

# calculate the p_value
p_value=2*(1-norm.cdf(np.absolute(z_0),loc=0,scale=1))
print(p_value)


### confidence interval
z_alpha_div_2 = norm.isf(0.025, loc=0, scale=1) # let alpha=0.05
LB=x_bar-z_alpha_div_2*sigma/np.sqrt(n)
UB=x_bar+z_alpha_div_2*sigma/np.sqrt(n)
CI=[LB,UB]
print(CI)


#################### 2. Inference on the mean of Normal distribution (variance unknwon)  ##########
# required modules 
import numpy as np
from statsmodels.stats import weightstats
import scipy as sc


# H_0: mu=mu_0
# H_A: mu>mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=10
np.random.seed(8) 
data=np.random.normal(loc=102,scale=10,size=n)

# t-test
tstat,p_value=weightstats.ztest(data,value=mu_0,alternative='larger')
print("p-value is: "+str(p_value))
print("t-stat is: "+str(tstat))


## upper confidence interval
s=np.std(data)
t_alpha = sc.stats.t.isf(0.05, df=n-1) # let alpha=0.05
print(t_alpha)
UB=x_bar+t_alpha*s/np.sqrt(n)
CI=[-float('inf'),UB]
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu<mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=10
np.random.seed(8) 
data=np.random.normal(loc=95,scale=10,size=n)

# t-test
tstat,p_value=weightstats.ztest(data,value=mu_0,alternative='smaller')
print("p-value is: "+str(p_value))
print("t-stat is: "+str(tstat))


## lower confidence interval
s=np.std(data)
t_alpha = sc.stats.t.isf(0.05, df=n-1) # let alpha=0.05
print(t_alpha)
LB=x_bar-t_alpha*s/np.sqrt(n)
CI=[LB,float('inf')]
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu \neq mu_0  <---------
mu_0=100

# simulating the data
sigma=12
n=10
np.random.seed(8) 
data=np.random.normal(loc=95,scale=10,size=n)

# t-test
tstat,p_value=weightstats.ztest(data,value=mu_0,alternative='two-sided')
print("p-value is: "+str(p_value))
print("t-stat is: "+str(tstat))


## confidence interval
s=np.std(data)
t_alpha_div_2 = sc.stats.t.isf(0.025, df=n-1) # let alpha=0.05
LB=x_bar-t_alpha_div_2*s/np.sqrt(n)
UB=x_bar+t_alpha_div_2*s/np.sqrt(n)
CI=[LB,UB]
print(CI)

#################### 3. Inference on the variance of Normal distribution  ##########
# required modules 
import numpy as np
import scipy as sc

# H_0: sigma^2=sigma0^2
# H_A: sigma^2>sigma0^2  <---------
sigma0_p2=100

# simulating the data
sigma=10
n=10
np.random.seed(8) 
data=np.random.normal(loc=100,scale=10,size=n)


s=np.std(data)
chi0_p2=(n-1)*pow(s,2)/sigma0_p2
print(chi0_p2)
chi_p2_alpha = sc.stats.chi2.isf(0.05,n-1)
print(chi_p2_alpha)
p_value=1-sc.stats.chi2.cdf(chi0_p2,n-1)
print(p_value)






