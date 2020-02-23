#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:44:02 2020

@author: Sam Davanloo

Statistical Inference

"""
## Rule
# If p_value is greater than alpha, then null cannot be rejected; otherwise, it will be rejected.


#################### 1. Inference on the mean of Normal distribution (variance known)  ##########
# required modules 
import numpy as np
from scipy.stats import norm


# H_0: mu=mu_0
# H_A: mu>mu_0  <---------
mu_0=76

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
p_value=1-norm.cdf(z_0,loc=0,scale=1)
print(p_value)


## lower confidence interval
z_alpha = norm.isf(0.05, loc=0, scale=1) # let alpha=0.05
LB=x_bar-z_alpha*sigma/np.sqrt(n)
CI=[LB,float('inf')] #Lower CI for H_A: mu>mu_0, reject if LB>mu_0
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu<mu_0  <---------
mu_0=88

# simulating the data
sigma=12
n=5
np.random.seed(8) 
data = np.random.normal(loc=90, scale=sigma, size=n)

# calculate the z-statistic
x_bar = data.mean()
z_0=(x_bar-mu_0)/(sigma/np.sqrt(n))
print(z_0)

# calculate the p_value
p_value=norm.cdf(z_0,loc=0,scale=1)
print(p_value) 


### upper confidence interval
z_alpha = norm.isf(0.05, loc=0, scale=1) # let alpha=0.05
UB=x_bar+z_alpha*sigma/np.sqrt(n)
CI=[-float('inf'),UB] #Upper CI for H_A: mu<mu_0, reject if UB<mu_0
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
CI=[LB,UB] # reject if UB<mu_0, or LB>mu_0
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
print("t-stat is: "+str(tstat))
print("p-value is: "+str(p_value))


## lower confidenfce interval
x_bar=data.mean()
s=np.std(data,ddof=1)
t_alpha = sc.stats.t.isf(0.05, df=n-1) # let alpha=0.05
print(t_alpha)
LB=x_bar-t_alpha*s/np.sqrt(n)
CI=[LB,float('inf')]
print(CI)

####################
# H_0: mu=mu_0
# H_A: mu<mu_0  <---------
mu_0=109

# simulating the data
sigma=12
n=10
np.random.seed(8) 
data=np.random.normal(loc=95,scale=10,size=n)

# t-test
tstat,p_value=weightstats.ztest(data,value=mu_0,alternative='smaller')
print("t-stat is: "+str(tstat))
print("p-value is: "+str(p_value))


## upper confidence interval
x_bar=data.mean()
s=np.std(data,ddof=1)
t_alpha = sc.stats.t.isf(0.05, df=n-1) # let alpha=0.05
print(t_alpha)
UB=x_bar+t_alpha*s/np.sqrt(n)
CI=[-float('inf'),UB]
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
print("t-stat is: "+str(tstat))
print("p-value is: "+str(p_value))


## confidence interval
x_bar=data.mean()
s=np.std(data,ddof=1)
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
sigma0_p2=120

# simulating the data
sigma=10
n=10
np.random.seed(8) 
data=np.random.normal(loc=100,scale=10,size=n)


s=np.std(data,ddof=1)
chi0_p2=(n-1)*pow(s,2)/sigma0_p2
print(chi0_p2)
chi_p2_alpha = sc.stats.chi2.isf(0.05,n-1)
print(chi_p2_alpha)
p_value=1-sc.stats.chi2.cdf(chi0_p2,n-1)
print(p_value)


## confidence interval
s=np.std(data)
chi2_alpha = sc.stats.chi2.isf(0.05, df=n-1) # let alpha=0.05
LB=(n-1)*pow(s,2)/chi2_alpha
CI=[LB,float('inf')]
print(CI)


### calculating the p-value for two-sided chi-square test
# H_0: sigma^2=sigma0^2
# H_A: sigma^2 \neq sigma0^2  <---------

s=np.std(data,ddof=1)
chi0_p2=(n-1)*pow(s,2)/sigma0_p2
print(chi0_p2)
p_value=2*min(sc.stats.chi2.cdf(chi0_p2,n-1),sc.stats.chi2.sf(chi0_p2,n-1))
print(p_value)

#################### 4. Inference on the Proportions   ##########
# required modules 
import numpy as np
import scipy as sc

# H_0: p=p_0
# H_A: p \neq p_0  <---------
p_0=0.5

x=46 # let x be the number of defects in n products
n=100

z_alpha_div_2=sc.stats.norm.isf(0.025,loc=0,scale=1)
print(z_alpha_div_2)

if x < n*p_0:
    z_0=((x+0.5)-n*p_0)/np.sqrt(n*p_0*(1-p_0))
else:
    z_0=((x-0.5)-n*p_0)/np.sqrt(n*p_0*(1-p_0))
    
print(z_0)

p_value=2*sc.stats.norm.sf(np.abs(z_0))
print(p_value)


## condidence interval
p_hat=x/n
z_alpha_div_2=sc.stats.norm.isf(0.025,loc=0,scale=1)
UB=p_hat+z_alpha_div_2*np.sqrt(p_hat*(1-p_hat)/n)
LB=p_hat-z_alpha_div_2*np.sqrt(p_hat*(1-p_hat)/n)
CI=[LB,UB]
print(CI)

#################### 5. Inference on the difference of means  ##########
        
#################### 5.1. Inference on the difference of means, variance known  ##########
# required modules 
import numpy as np
import scipy as sc

# H_0: mu_1-mu_2 = Delta
# H_A: mu_1-mu_2 \neq Delta  <---------
Delta=5

# simulating the data
sigma_1=10
sigma_2=8
n_1=10
n_2=12
np.random.seed(8) 
data_1 = np.random.normal(loc=110, scale=sigma_1, size=n_1)
data_2 = np.random.normal(loc=101, scale=sigma_2, size=n_2)


# z-test
x_bar_1=data_1.mean()
x_bar_2=data_2.mean()
z=((x_bar_1-x_bar_2)-(Delta))/np.sqrt(pow(sigma_1,2)/n_1+pow(sigma_2,2)/n_2)
p_value=2*sc.stats.norm.sf(abs(z))
print(p_value)


## condidence interval
z_alpha_div_2=sc.stats.norm.isf(0.025,loc=0,scale=1)
UB=x_bar_1-x_bar_2+z_alpha_div_2*np.sqrt(pow(sigma_1,2)/n_1+pow(sigma_2,2)/n_2)
LB=x_bar_1-x_bar_2-z_alpha_div_2*np.sqrt(pow(sigma_1,2)/n_1+pow(sigma_2,2)/n_2)
CI=[LB,UB]
print(CI)


###### diagnoostics

#%matplotlib inline
import matplotlib.pyplot as plt

# checking for normality
fig = plt.figure()
ax1 = fig.add_subplot(121)
data1_fg = sc.stats.probplot(data_1,dist='norm',plot=ax1)
ax2 = fig.add_subplot(122)
prob_plot = sc.stats.probplot(data_2,dist='norm',plot=ax2)
plt.show()



#################### 5.2. Inference on the difference of means, variance unknown  ##########
# required modules 
import numpy as np
import scipy as sc

##### Case I : Variance unKnown, but equal

# H_0: mu_1-mu_2 = Delta
# H_A: mu_1-mu_2 \neq Delta  <---------
Delta=5

# simulating the data
sigma_1=8
sigma_2=8
n_1=10
n_2=12
np.random.seed(8) 
data_1 = np.random.normal(loc=110, scale=sigma_1, size=n_1)
data_2 = np.random.normal(loc=101, scale=sigma_2, size=n_2)

# t-test
x_bar_1=data_1.mean()
x_bar_2=data_2.mean()
s2_1=np.var(data_1,ddof=1)
s2_2=np.var(data_2,ddof=1)
s2_p=((n_1-1)*s2_1+(n_2-1)*s2_2)/(n_1+n_2-2)   # pooled variance estimator
t=(x_bar_1-x_bar_2-Delta)/(np.sqrt(s2_p)*np.sqrt(1/n_1+1/n_2))
p_value=2*sc.stats.t.sf(abs(t),n_1+n_2-2)
print(p_value)


## condidence interval
t_alpha_div_2=sc.stats.t.isf(0.025,n_1+n_2-2)
UB=x_bar_1-x_bar_2+t_alpha_div_2*np.sqrt(s2_p)*np.sqrt(1/n_1+1/n_2)
LB=x_bar_1-x_bar_2-t_alpha_div_2*np.sqrt(s2_p)*np.sqrt(1/n_1+1/n_2)
CI=[LB,UB]
print(CI)

# checking constant variance
data = [data_1, data_2]
fig = plt.figure()
ax = fig.add_subplot(111)
bp_plot = ax.boxplot(data)
plt.show()


##### Case II : Variance Unknown, and unequal

# H_0: mu_1-mu_2 = Delta
# H_A: mu_1-mu_2 \neq Delta  <---------
Delta=5

# simulating the data
sigma_1=8
sigma_2=10
n_1=10
n_2=12
np.random.seed(8) 
data_1 = np.random.normal(loc=110, scale=sigma_1, size=n_1)
data_2 = np.random.normal(loc=101, scale=sigma_2, size=n_2)

# t-test
x_bar_1=data_1.mean()
x_bar_2=data_2.mean()
s2_1=np.var(data_1,ddof=1)
s2_2=np.var(data_2,ddof=1)
t=(x_bar_1-x_bar_2-Delta)/np.sqrt(s2_1/n_1+s2_2/n_2)
nu=pow(s2_1/n_1+s2_2/n_2,2)/(pow(s2_1/n_1,2)/(n_1-1)+pow(s2_2/n_2,2)/(n_2-1))-2
print(nu)
p_value=2*sc.stats.t.sf(abs(t),nu)
print(p_value)


## condidence interval
t_alpha_div_2=sc.stats.t.isf(0.025,nu)
UB=x_bar_1-x_bar_2+t_alpha_div_2*np.sqrt(s2_1/n_1+s2_2/n_2)
LB=x_bar_1-x_bar_2-t_alpha_div_2*np.sqrt(s2_1/n_1+s2_2/n_2)
CI=[LB,UB]
print(CI)


#################### 6. Inference on more than two populations (ANOVA)  ##########
import pandas as pd

# generate data
A = [12.6, 12, 11.8, 11.9, 13, 12.5, 14]
B = [10, 10.2, 10, 12, 14, 13]
C = [10.1, 13, 13.4, 12.9, 8.9, 10.7, 13.6, 12]
all_scores = A + B + C
print(all_scores)
company_names = (['A'] * len(A)) +  (['B'] * len(B)) +  (['C'] * len(C))
data = pd.DataFrame({'company': company_names, 'score': all_scores})
data

# descriptive statistics
data.groupby('company').mean()

# one-way ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
lm = ols('score ~ company',data=data).fit()
table = sm.stats.anova_lm(lm)
print(table)




