#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:47:50 2020

@author: Sam
"""


################## STATSMODELS ##################


import numpy as np
import statsmodels.api as sm

spector_data = sm.datasets.spector.load(as_pandas=False)
type(spector_data)
print(spector_data.exog)
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
print(spector_data.exog)
print(spector_data.endog)

mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())

##################

from pandas import DataFrame
import statsmodels.api as sm

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

df = DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price']) 

X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for the multiple linear regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example
Y = df['Stock_Index_Price']
X = sm.add_constant(X) # adding a constant
print(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
print_model = model.summary()
print(print_model)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(Y,predictions,'o')
plt.xlabel('Stock Index Price')
plt.ylabel('Predictions')


################## Scikit-Learn ##################
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# read data into a DataFrame
data = pd.read_csv('http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv')
data.head()
data.shape
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=2.5, aspect=0.8)

# create X and y
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

# instantiate and fit
lm = LinearRegression()
lm.fit(X, y)

# print the coefficients
print(lm.intercept_)
print(lm.coef_)

sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', size=3.5, aspect=0.7, kind='reg')

