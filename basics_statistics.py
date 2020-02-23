### Basic Statistical Analysis ###

#%%
import random
y = [random.random()*100 for i in range(10)]
print(y)

n=10
y=[0]*n
for i in range(n):
    y[i]=random.random()*100
    print(y[i])
print(max(y))
print(min(y))

#%%
import numpy as np
print(np.mean(y))
print(np.average(y))
print(np.std(y))
print(np.median(y))

#%%
import pandas as pd
df = pd.read_csv("/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120/data_1.csv") # data frame is at the core of Pandas
df
df = pd.read_excel("/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120/data_1.xlsx","sheet1")
df
df = pd.DataFrame(my_dict)

#%%
df = pd.read_csv('/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120'
                 '/DataFiles/data_1.csv',
                 nrows=2)
df
df = pd.read_csv("/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120/data_1.csv",na_values=["n.a."])
df
import os # importing a module
os.getcwd()
os.chdir('/Users/Sam/Box/0-Sam/MyGitHub/ISE-4120')
os.getcwd()
df.to_csv("data_new.csv")
df.to_csv("data_new.csv",index=False)
df.to_csv("data_new.csv",columns=['temperature','day'])
df.to_csv("data_new.csv",header=False)
# similarly we have df.to_excel()
df.shape
rows, columns=df.shape
rows
columns
df.head() # see top rows
df.tail(2) # see botttom 2 rows
df[2:5]
df.columns
df.day
df.event
df['event']
type(df.event)
df[['event','temperature']]

#%%
df['temperature'].max()
df.describe() #only acts on numerical columns
df[df.temperature>=29]
df[df.temperature==df['temperature'].max()]
df['day'][df.temperature==df['temperature'].max()]
df.day[df.temperature==df['temperature'].max()]

df.index
df.set_index('day')
df
df.set_index('day',inplace=True)
df
df.loc['1/4/20']
df.reset_index(inplace=True)
df

#%% Dealing with missing data
df=pd.read_csv('data_1_missing.csv')
df
df=pd.read_csv('data_1_missing.csv',parse_dates=["day"])
df
df.set_index('day',inplace=True)
df
new_df=df.fillna(0)
new_df
new_df=df.fillna({
        'temperature':0,
        'windspeed':0,
        'event': 'no_event'
        })
new_df
new_df=df.fillna(method="ffill")
new_df
new_df=df.fillna(method="bfill")
new_df
new_df=df.fillna(method="ffill",limit=1)
new_df

new_df=df.interpolate()  # by default, it is linear interpolation
new_df
new_df=df.interpolate(method="time") # takes time as the x-axis
new_df

new_df=df.dropna() # if at least one na in the row, it will be dropped
new_df
new_df=df.dropna(how="all") # all na rows will be dropped
new_df
new_df=df.dropna(thresh=2) # need at least two non-na to keep
new_df
