#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import stats

def to_timestamp(d):
    return d.timestamp()


# In[2]:


data = pd.read_csv('dog_rates_tweets.csv',parse_dates=[1])


# In[3]:


data['rating'] = data['text'].str.extract(r'(\d+(\.\d+)?)/10', expand = False)[0]
#print(data)
data = data.dropna()
#print(data)


# In[4]:


data['rating'] = data['rating'].astype(float)
data = data[data['rating'] <= 25]
#print(data)


# In[5]:


data['timestamp'] = data['created_at'].apply(to_timestamp)
fit = stats.linregress(data['timestamp'],data['rating'])
#print (data)


# In[6]:


data['prediction'] = data['timestamp']*fit.slope + fit.intercept


# In[7]:


fit.slope, fit.intercept


# In[8]:


plt.xticks(rotation=115)
plt.plot(data['created_at'],data['rating'],'b.',alpha=0.5)
plt.plot(data['created_at'], data['prediction'], 'r-', linewidth=3)
plt.show()


# In[9]:


p = fit.pvalue
print('P-value for dog rates is')
print(p)


# In[10]:


residual = data['rating'] - data['prediction']
print(residual)


# In[11]:


plt.hist(residual)

