#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


filename1 = sys.argv[1]
filename2 = sys.argv[2]


# In[2]:


data_one = pd.read_csv(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
data_two = pd.read_csv(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])

one_sorted = data_one.sort_values(by='views', ascending=False)
data_one['temp'] = data_two['views']


# In[3]:


plt.figure(figsize=(10, 5)) # change the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.title('Popularity Distribution')
plt.xlabel('Rank')
plt.ylabel('Views')
plt.plot(one_sorted['views'].values) # build plot 1


plt.subplot(1, 2, 2) # select the second
plt.title('Daily Correlation')
plt.xlabel('           Day 1 views')
plt.ylabel('           Day 2 views')

plt.xscale('log')
plt.yscale('log')
plt.scatter(data_one['views'],data_one['temp']) # build plot 2
plt.savefig('wikipedia.png')


# In[ ]:




