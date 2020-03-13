#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd


# In[11]:


totals = pd.read_csv('totals.csv').set_index(keys = ['name'])
counts = pd.read_csv('counts.csv').set_index(keys = ['name'])


# In[12]:


sum = totals.sum(axis = 1)


# In[13]:


print(sum.idxmin())


# In[14]:


totals_sum = totals.sum(axis = 0)
totals_count = counts.sum(axis = 0)


# In[16]:


average = totals_sum/totals_count
print('Average precipitation in each month:')
print(average)


# In[8]:


totals_sum = totals.sum(axis = 1)
totals_count = counts.sum(axis = 1)


# In[18]:


average = totals_sum/totals_count
print('Average precipitation in each city:')
print(average)

