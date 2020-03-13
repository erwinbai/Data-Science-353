#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np


# In[9]:


data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']


# In[16]:


sum = (np.sum(totals, axis = 1))
print('Row with lowest total precipitation:')
sum_arg = np.argmin(sum, axis = 0)
print(sum_arg)


# In[15]:


sum_totals = (np.sum(totals, axis = 0))
sum_counts = (np.sum(counts, axis = 0))
print('Average precipitation in each month:')
print(sum_totals/sum_counts)


# In[17]:


sum_totals = (np.sum(totals, axis = 1))
sum_counts = (np.sum(counts, axis = 1))
print('Average precipitation in each city:')
print(sum_totals/sum_counts)


# In[18]:


new_totals = np.reshape(totals,(9,4,3))
quater_totals = np.sum(new_totals,axis = 2)
print(quater_totals)

