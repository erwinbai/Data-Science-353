#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import scipy.stats as stats


# In[2]:


df = pd.read_csv('data.csv')
print(df)


# In[3]:


#Since we have more than one group we need ANOVA
anova = stats.f_oneway(df['qs1'],df['qs2'],df['qs3'],df['qs4'],df['qs5'],df['merge1'],df['partition_sort']).pvalue
print(anova)


# In[4]:


print('Since our P=0.0 < 0.005, there is a difference between the means of the groups')
#We know that there are some groups with different means but not sure which one


# In[5]:


#Continue with Post Hoc Analysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
x_melt = pd.melt(df)
#print(x_melt)


# In[6]:


posthoc = pairwise_tukeyhsd(
    x_melt['value'], x_melt['variable'],
    alpha=0.05)
#print(posthoc)


# In[7]:


fig = posthoc.plot_simultaneous()


# In[8]:


print('7 sorting algorithm ANOVA test P-value:',anova)
print('7 sorting algorithm posthoc:\n', posthoc)


# In[16]:


mean = df.mean(axis=0)
print('Mean of 7 sorting algorithm:\n',mean)

sort_mean = mean.sort_values()
print('Sorted mean of 7 sorting algorithm:\n',sort_mean)


# In[ ]:




