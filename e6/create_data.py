#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np


# In[2]:


random = np.random.rand(3000) #array with random integer of size 3000
#print(random)
# max = random.max()
# print(max)


# In[3]:


import time
from implementations import all_implementations
#... contains all_implementations = [qs1, qs2, qs3, qs4, qs5, merge1, partition_sort



count = np.empty((7,100)) #large n=100 and count through all 7 sort
j = 0
#print(result)

for sort in all_implementations:
    for i in range(100):  #large n and sort 100 times   
        st = time.time()
        res = sort(random)
        en = time.time()
        count[j][i] = en - st #store first time caculation for first result in to alg[0][0] etc..
    j =j+1
    
#print(count)


# In[4]:


#I need to transpose the data so every sort alg have the right time
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
col = np.transpose(count)

# Found a interesting one that also transpose array
# https://stackoverflow.com/questions/4937491/matrix-transpose-in-python
# I am still unsure how *zip works but it did 
#col = [*zip(*count)]


# In[5]:


data = pd.DataFrame(
                    col,
                    columns = ['qs1','qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
)
#print(data)
data.to_csv('data.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




