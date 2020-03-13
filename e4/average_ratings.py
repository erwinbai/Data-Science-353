#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import sys
import difflib


# In[13]:


def remove(element):
    #rstrip() idea is from https://www.programiz.com/python-programming/methods/string/rstrip
    return element.rstrip()  #helps with removing '|n' behind moive title
vec_remove =  np.vectorize(remove)

#input1 = 'movie_list.txt'
#input2 = 'movie_ratings.csv'

input1 = sys.argv[1]
input2 = sys.argv[2]


# In[14]:


data_list = open(input1).readlines()
data = pd.DataFrame(vec_remove(data_list), columns = ['title'])  # applying removing for '\n'
#data


# In[15]:


def good_enough(rating):
#Reference from given site on instructions for 
#https://docs.python.org/3/library/difflib.html #difflib.get_close_matches
    tmp = difflib.get_close_matches(rating,data['title'])
    # Create a list with close enough matches and return the first one tmp[0] as our title
    if len(tmp)==0:
        return None # cant match with anything
    else:
        return tmp[0]
    
close_match = np.vectorize(good_enough)


# In[16]:


data_rating = pd.read_csv(input2)
data_rating


# In[17]:


data_rating['title'] = close_match(data_rating['title'])
data_rating


# In[22]:


output = data_rating.groupby(['title']).mean()
# output1 = np.mean(output)  # this did not work since numpy operation are not valid with grouby. 
#system suggested using .groupby(...).mean() instead
#How to round using website belowin format of numpy.around(a,decimals=0)
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html
final_output = np.around(output, decimals=2)
final_output


# In[21]:


final_output.to_csv(sys.argv[3], header=False, index=True)


# In[ ]:





# In[ ]:




