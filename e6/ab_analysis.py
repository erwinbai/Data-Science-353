#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import scipy.stats as stats


# In[2]:


#input1 = 'searches.json'
input1 = sys.argv[1]
df = pd.stations = pd.read_json(input1, orient = 'records', lines=True)
# print(df)


# In[3]:


#odd uid show on a new search box
odd = df.loc[df['uid']%2 == 1].reset_index(drop=True) #348 uid
even = df.loc[df['uid']%2 == 0].reset_index(drop=True) #333 uid
# print(odd)
# print(even)


# In[4]:


odd_search_more = odd.loc[odd['search_count'] > 0].reset_index(drop=True)  #98 uid search more than once 
#print(odd_search_more)
odd_search_zero = odd.loc[odd['search_count'] == 0].reset_index(drop=True) # 250 uid
#print(odd_search_zero)


# In[5]:


even_search_more = even.loc[even['search_count'] > 0].reset_index(drop=True)  #111 uid search more than once 
#print(even_search_more)
even_search_zero = even.loc[even['search_count'] == 0].reset_index(drop=True) # 222 uid
#print(even_search_zero)


# In[6]:


#I was searching on how to count number of row and this site give me a hint
#https://thispointer.com/pandas-count-rows-in-a-dataframe-all-or-those-only-that-satisfy-a-condition/
#I realized i could just use .shape[0] to find row number
odd_rows_more = odd_search_more.shape[0]
odd_rows_zero = odd_search_zero.shape[0]
even_rows_more = even_search_more.shape[0]
even_rows_zero = even_search_zero.shape[0]
# print(odd_rows_more)
# print(odd_rows_zero)
# print(even_rows_more)
# print(even_rows_zero)


# In[7]:


contingency =[[even_rows_more,even_rows_zero],[odd_rows_more,odd_rows_zero]]
chi2, p_value1, dof, expected = stats.chi2_contingency(contingency)

p_value2 = stats.mannwhitneyu(even['search_count'],odd['search_count']).pvalue
# print(chi2)
print(p_value1)
# print(dof)
# print(expected)
print(p_value2) 


# In[8]:


#---------------------------------------------------------------Instructor


# In[9]:


odd_in = odd.loc[odd['is_instructor'] == True]
even_in = even.loc[even['is_instructor'] == True]

odd_search_more_in = odd_in.loc[odd_in['search_count'] > 0].reset_index(drop=True)
odd_search_zero_in = odd_in.loc[odd_in['search_count'] == 0].reset_index(drop=True)
even_search_more_in = even_in.loc[even_in['search_count'] > 0].reset_index(drop=True)
even_search_zero_in = even_in.loc[even_in['search_count'] == 0].reset_index(drop=True)

#print(odd_search_more_in)
odd_rows_more_in = odd_search_more_in.shape[0]
odd_rows_zero_in = odd_search_zero_in.shape[0]
even_rows_more_in = even_search_more_in.shape[0]
even_rows_zero_in = even_search_zero_in.shape[0]
#print(odd_rows_more_in)


# In[10]:


contingency2 =[[even_rows_more_in,even_rows_zero_in],[odd_rows_more_in,odd_rows_zero_in]]
chi2, p_value3, dof, expected = stats.chi2_contingency(contingency2)

p_value4 = stats.mannwhitneyu(even_in['search_count'],odd_in['search_count']).pvalue
# print(chi2)
print(p_value3)
# print(dof)
# print(expected)
print(p_value4) 


# In[11]:


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)
def main():
    searchdata_file = sys.argv[1]

    # ...

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p_value1,
        more_searches_p=p_value2,
        more_instr_p=p_value3,
        more_instr_searches_p=p_value4,
    ))


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




