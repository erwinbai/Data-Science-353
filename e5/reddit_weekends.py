#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd

#This link help to understand datetime weekdays()
#https://docs.python.org/3/library/datetime.html#datetime.date.weekday
import datetime as dt

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


# In[2]:


# sys.argv[1] = 'reddit-counts.json.gz'
# counts = pd.read_json(sys.argv[1], lines=True)
counts = pd.read_json('reddit-counts.json.gz', lines=True)

#Adapted from 
#https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
#answered by unutbu on Jun 12 2013
#which teaches how to select rows from a df based on value in column 

counts = counts.loc[counts['date'] >= '2012-01-01']  
counts = counts.loc[counts['date'] <= '2013-12-31']  
counts = counts.loc[counts['subreddit'] == 'canada']
counts = counts.reset_index(drop=True)
# print(counts)


# In[3]:


#This is provided by 
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.weekday.html
#which teches weekday() using pandas.DAtetimeIndex.weekday
# s = pd.date_range('2012-01-01','2012-1-31',freq='D').to_series()
# s = s.dt.dayofweek
# print(s)

#https://www.geeksforgeeks.org/python-pandas-series-dt-dayofweek/
#this graph representation of panda.series.dt.dayofweek further helped with the understanding of the above expression s=s.dt.dayofweek


# In[4]:


#dt.dayofweek adapted from above two link
counts['days'] = counts['date'].dt.dayofweek
weekday = counts.loc[counts['days'] <= 4]
weekday = weekday.reset_index(drop=True)
weekend = counts.loc[counts['days'] >= 5]
weekend = weekend.reset_index(drop=True)
#print(weekday)
#print(weekend)


# In[5]:


#below code is from lecture
# x1 = np.random.normal(6.0, 2.5, 17)
# x2 = np.random.normal(5.0, 2.5, 15)

# ttest = stats.ttest_ind(x1, x2).pvalue
# print(x1.mean(), x2.mean())
# print(ttest)
#above 


# In[6]:


ttest = stats.ttest_ind(weekday['comment_count'],weekend['comment_count']).pvalue
#print(ttest)


# In[7]:


stat1 = stats.normaltest(weekday['comment_count']).pvalue
stat2 = stats.normaltest(weekend['comment_count']).pvalue
# print(stat1)
# print(stat2)


# In[8]:


stat3 = stats.levene(weekday['comment_count'],weekend['comment_count']).pvalue
#print(stat3)


# In[9]:


# how to plot histogram
# https://matplotlib.org/3.1.1/gallery/statistics/hist.html
#plt.hist(weekday['comment_count'])


# In[10]:


#plt.hist(weekend['comment_count'])


# In[11]:


#x1 = np.log(weekday['comment_count']) # 0.0004
#x1 = np.exp(weekday['comment_count'])  #1.0091 e-07
x1 = np.sqrt(weekday['comment_count'])  #0.036
#plt.hist(weekday['comment_count'])
stat4 = stats.normaltest(x1).pvalue
#print(stat4)


# In[12]:


#x2 = np.log(weekend['comment_count']) #0.10
#x2 = np.exp(weekend['comment_count']) # 3.744e-74
x2 = np.sqrt(weekend['comment_count']) #0.10
#plt.hist(weekend['comment_count'])
stat5 = stats.normaltest(x2).pvalue
#print(stat5)


# In[13]:


stat6 = stats.levene(weekday['comment_count'],weekend['comment_count']).pvalue
#print(stat6)


# In[14]:


#I recvied help from a fellow class mate name "weibao sun" 
#from our class cmpt 353 for understanding and explaing the function and isocalendar() below
def getisocalendar (x):
        tuple = x.isocalendar()
        return str(tuple[0])+ ' ' +str(tuple[1])  
    # this means we are taking year+month(which we convert to week number by isocalendar)  


# In[15]:


weekday['yearmonth'] = weekday['date'].apply(getisocalendar)
weekend['yearmonth'] = weekend['date'].apply(getisocalendar)
# print(weekday)
x = weekday.groupby(['yearmonth']).agg('mean')
#x = x['comment_count'].sum()
#print(x)
y = weekend.groupby(['yearmonth']).agg('mean')
#y = y['comment_count'].sum()
#print(y)


# In[16]:


stat7 = stats.ttest_ind(x['comment_count'],y['comment_count']).pvalue
stat8 = stats.normaltest(x['comment_count']).pvalue
stat9 = stats.normaltest(y['comment_count']).pvalue
stat10 = stats.levene(x['comment_count'],y['comment_count']).pvalue
# print(stat7)
# print(stat8)
# print(stat9)


# In[17]:


man = stats.mannwhitneyu(weekday['comment_count'],weekend['comment_count']).pvalue
# print(man)


# In[18]:


def main():
    reddit_counts = sys.argv[1]
    #reddit_counts = 'reddit-counts.json.gz'
    # ...

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p= ttest,
        initial_weekday_normality_p= stat1,
        initial_weekend_normality_p= stat2,
        initial_levene_p=stat3,
        transformed_weekday_normality_p=stat4,
        transformed_weekend_normality_p=stat5,
        transformed_levene_p=stat6,
        weekly_weekday_normality_p=stat8,
        weekly_weekend_normality_p=stat9,
        weekly_levene_p=stat10,
        weekly_ttest_p=stat7,
        utest_p=man,
    ))


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




