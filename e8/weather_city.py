#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


labelled = pd.read_csv('monthly-data-labelled.csv')
unlabelled = pd.read_csv('monthly-data-unlabelled.csv')
#print(labelled)


# In[3]:


#print(unlabelled)


# In[4]:


#X = labelled[['tmax-01', 'tmax-02', 'tmx-03']]
#At first I was doing something like this, but after few selecting column, it seems completely stupid
#So I did some research
#https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/#targetText=To%20delete%20rows%20and%20columns,the%20need%20for%20'axis'.


# In[5]:


#This site give me the idea and way that we can use numeric indexing to select comlumn using column numbers
X = labelled.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]]
# Yes,this is a stupid way after my friends told me I dont need to select the column if i could just remove year and city
# print(X)


# In[6]:


y = labelled['city']
# print(y)


# In[7]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, train_size=0.75)


# In[8]:


# from sklearn.ensemble import VotingClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# The below code is from lecture
# model = VotingClassifier([
#     ('nb', GaussianNB()),
#     ('knn', KNeighborsClassifier(5)),
#     ('svm', SVC(kernel='linear', C=0.1)),
#     ('tree1', DecisionTreeClassifier(max_depth=4)),
#     ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
# ])
# model.fit(X_train, y_train)
# predictions = model.score(X_valid, y_valid)
# print(predictions)

#I am not using voting classifier since i cannot tell which one is the best


# In[9]:


# bayes = make_pipeline(
#     StandardScaler(),
#     GaussianNB()
#     )

# bayes.fit(X_train,y_train)
# print(bayes.score(X_valid,y_valid))


# In[10]:


# knn = make_pipeline(
#     StandardScaler(),
#     KNeighborsClassifier(n_neighbors=25)
#     )

# knn.fit(X_train,y_train)
# print(knn.score(X_valid,y_valid))


# In[11]:


rf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=400,max_depth=8)
    )

rf.fit(X_train,y_train)

print('The model that best fit is random forest classifier with:')
print(rf.score(X_valid,y_valid))


# In[12]:


data = labelled.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]]
predictions = rf.predict(data)
#print(predictions)


# In[13]:


pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)


# In[ ]:




