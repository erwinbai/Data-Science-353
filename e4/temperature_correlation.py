#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
from math import cos, asin, sqrt, pi
import matplotlib.pyplot as plt


# In[2]:


input1 = sys.argv[1]
input2 = sys.argv[2]
#input1 = 'station.json.gz'
#input2 = 'city_data.csv'
stations = pd.read_json(input1, lines=True)
stations['avg_tmax'] = stations['avg_tmax']/10  #since avgtmax is 10C, we need to divid by 10
stations  


# In[3]:


city_data = pd.read_csv(input2)
city_data = city_data.dropna() #remove NaN
city_data


# In[4]:


city_data['area'] = city_data['area']/10**6  #convert to km^2
city_data = city_data[city_data['area'] <= 10000].reset_index(drop=True) # order index 
city_data['density'] = city_data['population']/city_data['area']  # density is population divide area


# In[5]:


#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
#The below code for caculation of two points is Edited by Alexander Volkov, Answered by Salvador Dali 
#used from excersie 3
def distance_between_points(lat1, lon1, lat2, lon2):
    p = pi/180     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) *  (1 - cos((lon2 - lon1) * p)) / 2
    
    return 12742 * asin(sqrt(a))
distance_for_points = np.vectorize(distance_between_points)


# In[6]:


def distance(city, stations):
    #https://www.geeksforgeeks.org/python-pandas-series/
    #this site helped with understanding using pd.Series
    list = (distance_for_points(city['latitude'],city['longitude'],stations['latitude'],stations['longitude']))
    DataFrame = pd.Series(list)
    return DataFrame

def best_tmax(city, stations): #the best avg_tmax
    DataFrame = distance(city, stations)
    #uses distance to find best temperture
    best_tmax = stations['avg_tmax'][DataFrame.idxmin()]
    return best_tmax
#print(stations)


# In[7]:


city_data['best_tmax'] = city_data.apply(best_tmax, axis = 1, stations = stations)
city_data


# In[8]:


plt.plot( city_data['best_tmax'], city_data['density'], 'b.')
plt.title('Temperature vs Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)')  #degree C
plt.ylabel('Population Density (people/km\u00b2)')  #km^2
plt.savefig(sys.argv[3])


# In[ ]:




