#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.dom.minidom as dom
import pandas as pd
import numpy as np
import sys
from pykalman import KalmanFilter
from math import cos, asin, sqrt, pi


# In[2]:


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


# In[3]:


def get_data(filename):
    file = dom.parse(filename, parser=None, bufsize=None)
    data_trkpt = file.getElementsByTagName('trkpt')
    data_lat_lon = pd.DataFrame(columns=['lat', 'lon'])
    
    for i in range(len(data_trkpt)):
        data_lat_lon.loc[i]=float(data_trkpt[i].attributes['lat'].value) , float(data_trkpt[i].attributes['lon'].value)
    return data_lat_lon


# In[4]:


# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
#The below code for caculation of two points is Edited by Alexander Volkov, Answered by Salvador Dali 


# In[5]:


def distance_between_points(lat1, lon1, lat2, lon2):
    p = pi/180     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) *  (1 - cos((lon2 - lon1) * p)) / 2
    
    return 12742 * asin(sqrt(a))


# In[6]:


distance_for_points = np.vectorize(distance_between_points)


# In[7]:


def distance(dataf):
    dataf['distance']=distance_for_points(dataf['lat'],dataf['lon'],dataf['lat'].shift(periods=-1),dataf['lon'].shift(periods=-1));
    dis = dataf['distance'].sum()
    
    del dataf['distance']
    return dis * 1000


# In[8]:


def smooth(dataf):
    initial_state = dataf.iloc[0]
    #lat lon
    observation_covariance = np.diag([17.5/10**5, 17.5/10**5]) ** 2 # TODO: shouldn't be zero # 20+15/2 = 17.5 use this to divide 10^5
    transition_covariance = np.diag([10/10**5,10/10**5]) ** 2 # TODO: shouldn't be zero
    transition = [[1, 0], [0, 1]] # TODO: shouldn't (all) be zero
    
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(dataf)
    temp = pd.DataFrame(kalman_smoothed,columns=['lat','lon'])
    return temp


# In[9]:


def main():
    points = get_data(sys.argv[1])
    
    print('Unfiltered distance: %0.2f' % (distance(points),))
    smoothed_points = smooth(points)
    
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()

