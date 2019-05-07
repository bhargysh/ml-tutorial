#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys, os

import numpy as np
import pandas as pd
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

from matplotlib import pyplot as plt

# Handy functions
sys.path.append("/app/tools")
from bcatools import *
from tqdm import tqdm_notebook
tqdm_notebook().pandas()


# ## Load the listings dataset
# We will be using a python library called pandas for manipulating these tables of data.
# 
# There are two datasets we have to work with:
# 1. A set of properties sold over the last year that are within 40Km of the Melbourne CBD.
# 2. A set of the locations of all the train stations in victoria.

# In[3]:


df_listings = pd.read_csv('/app/data/sold_listings.csv')


# We can look at a sample of this data.

# In[4]:


df_listings[:5]


# ## Start Calculating Features!
# 
# We know, intuitively, that the price of a house depends on where it is. Instead of treating location at the suburb level, we can obtain a smoother picture by calculating the distance and bearing of the property from the melbourne CBD.
# 
# There are two functions included to do this. The first function __`haversine_distance(loc1, loc2)`__, gives you the _crows fly_ distance between two sets of points, expressed as geo-coordinates _(latitude, longitude)_. The second function __`bearing(loc1, loc2)`__ takes the same arguments, but computes the direction (in degrees) that one faces when travelling from the first location to the second. 
# 
# Here's an example of how to use them.
# 

# In[5]:


geo1 = (-37.852953, 144.724428)
geo2 = (-37.763273, 145.019996)

distance = haversine_distance(geo1, geo2)
brng = bearing(geo1, geo2)

print("The distance between the two points is {:.3f}Km, and the direction of B from A is {:.2f} degrees from North.".format(distance, brng))


# Now you need to calculate this for the properties in the dataset. The location the the melbourne cbd is given here.
# Below is a function that takes a single order pair of (latitude, longitude) and returns the distance from the cbd. Your job is to fill in the blank line to complete the function.

# In[6]:


mel_cbd_loc = (-37.8143349, 144.9624329)

def dist_from_cbd(loc):
    # FILL IN THE BLANK HERE
    dist = haversine_distance(mel_cbd_loc, loc)
    return dist


# Now we can apply this function to every row of our pandas dataframe, selecting only the `latitude` and `longitude` columns and save the result in a new column called `dist from cbd`.

# In[7]:


df_listings['dist_from_cbd'] = df_listings[['latitude','longitude']].apply(dist_from_cbd, axis=1)


# Repeat this for the bearing to add a column called `dir_from_cbd`

# In[ ]:





# In[8]:


# REMOVE THIS CODE
def dir_from_cbd(loc):
    # FILL IN THE BLANK HERE
    brng = bearing(mel_cbd_loc, loc)
    return brng

df_listings['dir_from_cbd'] = df_listings[['latitude','longitude']].apply(dir_from_cbd, axis=1)


# # Find the closest train station
# 
# Now we wish to find which train station is closest to each property.

# ### Load the station dataset
# The file is `'../data/stations.csv'`

# In[9]:


# REMOVE THIS CODE
df_stations = pd.read_csv('/app/data/stations.csv', index_col="station_id")


# In[10]:


df_stations[:5]


# In[ ]:





# We want to add two columns to our `df_listings`:
#    1. `station_id`, which is the `station_id`, which is the index of the station in the `df_stations` dataframe.
#    2. `dist_to_station`, which is the distance to that station.
#    
# You can do this anyway you like. 

# In[11]:


station_locs = df_stations[['latitude', 'longitude']].values
# An array of [[lat, lon],...] for all the stations


def closest_station(loc):
    '''
    Take a sing pair (lat, lon) and returns the (idx, distance) of the closest station in the list `station_locs`
    '''
    distances = np.apply_along_axis(lambda x: haversine_distance(loc,x), 1, station_locs)
    idx = np.argmin(distances)
    dist = distances[idx]
    return idx, dist


listing_locs = df_listings[['latitude', 'longitude']].values
# Array of [[lat, lon]] for the listings

# map over `listing_locs` and apply func `closest_station`
closest_stations = np.apply_along_axis(closest_station, 1, listing_locs)


# In[ ]:


ids, distance = zip(*closest_stations)


# In[ ]:


df_listings["station_id"] = ids
df_listings["dist_to_station"] = distance


# ### Export the dataset to csv

# In[ ]:


export_loc = '/app/data/processed/sold_listings_with_station.csv'
df_listings.to_csv(export_loc, index=False)


# In[ ]:




