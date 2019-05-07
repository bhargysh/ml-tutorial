#!/usr/bin/env python
# coding: utf-8


######## IMPORTING LIBRARIES ############
import numpy as np
import pandas as pd
import sys, os
from functools import reduce

import matplotlib.pyplot as plt
from math import pi, sqrt, sin, cos, atan, atan2 

# # Plotly
# import plotly.offline as py
# import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)

# Handy functions
sys.path.append("/app/tools")
from bcatools import *




############### READING DATA ################

# Redshift doesn't give headers so we export them as a separate file
with open('/app/data/raw/sold_listings_vic_headers.csv') as f:
    columns = [line.replace('\n','') for line in f.readlines()]
    
df_listings_raw = pd.read_csv("/app/data/raw/sold_listings_vic.csv", delimiter='|', header=-1)
df_listings_raw.columns = columns

columns_to_keep = ['listing_id',
 'property_id',
 'sold_price',
 'sold_date',
 'property_type',
 'garages',
 'carports',
 'bedrooms',
 'bathrooms',
 'toilets',
 'suburb',
 'latitude',
 'longitude',
 'year_built',
 'landsize',
 'buildingsize']

df_list = df_listings_raw[columns_to_keep]

############### FIX NAN VALUES #################

# #### Fill NaNs:
# - 'garages',
# - 'carports',
# - 'bedrooms',
# - 'bathrooms',
# - 'toilets'

# make a list of the subset of columns that need NaNs set to 0.

fill_na_cols = ['garages', 'carports', 'bedrooms', 'bathrooms', 'toilets']

# Use the df.fillna() function to do the operation on this subset of columns. 
# You can see the function's docs by hittin SHIFT + TAB when the cursor is inside the function parentheses. 
df_filled = df_list[fill_na_cols].fillna(value=0)

# use df.drop() to drop the columns from df_listings, then user df.join to add in the filled columns.
# you'll need to use the keyword argument 'axis=1' to specify that you are dropping columns and not rows.
df_list_filled = df_list.drop(fill_na_cols, axis=1).join(df_filled)

# #### DROP NaNs
df_list_clean = df_list_filled.dropna()

# num dropped
initial, _ = df_listings_raw.shape
final , _ = df_list_clean.shape

print("Initial: {}. Final: {}. Difference: {}.".format(initial, final, final-initial))




############## FIND THE DISTANCE FROM THE CBD ###############

# #### Distance from CBD
# 
# Used only to filter out listings further than 40Km aay. Do not inlcude in the output dataset.

mel_cbd_loc = (-37.8143349, 144.9624329)

listing_locations = df_list_clean[['latitude', 'longitude']].values

dist = np.apply_along_axis(lambda loc: haversine_distance(loc, mel_cbd_loc), 1, listing_locations)

# make a list of True/False that tells us which listings to keep
keep = dist<=40

# We can then use it as an index for the pandas dataframe to select only those listings.
df_list_filtered = df_list_clean[keep].copy()

# num dropped
initial, _ = df_listings_raw.shape
final , _ = df_list_filtered.shape

print("Initial: {}. Final: {}. Difference: {}.".format(initial, final, final-initial))

show(df_list_filtered)

# ### Check the distibution is uniform
brng = np.apply_along_axis(lambda loc: bearing(mel_cbd_loc, loc), 1, listing_locations)
pts = np.array(list(zip(dist[keep], brng[keep])))

r, theta = pts[:,0], pts[:,1],
theta_rad = list(map(lambda x: -x*(pi/180), theta))

plt.clf()
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='polar', )
ax.set_theta_zero_location('N')
c = ax.scatter(theta_rad, r, s=8, cmap='hsv', alpha=0.75,)

plt.show()



############## EXPORT DATA ################

# ### Export the dataset for use by the grads
df_list_filtered.to_csv('/app/data/sold_listings.csv', header=True, index=False)
