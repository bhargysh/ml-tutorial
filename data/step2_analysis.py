#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
# Handy functions
sys.path.append("/app/tools")
import bcatools as bt
import bcaplots as bp

import numpy as np
import pandas as pd
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', bt.num_formatter(3))

from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterExponent
import seaborn as sns


# In[ ]:


export_loc = '/app/data/processed/sold_listings_with_station.csv'
df_listings = pd.read_csv(export_loc)


# ## Get some basic stats for the columns in the dataset
# 
# mean, median, etc.
# pandas has a neat function `.describe()` that does this for you.

# In[ ]:


df_listings.describe()


# ## check for skews  and biases in the data
# ### start with distance from CBD
# Plot a histogram of distance from Melbourne CBD. you can use a function we've made for you called __`bp.hist(series, n_bins=?)`__

# In[ ]:


fig = bp.hist(df_listings.dist_from_cbd,
              n_bins=100, 
              plot_title='Distance from CBD',
              xlabel='Km',
              ylabel='count')
plt.show(fig)
print("total = {}".format(df_listings.shape[0]))


# ## what about sold price?

# In[ ]:


bp.hist(df_listings.sold_price, n_bins=200)
plt.show()


# This is heavily skewed. we can adjust for this skewness by taking the logarithm of this data.

# In[ ]:


df_listings['log_price'] = np.log(df_listings.sold_price)
# make a function to convert the axes values back to the real numbers
plot_ticker = lambda x: bp.millify(np.exp(x))

bp.hist(df_listings.log_price, n_bins=200, tick_func=plot_ticker)
plt.show()


# Explore the relationship between price and distance from cbd. you can use our prettified scatter plot __`bp.facet_scatter()`__

# In[ ]:


bp.facet_scatter(df_listings, 
              xcol='dist_from_cbd', 
              ycol='log_price', 
              facet='property_type', 
              trendline=True,
              tick_func=plot_ticker
            )
plt.show()


# what about against a categorical variable like bedrooms. how des price correlate to that?

# In[ ]:


df_listings.boxplot(column="log_price", by="bedrooms", figsize=(10,6))
plt.show()


# In[ ]:


df_listings[['bedrooms','listing_id']].groupby("bedrooms").count().plot(kind='bar', figsize=(10,6))
plt.show()
df_listings[['bedrooms','listing_id']].groupby("bedrooms").count()


# In[ ]:


r = df_listings.dist_from_cbd
theta = df_listings.dir_from_cbd

# convert theta to radians
# North in geocoord is 0, and the angle goes in a clockwise direction, but, for radians,
# 0 points East, and the angle goes in a counter-clockwise direction. 
# you'll need to convert the geoocords into radians.

theta_rad = list(map(lambda x: np.pi/2-x*(np.pi/180), theta))

x = r*np.cos(theta_rad)
y = r*np.sin(theta_rad)

bp.hex_plot(x, y)


# # Hexplot of price

# In[ ]:


df = df_listings[(df_listings.dist_from_cbd > 3) & (df_listings.dist_from_cbd < 8)]
r = df.dist_from_cbd
theta = df.dir_from_cbd

# convert theta to radians
# North in geocoord is 0, and the angle goes in a clockwise direction, but, for radians,
# 0 points East, and the angle goes in a counter-clockwise direction. 
# you'll need to convert the geoocords into radians.

theta_rad = list(map(lambda x: np.pi/2-x*(np.pi/180), theta))

x = r*np.cos(theta_rad)
y = r*np.sin(theta_rad)

bp.hex_plot(x, y, z=df.dist_to_station, gridsize=80)


# In[ ]:




